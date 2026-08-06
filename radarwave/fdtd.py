"""Two-dimensional FDTD solvers for radar wave propagation.

This is a vectorised, anisotropy-aware rewrite of the O(2,4) convolutional-PML
scheme of Irving and Knight (2006).  Two polarisations are available and, for
vertical propagation through ice whose fabric principal axes are aligned with
the model axes, they are exactly the two eigenpolarisations a quad-polarimetric
sounder transmits:

``mode="TM"``
    Out-of-plane electric field ``Ey`` with ``Hx``, ``Hz``.  The wave samples
    only ``eps['yy']``.

``mode="TE"``
    In-plane electric field ``Ex``, ``Ez`` with ``Hy``.  ``Ex`` samples
    ``eps['xx']`` and ``Ez`` samples ``eps['zz']``.

Differences from the original MATLAB and its first Python port:

* update coefficients are pre-sampled onto each field component's own grid, so
  the time loop is pure slice arithmetic instead of ``np.ix_`` fancy indexing
  (roughly an order of magnitude faster, and it no longer allocates per step);
* the PML memory variables are updated over contiguous slabs;
* per-axis permittivity and conductivity, which is what makes the fabric
  simulations possible;
* the receiver-gather and output-time indexing bugs of the earlier port are
  fixed, and the field coordinate vectors are built from integer strides
  instead of ``np.arange`` on floats (which silently dropped a node).

Arrays are indexed ``[i, j]`` with ``i`` horizontal and ``j`` vertical
(downwards).  Transpose before plotting.
"""

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .constants import EPS0, MU0
from .grid import PropertyGrid

__all__ = ["FDTD2D", "Result", "run_common_offset"]

# Default convolutional-PML profile parameters (Roden and Gedney, 2000).
#
# The original code shipped with ``kmax = 5``.  Real coordinate stretching
# divides the damping conductivity by kappa, and with kappa reaching ~5.9 at
# the outer edge that throws away most of the absorption: measured boundary
# reflection was -23.5 dB and, tellingly, did not improve at all when the layer
# was made thicker.  Dropping to kmax = 1 recovers 22 dB (-45 dB at npml = 10,
# -51 dB at npml = 20) and the layer once again improves with thickness.
# Stretching only helps against evanescent waves hugging the boundary, which is
# not the regime these ice-sounding models run in; ``pml_kmax`` is still
# exposed for callers who need it.
_PML_M = 4  # polynomial grading exponent
_PML_KMAX = 1.0  # maximum real stretching factor
_PML_ALPHA = 0.0  # CFS frequency-shift parameter (S/m); measured to not help here
_PML_SIGMA_SCALE = 0.5  # multiplies Gedney's optimal sigma_max


@dataclass
class Result:
    """Output of a single- or multi-shot FDTD run."""

    gather: np.ndarray  # (n_out, n_rec, n_src) recorded field
    t: np.ndarray  # (n_out,) output times (s)
    src: np.ndarray  # (n_src, 2) discretised source positions (m)
    rec: np.ndarray  # (n_rec, 2) discretised receiver positions (m)
    snapshots: Optional[np.ndarray] = None  # (n_snap, nxs, nzs) float32
    snapshot_times: Optional[np.ndarray] = None  # (n_snap,)
    snapshot_x: Optional[np.ndarray] = None
    snapshot_z: Optional[np.ndarray] = None
    component: str = "Ey"

    @property
    def common_offset(self) -> np.ndarray:
        """Trace ``k`` from shot ``k`` -- the common-offset section ``(n_out, n_src)``.

        A gather that already carries one receiver per shot (what
        :func:`run_common_offset` returns) is that section already.
        """
        if self.gather.shape[1] == 1:
            return self.gather[:, 0, :]
        n = min(self.gather.shape[1], self.gather.shape[2])
        return np.stack([self.gather[:, k, k] for k in range(n)], axis=1)


def _isect(a: slice, b: slice) -> slice:
    start = max(a.start, b.start)
    stop = min(a.stop, b.stop)
    return slice(start, max(start, stop))


def _pml_slabs(n_field: int, offset: int, k_lo: int, k_hi: int):
    """Field-index slabs whose property index falls inside the PML.

    ``offset`` is the property index of field node 0 (property index is
    ``offset + 2*i``).  ``k_lo``/``k_hi`` are the inner edges of the low and
    high PML regions in property-index space.
    """
    k = offset + 2 * np.arange(n_field)
    lo = np.nonzero(k <= k_lo)[0]
    hi = np.nonzero(k >= k_hi)[0]
    s_lo = slice(0, int(lo[-1]) + 1) if lo.size else slice(0, 0)
    s_hi = slice(int(hi[0]), n_field) if hi.size else slice(0, 0)
    s_in = slice(s_lo.stop, s_hi.start if hi.size else n_field)
    return s_lo, s_hi, s_in


class FDTD2D:
    """Fourth-order-in-space, second-order-in-time FDTD engine with CPML.

    Parameters
    ----------
    grid : PropertyGrid
        Electrical properties, already padded for the absorbing boundary (see
        :meth:`PropertyGrid.padded`).
    dt : float
        Time step (s).  Must not exceed :func:`radarwave.grid.max_time_step`.
    npml : int
        Number of absorbing cells; the layer is ``2*npml + 2`` property cells
        thick at each edge.
    mode : {'TM', 'TE'}
        Polarisation, see the module docstring.
    """

    def __init__(
        self,
        grid: PropertyGrid,
        dt: float,
        npml: int = 10,
        mode: str = "TM",
        *,
        pml_kmax: float = _PML_KMAX,
        pml_alpha: float = _PML_ALPHA,
        pml_sigma_scale: float = _PML_SIGMA_SCALE,
    ):
        grid.validate()
        self.pml_kmax = float(pml_kmax)
        self.pml_alpha = float(pml_alpha)
        self.pml_sigma_scale = float(pml_sigma_scale)
        mode = mode.upper()
        if mode not in ("TM", "TE"):
            raise ValueError("mode must be 'TM' or 'TE'")
        nxp, nzp = grid.shape
        if npml >= nxp / 2 or npml >= nzp / 2:
            raise ValueError("too many PML layers for this grid")
        dt_max = (6.0 / 7.0) * np.sqrt(
            grid.eps_min * EPS0 * grid.mu_min * MU0 / (1.0 / grid.dx**2 + 1.0 / grid.dz**2)
        )
        if dt > dt_max:
            raise ValueError(
                f"dt = {dt:.4g} s exceeds the stability limit {dt_max:.4g} s; "
                "reduce dt or coarsen the grid"
            )

        self.grid = grid
        self.dt = float(dt)
        self.npml = int(npml)
        self.mode = mode
        self.nx = grid.nx
        self.nz = grid.nz
        self.dx = grid.dx
        self.dz = grid.dz

        # Field-component coordinate vectors (integer strides on the property
        # grid; the original port used np.arange on floats and lost a node).
        self.x_half = grid.x[1::2]  # property index 2i+1
        self.x_full = grid.x[0::2]  # property index 2i
        self.z_half = grid.z[1::2]
        self.z_full = grid.z[0::2]

        self._build_pml()
        self._build_coefficients()
        self._build_slices()

    # -- setup -----------------------------------------------------------

    def _build_pml(self):
        grid = self.grid
        nxp, nzp = grid.shape
        npml = self.npml

        # A single PML profile shared by all field components, derived from the
        # orientation-averaged permittivity.  Ice anisotropy is ~1 percent, far
        # below the level at which a per-component profile would matter, and a
        # shared profile keeps the boundary reflectionless for both modes.
        eps_ref = (grid.eps["xx"] + grid.eps["yy"] + grid.eps["zz"]) / 3.0

        # Inner edges of the PML in property-index space (0-based).
        self.k_lo = 2 * npml + 1
        self.k_hi = nxp - 2 * npml - 2
        self.l_lo = 2 * npml + 1
        self.l_hi = nzp - 2 * npml - 2

        xdel = np.zeros((nxp, nzp))
        k = np.arange(0, self.k_lo + 1)
        xdel[k, :] = ((self.k_lo - k) / (2.0 * npml))[:, None]
        k = np.arange(self.k_hi, nxp)
        xdel[k, :] = ((k - self.k_hi) / (2.0 * npml))[:, None]

        zdel = np.zeros((nxp, nzp))
        l = np.arange(0, self.l_lo + 1)
        zdel[:, l] = ((self.l_lo - l) / (2.0 * npml))[None, :]
        l = np.arange(self.l_hi, nzp)
        zdel[:, l] = ((l - self.l_hi) / (2.0 * npml))[None, :]

        scale = self.pml_sigma_scale
        sigx_max = scale * (_PML_M + 1) / (150 * np.pi * np.sqrt(eps_ref) * self.dx)
        sigz_max = scale * (_PML_M + 1) / (150 * np.pi * np.sqrt(eps_ref) * self.dz)
        sigx = sigx_max * xdel**_PML_M
        sigz = sigz_max * zdel**_PML_M
        self.Kx = 1.0 + (self.pml_kmax - 1.0) * xdel**_PML_M
        self.Kz = 1.0 + (self.pml_kmax - 1.0) * zdel**_PML_M

        # CFS alpha grades the other way from sigma: largest at the inner edge.
        alpha_x = self.pml_alpha * np.clip(1.0 - xdel, 0.0, None) * (xdel > 0)
        alpha_z = self.pml_alpha * np.clip(1.0 - zdel, 0.0, None) * (zdel > 0)

        dt = self.dt
        self.Bx = np.exp(-(sigx / self.Kx + alpha_x) * (dt / EPS0))
        self.Bz = np.exp(-(sigz / self.Kz + alpha_z) * (dt / EPS0))
        self.Ax = (
            sigx / (sigx * self.Kx + self.Kx**2 * alpha_x + 1e-20) * (self.Bx - 1.0)
        ) / (24.0 * self.dx)
        self.Az = (
            sigz / (sigz * self.Kz + self.Kz**2 * alpha_z + 1e-20) * (self.Bz - 1.0)
        ) / (24.0 * self.dz)

    def _electric_coefficients(self, axis, sx, sz):
        """Sample the E-field update coefficients for one principal axis."""
        grid = self.grid
        eps = grid.eps[axis][sx, sz] * EPS0
        sig = grid.sig[axis][sx, sz]
        dt = self.dt
        denom = 1.0 + dt * sig / (2.0 * eps)
        return {
            "Ca": (1.0 - dt * sig / (2.0 * eps)) / denom,
            "Cbx": (dt / eps) / (denom * 24.0 * self.dx * self.Kx[sx, sz]),
            "Cbz": (dt / eps) / (denom * 24.0 * self.dz * self.Kz[sx, sz]),
            "Cc": (dt / eps) / denom,
        }

    def _magnetic_coefficients(self, sx, sz):
        mu = self.grid.mu[sx, sz] * MU0
        dt = self.dt
        return {
            "Dbx": dt / (mu * self.Kx[sx, sz] * 24.0 * self.dx),
            "Dbz": dt / (mu * self.Kz[sx, sz] * 24.0 * self.dz),
            "Dc": dt / mu,
        }

    def _build_coefficients(self):
        half, full = np.s_[1::2], np.s_[0::2]
        if self.mode == "TM":
            # Ey at (2i+1, 2j+1); Hx at (2i+1, 2j); Hz at (2i, 2j+1)
            self.cE = self._electric_coefficients("yy", half, half)
            self.cHx = self._magnetic_coefficients(half, full)
            self.cHz = self._magnetic_coefficients(full, half)
            self.pmlE = (self.Bx[half, half], self.Ax[half, half],
                         self.Bz[half, half], self.Az[half, half])
            self.pmlHx = (self.Bz[half, full], self.Az[half, full])
            self.pmlHz = (self.Bx[full, half], self.Ax[full, half])
        else:
            # Ex at (2i+1, 2j); Ez at (2i, 2j+1); Hy at (2i+1, 2j+1)
            self.cEx = self._electric_coefficients("xx", half, full)
            self.cEz = self._electric_coefficients("zz", full, half)
            self.cHy = self._magnetic_coefficients(half, half)
            self.pmlEx = (self.Bz[half, full], self.Az[half, full])
            self.pmlEz = (self.Bx[full, half], self.Ax[full, half])
            self.pmlHy = (self.Bx[half, half], self.Ax[half, half],
                          self.Bz[half, half], self.Az[half, half])

    def _build_slices(self):
        nx, nz = self.nx, self.nz

        # Update regions (translated from the MATLAB 1-based index ranges).
        if self.mode == "TM":
            self.rHx = (slice(1, nx - 2), slice(2, nz - 2))
            self.rHz = (slice(2, nx - 2), slice(1, nz - 2))
            self.rE = (slice(1, nx - 2), slice(1, nz - 2))
            comps = {
                "Hx": (self.rHx, (nx - 1, nz), 1, 0),
                "Hz": (self.rHz, (nx, nz - 1), 0, 1),
                "E": (self.rE, (nx - 1, nz - 1), 1, 1),
            }
        else:
            self.rHy = (slice(1, nx - 2), slice(1, nz - 2))
            self.rEx = (slice(1, nx - 2), slice(2, nz - 2))
            self.rEz = (slice(2, nx - 2), slice(1, nz - 2))
            comps = {
                "Hy": (self.rHy, (nx - 1, nz - 1), 1, 1),
                "Ex": (self.rEx, (nx - 1, nz), 1, 0),
                "Ez": (self.rEz, (nx, nz - 1), 0, 1),
            }

        self._slabs = {}
        for name, (region, shape, off_i, off_j) in comps.items():
            xi_lo, xi_hi, xi_in = _pml_slabs(shape[0], off_i, self.k_lo, self.k_hi)
            zj_lo, zj_hi, zj_in = _pml_slabs(shape[1], off_j, self.l_lo, self.l_hi)
            ri, rj = region
            self._slabs[name] = {
                # x-side slabs span the full update region in j
                "x": [(_isect(s, ri), rj) for s in (xi_lo, xi_hi)],
                # z-side slabs use only the interior in i, so corners are not
                # counted twice
                "z": [(_isect(xi_in, ri), _isect(s, rj)) for s in (zj_lo, zj_hi)],
            }

    # -- helpers ---------------------------------------------------------

    def component_axes(self, component: str):
        """``(x, z)`` coordinate vectors for a named field component."""
        table = {
            "Ey": (self.x_half, self.z_half),
            "Hx": (self.x_half, self.z_full),
            "Hz": (self.x_full, self.z_half),
            "Ex": (self.x_half, self.z_full),
            "Ez": (self.x_full, self.z_half),
            "Hy": (self.x_half, self.z_half),
        }
        if component not in table:
            raise ValueError(f"unknown component {component!r}")
        return table[component]

    def _node_index(self, positions, component):
        xs, zs = self.component_axes(component)
        pos = np.atleast_2d(np.asarray(positions, dtype=float))
        i = np.argmin(np.abs(xs[None, :] - pos[:, 0:1]), axis=1)
        j = np.argmin(np.abs(zs[None, :] - pos[:, 1:2]), axis=1)
        return i, j, np.column_stack([xs[i], zs[j]])

    # -- the time loop ---------------------------------------------------

    def run(
        self,
        srcloc,
        srcpulse,
        recloc=None,
        *,
        outstep: int = 1,
        snapshot_every: Optional[int] = None,
        snapshot_stride: int = 1,
        snapshot_component: Optional[str] = None,
        simultaneous: bool = False,
        source_component: str = "Ex",
        receiver_component: Optional[str] = None,
        progress: Optional[int] = None,
    ) -> Result:
        """Propagate ``srcpulse`` from each source and record at the receivers.

        Parameters
        ----------
        srcloc : (n_src, 2) array
            Source ``(x, z)`` positions (m).
        srcpulse : (n_t,) array
            Source time series; its length sets the number of iterations.
        recloc : (n_rec, 2) array, optional
            Receiver positions.  Defaults to the source positions.
        outstep : int
            Record a sample every ``outstep`` iterations.
        snapshot_every : int, optional
            Store the wavefield every this many iterations (first shot only).
        snapshot_stride : int
            Spatial decimation applied to stored snapshots.
        simultaneous : bool
            Fire every source in the same run (a phased array / plane wave)
            instead of one shot per source.
        source_component : {'Ex', 'Ez'}
            Which in-plane component carries the source in TE mode.
        progress : int, optional
            Print a progress line every this many iterations.

        Returns
        -------
        Result
        """
        srcpulse = np.asarray(srcpulse, dtype=float)
        srcloc = np.atleast_2d(np.asarray(srcloc, dtype=float))[:, :2]
        recloc = srcloc if recloc is None else np.atleast_2d(np.asarray(recloc, dtype=float))[:, :2]

        if self.mode == "TM":
            src_comp = rec_comp = "Ey"
            snap_comp = snapshot_component or "Ey"
        else:
            src_comp = source_component
            rec_comp = receiver_component or source_component
            snap_comp = snapshot_component or source_component

        si, sj, src_xz = self._node_index(srcloc, src_comp)
        ri, rj, rec_xz = self._node_index(recloc, rec_comp)

        numit = srcpulse.size
        n_out = (numit - 1) // outstep + 1
        n_shots = 1 if simultaneous else srcloc.shape[0]
        gather = np.zeros((n_out, rec_xz.shape[0], n_shots))
        t_out = np.arange(n_out) * (outstep * self.dt)

        snaps, snap_t = [], []
        for shot in range(n_shots):
            idx = (si, sj) if simultaneous else (si[shot : shot + 1], sj[shot : shot + 1])
            want_snap = snapshot_every is not None and shot == 0
            s_out, s_t = self._time_loop(
                idx,
                srcpulse,
                (ri, rj),
                gather[:, :, shot],
                src_comp,
                rec_comp,
                outstep,
                snapshot_every if want_snap else None,
                snapshot_stride,
                snap_comp,
                progress,
                shot,
                n_shots,
            )
            if want_snap:
                snaps, snap_t = s_out, s_t

        snap_arr = np.asarray(snaps, dtype=np.float32) if snaps else None
        sx = sz = None
        if snap_arr is not None:
            ax, az = self.component_axes(snap_comp)
            sx, sz = ax[::snapshot_stride], az[::snapshot_stride]

        return Result(
            gather=gather,
            t=t_out,
            src=src_xz,
            rec=rec_xz,
            snapshots=snap_arr,
            snapshot_times=np.asarray(snap_t) if snap_t else None,
            snapshot_x=sx,
            snapshot_z=sz,
            component=rec_comp,
        )

    def _time_loop(
        self,
        src_idx,
        srcpulse,
        rec_idx,
        gather_out,
        src_comp,
        rec_comp,
        outstep,
        snapshot_every,
        snapshot_stride,
        snap_comp,
        progress,
        shot,
        n_shots,
    ):
        nx, nz = self.nx, self.nz
        dt = self.dt
        numit = srcpulse.size
        si, sj = src_idx
        ri, rj = rec_idx
        snaps, snap_t = [], []

        if self.mode == "TM":
            E = np.zeros((nx - 1, nz - 1))
            Hx = np.zeros((nx - 1, nz))
            Hz = np.zeros((nx, nz - 1))
            dEdz = np.zeros((nx - 1, nz))
            dEdx = np.zeros((nx, nz - 1))
            dHxdz = np.zeros((nx - 1, nz - 1))
            dHzdx = np.zeros((nx - 1, nz - 1))
            PHx = np.zeros((nx - 1, nz))
            PHz = np.zeros((nx, nz - 1))
            PEx = np.zeros((nx - 1, nz - 1))
            PEz = np.zeros((nx - 1, nz - 1))
            fields = {"Ey": E, "Hx": Hx, "Hz": Hz}
        else:
            Hy = np.zeros((nx - 1, nz - 1))
            Ex = np.zeros((nx - 1, nz))
            Ez = np.zeros((nx, nz - 1))
            dExdz = np.zeros((nx - 1, nz - 1))
            dEzdx = np.zeros((nx - 1, nz - 1))
            dHydz = np.zeros((nx - 1, nz))
            dHydx = np.zeros((nx, nz - 1))
            PHyx = np.zeros((nx - 1, nz - 1))
            PHyz = np.zeros((nx - 1, nz - 1))
            PEx = np.zeros((nx - 1, nz))
            PEz = np.zeros((nx, nz - 1))
            fields = {"Hy": Hy, "Ex": Ex, "Ez": Ez}

        src_field = fields[src_comp]
        rec_field = fields[rec_comp]
        snap_field = fields[snap_comp]

        for it in range(numit):
            if self.mode == "TM":
                self._step_tm(E, Hx, Hz, dEdz, dEdx, dHxdz, dHzdx, PHx, PHz, PEx, PEz)
            else:
                self._step_te(Hy, Ex, Ez, dExdz, dEzdx, dHydz, dHydx, PHyx, PHyz, PEx, PEz)

            src_field[si, sj] += srcpulse[it]

            if it % outstep == 0:
                gather_out[it // outstep, :] = rec_field[ri, rj]
            if snapshot_every is not None and it % snapshot_every == 0:
                snaps.append(
                    snap_field[::snapshot_stride, ::snapshot_stride].astype(np.float32).copy()
                )
                snap_t.append(it * dt)
            if progress and it % progress == 0:
                print(
                    f"  shot {shot + 1}/{n_shots}  it {it + 1}/{numit}  "
                    f"t = {it * dt * 1e9:8.1f} ns",
                    flush=True,
                )

        return snaps, snap_t

    # -- single time step ------------------------------------------------

    def _step_tm(self, E, Hx, Hz, dEdz, dEdx, dHxdz, dHzdx, PHx, PHz, PEx, PEz):
        nx, nz = self.nx, self.nz

        # --- Hx from dEy/dz ------------------------------------------------
        i, j = self.rHx
        d = dEdz[i, j]
        np.subtract(E[i, 2 : nz - 2], E[i, 1 : nz - 3], out=d)
        d *= 27.0
        d -= E[i, 3 : nz - 1]
        d += E[i, 0 : nz - 4]
        Hx[i, j] -= self.cHx["Dbz"][i, j] * d

        Bz, Az = self.pmlHx
        for si, sj in self._slabs["Hx"]["x"] + self._slabs["Hx"]["z"]:
            PHx[si, sj] *= Bz[si, sj]
            PHx[si, sj] += Az[si, sj] * dEdz[si, sj]
            Hx[si, sj] -= self.cHx["Dc"][si, sj] * PHx[si, sj]

        # --- Hz from dEy/dx ------------------------------------------------
        i, j = self.rHz
        d = dEdx[i, j]
        np.subtract(E[2 : nx - 2, j], E[1 : nx - 3, j], out=d)
        d *= 27.0
        d -= E[3 : nx - 1, j]
        d += E[0 : nx - 4, j]
        Hz[i, j] += self.cHz["Dbx"][i, j] * d

        Bx, Ax = self.pmlHz
        for si, sj in self._slabs["Hz"]["x"] + self._slabs["Hz"]["z"]:
            PHz[si, sj] *= Bx[si, sj]
            PHz[si, sj] += Ax[si, sj] * dEdx[si, sj]
            Hz[si, sj] += self.cHz["Dc"][si, sj] * PHz[si, sj]

        # --- Ey from curl H ------------------------------------------------
        i, j = self.rE
        dz_ = dHxdz[i, j]
        np.subtract(Hx[i, 2 : nz - 1], Hx[i, 1 : nz - 2], out=dz_)
        dz_ *= 27.0
        dz_ -= Hx[i, 3:nz]
        dz_ += Hx[i, 0 : nz - 3]

        dx_ = dHzdx[i, j]
        np.subtract(Hz[2 : nx - 1, j], Hz[1 : nx - 2, j], out=dx_)
        dx_ *= 27.0
        dx_ -= Hz[3:nx, j]
        dx_ += Hz[0 : nx - 3, j]

        E[i, j] *= self.cE["Ca"][i, j]
        E[i, j] += self.cE["Cbx"][i, j] * dx_ - self.cE["Cbz"][i, j] * dz_

        Bx, Ax, Bz, Az = self.pmlE
        for si, sj in self._slabs["E"]["x"] + self._slabs["E"]["z"]:
            PEx[si, sj] *= Bx[si, sj]
            PEx[si, sj] += Ax[si, sj] * dHzdx[si, sj]
            PEz[si, sj] *= Bz[si, sj]
            PEz[si, sj] += Az[si, sj] * dHxdz[si, sj]
            E[si, sj] += self.cE["Cc"][si, sj] * (PEx[si, sj] - PEz[si, sj])

    def _step_te(self, Hy, Ex, Ez, dExdz, dEzdx, dHydz, dHydx, PHyx, PHyz, PEx, PEz):
        nx, nz = self.nx, self.nz

        # --- Hy from dEx/dz - dEz/dx ---------------------------------------
        i, j = self.rHy
        dz_ = dExdz[i, j]
        np.subtract(Ex[i, 2 : nz - 1], Ex[i, 1 : nz - 2], out=dz_)
        dz_ *= 27.0
        dz_ -= Ex[i, 3:nz]
        dz_ += Ex[i, 0 : nz - 3]

        dx_ = dEzdx[i, j]
        np.subtract(Ez[2 : nx - 1, j], Ez[1 : nx - 2, j], out=dx_)
        dx_ *= 27.0
        dx_ -= Ez[3:nx, j]
        dx_ += Ez[0 : nx - 3, j]

        Hy[i, j] += self.cHy["Dbz"][i, j] * dz_ - self.cHy["Dbx"][i, j] * dx_

        Bx, Ax, Bz, Az = self.pmlHy
        for si, sj in self._slabs["Hy"]["x"] + self._slabs["Hy"]["z"]:
            PHyx[si, sj] *= Bx[si, sj]
            PHyx[si, sj] += Ax[si, sj] * dEzdx[si, sj]
            PHyz[si, sj] *= Bz[si, sj]
            PHyz[si, sj] += Az[si, sj] * dExdz[si, sj]
            Hy[si, sj] += self.cHy["Dc"][si, sj] * (PHyz[si, sj] - PHyx[si, sj])

        # --- Ex from dHy/dz -------------------------------------------------
        # -Hy(j+1) + 27 Hy(j) - 27 Hy(j-1) + Hy(j-2)
        i, j = self.rEx
        d = dHydz[i, j]
        np.subtract(Hy[i, 2 : nz - 2], Hy[i, 1 : nz - 3], out=d)
        d *= 27.0
        d -= Hy[i, 3 : nz - 1]
        d += Hy[i, 0 : nz - 4]

        Ex[i, j] *= self.cEx["Ca"][i, j]
        Ex[i, j] += self.cEx["Cbz"][i, j] * d

        Bz, Az = self.pmlEx
        for si, sj in self._slabs["Ex"]["x"] + self._slabs["Ex"]["z"]:
            PEx[si, sj] *= Bz[si, sj]
            PEx[si, sj] += Az[si, sj] * dHydz[si, sj]
            Ex[si, sj] += self.cEx["Cc"][si, sj] * PEx[si, sj]

        # --- Ez from dHy/dx -------------------------------------------------
        # -Hy(i+1) + 27 Hy(i) - 27 Hy(i-1) + Hy(i-2)
        i, j = self.rEz
        d = dHydx[i, j]
        np.subtract(Hy[2 : nx - 2, j], Hy[1 : nx - 3, j], out=d)
        d *= 27.0
        d -= Hy[3 : nx - 1, j]
        d += Hy[0 : nx - 4, j]

        Ez[i, j] *= self.cEz["Ca"][i, j]
        Ez[i, j] -= self.cEz["Cbx"][i, j] * d

        Bx, Ax = self.pmlEz
        for si, sj in self._slabs["Ez"]["x"] + self._slabs["Ez"]["z"]:
            PEz[si, sj] *= Bx[si, sj]
            PEz[si, sj] += Ax[si, sj] * dHydx[si, sj]
            Ez[si, sj] -= self.cEz["Cc"][si, sj] * PEz[si, sj]

    # -- reporting -------------------------------------------------------

    def summary(self) -> str:
        nxp, nzp = self.grid.shape
        return (
            f"FDTD2D({self.mode}) {self.nx} x {self.nz} field nodes "
            f"({nxp} x {nzp} properties), dx = {self.dx:.4g} m, dz = {self.dz:.4g} m, "
            f"dt = {self.dt * 1e9:.4g} ns, npml = {self.npml}"
        )


def run_common_offset(grid, dt, srcloc, recloc, srcpulse, *, npml=10, mode="TM",
                      outstep=1, processes=None, **kwargs):
    """Run one shot per source position, optionally across several processes.

    Returns a :class:`Result` whose ``gather`` has shape ``(n_out, 1, n_src)``:
    shot ``k`` recorded at receiver ``k`` and nothing else, i.e. the
    common-offset section, reachable as :attr:`Result.common_offset`.  The
    layout is the same whether or not the run was parallelised -- a worker
    process can only record its own shot's receiver, so the serial path is
    collapsed to match rather than returning every receiver for every shot,
    which would silently make ``gather[:, 0, :]`` a common-*receiver* section.
    """
    import multiprocessing as mp

    srcloc = np.atleast_2d(np.asarray(srcloc, dtype=float))
    recloc = np.atleast_2d(np.asarray(recloc, dtype=float))
    n = srcloc.shape[0]
    if recloc.shape[0] != n:
        raise ValueError(
            f"need one receiver per shot: {n} sources but {recloc.shape[0]} receivers"
        )

    if processes in (None, 1) or n == 1:
        sim = FDTD2D(grid, dt, npml=npml, mode=mode)
        res = sim.run(srcloc, srcpulse, recloc, outstep=outstep, **kwargs)
        return replace(res, gather=res.common_offset[:, None, :])

    ctx = mp.get_context("fork")
    with ctx.Pool(
        processes=processes,
        initializer=_worker_init,
        initargs=(grid, dt, npml, mode, srcpulse, outstep, kwargs),
    ) as pool:
        out = pool.map(_worker_shot, [(srcloc[k], recloc[k]) for k in range(n)])

    traces = np.stack([o[0] for o in out], axis=-1)  # (n_out, 1, n_src)
    return Result(
        gather=traces,
        t=out[0][1],
        src=np.stack([o[2] for o in out]),
        rec=np.stack([o[3] for o in out]),
        component=out[0][4],
    )


_WORKER = {}


def _worker_init(grid, dt, npml, mode, srcpulse, outstep, kwargs):
    _WORKER["sim"] = FDTD2D(grid, dt, npml=npml, mode=mode)
    _WORKER["pulse"] = srcpulse
    _WORKER["outstep"] = outstep
    _WORKER["kwargs"] = {k: v for k, v in kwargs.items() if not k.startswith("snapshot")}


def _worker_shot(args):
    src, rec = args
    sim = _WORKER["sim"]
    res = sim.run(
        src[None, :],
        _WORKER["pulse"],
        rec[None, :],
        outstep=_WORKER["outstep"],
        **_WORKER["kwargs"],
    )
    return res.gather[:, :, 0], res.t, res.src[0], res.rec[0], res.component
