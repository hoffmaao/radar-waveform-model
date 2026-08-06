"""Property grids and discretisation helpers for the FDTD solvers.

The solvers use the staggered convention of Irving and Knight (2006): the
electrical property arrays (``eps``, ``mu``, ``sig``) live on a grid with half
the spacing of the electromagnetic field arrays and must therefore have an odd
number of rows and columns.  A field grid of ``nx`` by ``nz`` nodes corresponds
to a property grid of ``2*nx-1`` by ``2*nz-1`` nodes.

Arrays are indexed ``[i, j]`` with ``i`` horizontal (x) and ``j`` vertical (z,
positive downwards).
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .constants import EPS0, MU0

__all__ = [
    "PropertyGrid",
    "max_spatial_step",
    "max_time_step",
    "pad_grid",
    "regrid",
]

ArrayLike = Union[float, np.ndarray]


def _as_field(value: ArrayLike, shape: Tuple[int, int]) -> np.ndarray:
    """Broadcast a scalar or array to a full property-grid array."""
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f"expected array of shape {shape}, got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=float)


@dataclass
class PropertyGrid:
    """Electrical properties sampled on the half-spaced property grid.

    Parameters
    ----------
    x, z : ndarray
        Property-grid coordinates (m), uniformly spaced with the half field
        spacing and of odd length.  ``z`` increases downwards.
    eps : dict
        Relative permittivity per principal axis, with keys ``'xx'``, ``'yy'``
        and ``'zz'``.  Supplying a single array through :meth:`isotropic` fills
        all three with the same values.  Off-diagonal terms are not supported:
        the fabric principal axes are assumed aligned with the model axes,
        which covers vertical-single-maximum, girdle and horizontally
        anisotropic ice.
    mu : ndarray
        Relative permeability (1.0 everywhere for ice and rock).
    sig : dict
        Conductivity (S/m) per principal axis, same keys as ``eps``.
    """

    x: np.ndarray
    z: np.ndarray
    eps: Dict[str, np.ndarray]
    mu: np.ndarray
    sig: Dict[str, np.ndarray]
    metadata: dict = field(default_factory=dict)

    # -- construction ----------------------------------------------------

    @classmethod
    def uniform(cls, xlim, zlim, dx, dz=None, eps=1.0, mu=1.0, sig=0.0):
        """Build a uniform grid covering ``xlim`` by ``zlim`` at field step ``dx``.

        The returned grid stores properties at ``dx/2`` spacing.  ``xlim`` and
        ``zlim`` are inclusive and are snapped outwards to a whole number of
        field cells.
        """
        dz = dx if dz is None else dz
        nx = int(np.ceil((xlim[1] - xlim[0]) / dx)) + 1
        nz = int(np.ceil((zlim[1] - zlim[0]) / dz)) + 1
        x = xlim[0] + np.arange(2 * nx - 1) * (dx / 2.0)
        z = zlim[0] + np.arange(2 * nz - 1) * (dz / 2.0)
        shape = (x.size, z.size)
        return cls(
            x=x,
            z=z,
            eps=cls._axis_dict(eps, shape),
            mu=_as_field(mu, shape),
            sig=cls._axis_dict(sig, shape),
        )

    @staticmethod
    def _axis_dict(value, shape) -> Dict[str, np.ndarray]:
        if isinstance(value, dict):
            missing = {"xx", "yy", "zz"} - set(value)
            if missing:
                raise ValueError(f"missing principal components: {sorted(missing)}")
            return {k: _as_field(value[k], shape) for k in ("xx", "yy", "zz")}
        arr = _as_field(value, shape)
        return {k: arr.copy() for k in ("xx", "yy", "zz")}

    # -- geometry --------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.x.size, self.z.size)

    @property
    def dx(self) -> float:
        """Field-grid spacing in x (twice the property spacing)."""
        return 2.0 * (self.x[1] - self.x[0])

    @property
    def dz(self) -> float:
        return 2.0 * (self.z[1] - self.z[0])

    @property
    def nx(self) -> int:
        """Number of field nodes in x."""
        return (self.x.size + 1) // 2

    @property
    def nz(self) -> int:
        return (self.z.size + 1) // 2

    def validate(self) -> None:
        nxp, nzp = self.shape
        if nxp % 2 != 1 or nzp % 2 != 1:
            raise ValueError("property grids need an odd number of rows and columns")
        for name in ("xx", "yy", "zz"):
            if self.eps[name].shape != (nxp, nzp):
                raise ValueError(f"eps['{name}'] has the wrong shape")
            if self.sig[name].shape != (nxp, nzp):
                raise ValueError(f"sig['{name}'] has the wrong shape")
            # Tolerance because interpolating a field that is exactly 1 (air)
            # lands a few cells on 0.999999999999999.
            if np.any(self.eps[name] < 1.0 - 1e-9):
                raise ValueError(f"eps['{name}'] must be >= 1 (relative permittivity)")
        if self.mu.shape != (nxp, nzp):
            raise ValueError("mu has the wrong shape")
        if not np.all(np.isfinite(self.mu)) or np.any(self.mu <= 0):
            raise ValueError("mu must be finite and positive")

    # -- convenience -----------------------------------------------------

    @property
    def eps_min(self) -> float:
        return min(float(np.min(self.eps[k])) for k in ("xx", "yy", "zz"))

    @property
    def eps_max(self) -> float:
        return max(float(np.max(self.eps[k])) for k in ("xx", "yy", "zz"))

    @property
    def mu_min(self) -> float:
        return float(np.min(self.mu))

    @property
    def mu_max(self) -> float:
        return float(np.max(self.mu))

    def with_properties(self, eps=None, mu=None, sig=None) -> "PropertyGrid":
        """Return a copy with some properties replaced."""
        shape = self.shape
        return replace(
            self,
            eps=self.eps if eps is None else self._axis_dict(eps, shape),
            mu=self.mu if mu is None else _as_field(mu, shape),
            sig=self.sig if sig is None else self._axis_dict(sig, shape),
            metadata=dict(self.metadata),
        )

    def padded(self, npml: int, layers: Optional[int] = None) -> "PropertyGrid":
        """Extend the grid outwards so the PML sits outside the physical domain.

        The absorbing layer occupies ``2*npml + 2`` property cells at each edge;
        by default that many cells are added, with edge properties replicated
        outwards so the PML never straddles a material contrast.
        """
        n = 2 * npml + 2 if layers is None else int(layers)
        hx = self.x[1] - self.x[0]
        hz = self.z[1] - self.z[0]
        x2 = np.concatenate(
            [self.x[0] - hx * np.arange(n, 0, -1), self.x, self.x[-1] + hx * np.arange(1, n + 1)]
        )
        z2 = np.concatenate(
            [self.z[0] - hz * np.arange(n, 0, -1), self.z, self.z[-1] + hz * np.arange(1, n + 1)]
        )
        pad = ((n, n), (n, n))
        return PropertyGrid(
            x=x2,
            z=z2,
            eps={k: np.pad(v, pad, mode="edge") for k, v in self.eps.items()},
            mu=np.pad(self.mu, pad, mode="edge"),
            sig={k: np.pad(v, pad, mode="edge") for k, v in self.sig.items()},
            metadata={**self.metadata, "npml_pad": n},
        )

    def mesh(self):
        """Return ``(X, Z)`` meshgrids matching the property-array indexing."""
        return np.meshgrid(self.x, self.z, indexing="ij")

    def extent(self):
        """Extent tuple for ``imshow`` of a transposed property array."""
        return (self.x[0], self.x[-1], self.z[-1], self.z[0])


def max_spatial_step(eps_max, mu_max, srcpulse, t, thres=0.02):
    """Largest field spacing that keeps numerical dispersion acceptable.

    Five nodes per shortest significant wavelength is the Irving and Knight
    (2006) rule of thumb for their O(2,4) scheme.

    Returns
    -------
    dxmax, wlmin, fmax
        Maximum spacing (m), minimum wavelength (m) and the highest frequency
        (Hz) whose amplitude exceeds ``thres`` times the spectral peak.
    """
    t = np.asarray(t, dtype=float)
    eps = eps_max * EPS0
    mu = mu_max * MU0

    n = int(2 ** np.ceil(np.log2(len(srcpulse))))
    spec = np.abs(np.fft.rfft(srcpulse, n))
    spec /= np.max(spec)
    freqs = np.fft.rfftfreq(n, t[1] - t[0])

    fmax = float(freqs[np.max(np.nonzero(spec >= thres)[0])])
    wlmin = 1.0 / (fmax * np.sqrt(eps * mu))
    return wlmin / 5.0, wlmin, fmax


def max_time_step(eps_min, mu_min, dx, dz):
    """Largest stable time step (s) for the O(2,4) scheme.

    ``eps_min`` and ``mu_min`` are relative values; the 6/7 factor is the
    stability margin of the fourth-order spatial stencil.
    """
    return (6.0 / 7.0) * np.sqrt(eps_min * EPS0 * mu_min * MU0 / (1.0 / dx**2 + 1.0 / dz**2))


def pad_grid(A, x, z, n):
    """Replicate-pad a property array and extend its coordinate vectors."""
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    x2 = np.concatenate([x[0] - dx * np.arange(n, 0, -1), x, x[-1] + dx * np.arange(1, n + 1)])
    z2 = np.concatenate([z[0] - dz * np.arange(n, 0, -1), z, z[-1] + dz * np.arange(1, n + 1)])
    return np.pad(np.asarray(A, dtype=float), ((n, n), (n, n)), mode="edge"), x2, z2


def regrid(A, x, z, x2, z2, method="linear"):
    """Resample a property array onto new coordinate vectors.

    Replaces ``scipy.interpolate.interp2d``, which was removed in SciPy 1.14.
    Queries outside the input range are clamped to the nearest edge rather
    than returning NaN, which is what padding a property grid expects.
    """
    A = np.asarray(A, dtype=float)
    interp = RegularGridInterpolator(
        (np.asarray(x, dtype=float), np.asarray(z, dtype=float)),
        A,
        method=method,
        bounds_error=False,
        fill_value=None,
    )
    X2, Z2 = np.meshgrid(np.asarray(x2, dtype=float), np.asarray(z2, dtype=float), indexing="ij")
    return interp((X2, Z2))
