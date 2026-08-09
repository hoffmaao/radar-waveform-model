"""Export a radarwave model to gprMax, as an independent check on the physics.

This module exports a *scalar* map.  gprMax does model diagonal anisotropy -- its
volumetric object commands take three material identifiers, one per field
component -- but that arrives through ``#box`` and friends, while the bulk import
path used here, ``#geometry_objects_read``, stores one material index per cell
and so can only carry one permittivity per cell.  See
:mod:`anisotropic_gprmax` for the anisotropic comparison, which builds its
geometry from object commands instead.

Exporting a scalar map is still an exact statement of the problem radarwave
solved for one eigenpolarisation, because the out-of-plane solver also carries a
single permittivity per cell: the wave it propagates *is* an isotropic problem
with ``eps = eps_component(x, z)``.  So the two traces are directly comparable.

What this validates: geometric spreading, reflection coefficients, traveltime
through a heterogeneous column, and the absolute amplitude scale.  What it
cannot: the anisotropy, since both codes are handed the same scalar map.

gprMax is never imported here.  This module only writes its input files, so the
repository never depends on it.
"""

from pathlib import Path

import numpy as np

__all__ = ["write_case"]


def _bin(values, levels):
    """Bin on quantiles, not on a linear span.

    Linear bins are useless here.  An ice model runs from air at 1.0 to ice at
    about 3.19, but every contrast that produces a reflection -- the layering,
    the fabric jump -- lives inside a window of roughly 0.03 near the top of
    that range.  Spreading bins evenly over the span puts the whole
    stratigraphy into two or three of them and hands the other code a
    homogeneous half-space.  Quantile edges put the resolution where the values
    actually are.
    """
    values = np.asarray(values, dtype=float)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-18:
        return np.zeros(values.shape, dtype=np.int64), np.array([lo])
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, levels + 1)))
    if edges.size < 2:
        return np.zeros(values.shape, dtype=np.int64), np.array([lo])
    idx = np.clip(np.digitize(values, edges[1:-1]), 0, edges.size - 2)
    # Represent each bin by the mean of the values that landed in it, so a bin
    # holding a single physical value reproduces it exactly.
    reps = np.array([values[idx == k].mean() if np.any(idx == k)
                     else 0.5 * (edges[k] + edges[k + 1])
                     for k in range(edges.size - 1)])
    return idx, reps


def _quantise(eps, sig, eps_levels=4096, sig_levels=64):
    """Bin the property maps onto a finite material list.

    gprMax addresses materials by index, so a continuous map has to become a
    lookup.  The firn gradient spans a wide range of permittivity and layer
    contrasts of a fraction of a percent have to survive the binning, so the
    level count is the resolution of the comparison rather than a cosmetic
    choice.

    Only the (eps, sigma) pairs that actually occur are emitted.  Taking the
    full product instead costs thousands of unused materials, and gprMax builds
    and prints every one of them.

    The level counts look extravagant, and are not.  Quantiles allocate bins by
    cell count, and most cells sit in deep ice spanning ~0.03 in eps -- so at
    256 levels the firn gradient, which is where the density layers live, is
    left with bins wider than the ~0.01 layer contrasts, and the layers
    disappear from the export.  Thin conductivity layers lose the same way in
    sigma.  4096/64 leaves both with several bins per contrast.
    """
    e_idx, e_vals = _bin(eps, eps_levels)
    s_idx, s_vals = _bin(sig, sig_levels)
    flat = e_idx * len(s_vals) + s_idx
    used, inverse = np.unique(flat, return_inverse=True)
    combos = [(float(e_vals[u // len(s_vals)]), float(s_vals[u % len(s_vals)]))
              for u in used]
    return inverse.reshape(eps.shape).astype(np.int16), combos


def write_case(outdir, name, *, eps, sig, dx, dz, src_xz, rec_xz, pulse, dt,
               n_steps, eps_levels=4096, sig_levels=64):
    """Write ``name.in``, ``name_geom.h5``, ``name_materials.txt``, ``name_wave.txt``.

    ``eps`` and ``sig`` are the (nx, nz) maps the TM solver would use, i.e. the
    component that eigenpolarisation sees.  Coordinates are radarwave's: x
    across, z downward from the top of the grid.

    Returns the path of the ``.in`` file.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    eps = np.asarray(eps, dtype=float)
    sig = np.asarray(sig, dtype=float)
    nx, nz = eps.shape

    idx, combos = _quantise(eps, sig, eps_levels, sig_levels)
    # gprMax is 3-D; one cell in y makes it 2-D, and a hertzian dipole along y
    # then acts as a line source in the invariant direction -- the same
    # out-of-plane Ey the TM solver carries.
    data = idx.reshape(nx, 1, nz)

    import h5py

    geom = outdir / f"{name}_geom.h5"
    with h5py.File(geom, "w") as f:
        f.attrs["dx_dy_dz"] = (dx, dx, dz)
        f.create_dataset("data", data=data)

    mats = outdir / f"{name}_materials.txt"
    with mats.open("w") as f:
        for k, (e, s) in enumerate(combos):
            f.write(f"#material: {e:.9g} {s:.9g} 1 0 m{k}\n")

    # The time column is not optional in practice.  gprMax picks its own
    # Courant-limited step, and without an explicit time vector it assumes the
    # samples land on *its* grid -- stretching the pulse by the ratio of the two
    # steps and shifting its centre frequency by the same factor.  Supplying the
    # vector makes it resample onto the same waveform in physical time.
    # Pad past the window as well.  gprMax runs ceil(window / its own dt)
    # iterations, so its last sample time can sit just beyond the supplied
    # vector and the interpolation raises rather than extrapolating.
    wave = outdir / f"{name}_wave.txt"
    amps = np.concatenate([np.asarray(pulse, dtype=float),
                           np.zeros(max(16, len(pulse) // 20))])
    times = np.arange(len(amps)) * dt
    with wave.open("w") as f:
        f.write("time src\n")
        for tt, a in zip(times, amps):
            f.write(f"{tt:.9e} {a:.9e}\n")

    sx, sz = src_xz
    rx, rz = rec_xz
    inp = outdir / f"{name}.in"
    with inp.open("w") as f:
        f.write(f"#title: {name}\n")
        f.write(f"#domain: {nx * dx:.9g} {dx:.9g} {nz * dz:.9g}\n")
        f.write(f"#dx_dy_dz: {dx:.9g} {dx:.9g} {dz:.9g}\n")
        f.write(f"#time_window: {n_steps * dt:.9g}\n")
        # gprMax defaults to a 10-cell PML.  With metre-scale cells and an air
        # layer that is a fraction of the 5 m air wavelength thick, that
        # reflects visibly off the top edge; 20 cells pushes the artefact well
        # below the coda being compared.  Per-face form, because the
        # single-argument form also applies to the one-cell-thick y axis and
        # gprMax refuses a PML thicker than the domain.
        f.write("#pml_cells: 20 0 20 20 0 20\n")
        f.write(f"#excitation_file: {wave.name}\n")
        f.write(f"#geometry_objects_read: 0 0 0 {geom.name} {mats.name}\n")
        # A dipole along the invariant axis is the 2-D line source.
        f.write(f"#hertzian_dipole: y {sx:.9g} 0 {sz:.9g} src\n")
        f.write(f"#rx: {rx:.9g} 0 {rz:.9g}\n")
    return inp
