"""Check radarwave's anisotropy against gprMax, which models it natively.

gprMax gives every volumetric object three material identifiers, one per field
component, which is exactly the diagonal permittivity tensor radarwave carries.
So it can be handed the fabric directly, with none of the substitution
radarwave's out-of-plane solver needs.

The case is deliberately the simplest one that still exercises the physics: a
homogeneous, strongly anisotropic slab, a source at the top and a receiver
directly below it.  A wave going straight down with E along x travels at
c/sqrt(eps_xx) and with E along y at c/sqrt(eps_yy), so the split between the
two arrivals is known in closed form and both codes can be scored against it
rather than only against each other.

At nadir both x and y are principal axes perpendicular to the ray, so
radarwave's trick of feeding eps_xx to the out-of-plane solver should be exact
here.  Comparing that against its own TE mode and against gprMax localises
where the disagreement seen off-nadir actually comes from.

Usage: anisotropic_gprmax.py <gprmax-python> <gprmax-repo>
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from radarwave import (FDTD2D, PropertyGrid, blackharrispulse,   # noqa: E402
                       envelope_peak_time, max_time_step)
from radarwave.constants import C0                                # noqa: E402
from radarwave.ice import fabric_permittivity                     # noqa: E402
from radarwave.polarimetry import analytic, subsample_lag         # noqa: E402

OUT = REPO / "figures" / "validation"
FC = 60e6
DEPTH = 150.0                     # receiver directly below the source
LAM = np.array([0.97, 0.01, 0.02])  # a strong single maximum along x


def _peak_time(t, trace):
    env = np.abs(analytic(trace))
    k = int(np.argmax(env))
    return float(t[k]), float(env[k])


def main(gpr_python, gpr_repo):
    eps = fabric_permittivity(LAM)
    print(f"lam = {tuple(LAM)}")
    print(f"eps = ({eps[0]:.5f}, {eps[1]:.5f}, {eps[2]:.5f})")
    v = C0 / np.sqrt(eps)
    print(f"v   = ({v[0]/1e6:.3f}, {v[1]/1e6:.3f}, {v[2]/1e6:.3f}) m/us")
    t_x, t_y = DEPTH / v[0], DEPTH / v[1]
    print(f"one-way to {DEPTH:.0f} m: E||x {t_x*1e9:.3f} ns, E||y {t_y*1e9:.3f} ns")
    print(f"analytic split = {(t_x - t_y)*1e9:+.3f} ns\n")

    dx = 0.15
    xlim, zlim = (-45.0, 45.0), (-3.0, DEPTH + 45.0)
    grid = PropertyGrid.uniform(xlim, zlim, dx, eps=1.0)
    shape = grid.eps["xx"].shape
    model = grid.with_properties(eps={c: np.full(shape, eps[k])
                                      for k, c in enumerate(("xx", "yy", "zz"))})
    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, 1.25e-6, dt)
    pulse = blackharrispulse(FC, t)
    t_wave = envelope_peak_time(pulse, t)
    src = np.array([[0.0, 0.0]])
    rec = np.array([[0.0, DEPTH]])
    print(f"grid {shape}, dx {dx} m, dt {dt*1e12:.2f} ps, {len(t)} steps\n")

    got = {}
    # Out-of-plane: the solver's Ey sees eps_yy directly.
    got["ours Ey (TM)"] = FDTD2D(model, dt, npml=12, mode="TM").run(
        src, pulse, rec, outstep=1).gather[:, 0, 0]
    # ex03's trick: hand the out-of-plane solver eps_xx and call it E||x.
    sub = model.with_properties(
        eps={"xx": model.eps["xx"], "yy": model.eps["xx"], "zz": model.eps["zz"]})
    got["ours Ex (TM sub)"] = FDTD2D(sub, dt, npml=12, mode="TM").run(
        src, pulse, rec, outstep=1).gather[:, 0, 0]
    # The genuine in-plane solver, no substitution.
    got["ours Ex (TE)"] = FDTD2D(model, dt, npml=12, mode="TE").run(
        src, pulse, rec, outstep=1, source_component="Ex").gather[:, 0, 0]

    # ---- gprMax, anisotropy declared natively -------------------------------
    d = OUT / "gprmax"
    d.mkdir(parents=True, exist_ok=True)
    # The property grid samples at HALF-cell resolution (see
    # compare_ex03_gprmax.py): sizing the domain from its shape stretches it
    # 2x per axis.  Count the cell centres instead.
    xc, zc = grid.x[1::2], grid.z[1::2]
    assert abs((xc[1] - xc[0]) - dx) < 1e-9, "cell-centre stride is not dx"
    nx, nz = xc.size, zc.size
    wave = d / "aniso_wave.txt"
    # Padded past the window: gprMax's own dt makes its last sample time land
    # just outside the supplied vector otherwise, and it raises rather than
    # extrapolating.
    amps = np.concatenate([pulse, np.zeros(max(16, len(pulse) // 20))])
    with wave.open("w") as f:
        f.write("time src\n")
        for tt, a in zip(np.arange(len(amps)) * dt, amps):
            f.write(f"{tt:.9e} {a:.9e}\n")

    for comp in ("y", "x"):
        # gprMax names its 2-D mode after the single-cell axis, and the dipole
        # must lie along that axis: thin in y is "TMy" and carries Ey.  To get
        # E along x the domain is made thin in x instead ("TMx"), the plane
        # becomes y-z, and the wave still runs down z.  E is then along a
        # principal axis for every ray in the plane, so each run samples
        # exactly one tensor component -- no mode mixing at all.
        inp = d / f"aniso_{comp}.in"
        lat = nx * dx        # lateral extent of the propagation plane
        sx, sz = 0.0 - xlim[0], 0.0 - zlim[0]
        with inp.open("w") as f:
            f.write(f"#title: aniso_{comp}\n")
            if comp == "y":
                f.write(f"#domain: {lat:.9g} {dx:.9g} {nz*dx:.9g}\n")
            else:
                f.write(f"#domain: {dx:.9g} {lat:.9g} {nz*dx:.9g}\n")
            f.write(f"#dx_dy_dz: {dx:.9g} {dx:.9g} {dx:.9g}\n")
            f.write(f"#time_window: {len(t)*dt:.9g}\n")
            for k, c in enumerate("xyz"):
                f.write(f"#material: {eps[k]:.9g} 0 1 0 f{c}\n")
            # Three identifiers = diagonal anisotropy, one per field component.
            if comp == "y":
                f.write(f"#box: 0 0 0 {lat:.9g} {dx:.9g} {nz*dx:.9g} fx fy fz\n")
                f.write(f"#excitation_file: {wave.name}\n")
                f.write(f"#hertzian_dipole: y {sx:.9g} 0 {sz:.9g} src\n")
                f.write(f"#rx: {sx:.9g} 0 {DEPTH - zlim[0]:.9g}\n")
            else:
                f.write(f"#box: 0 0 0 {dx:.9g} {lat:.9g} {nz*dx:.9g} fx fy fz\n")
                f.write(f"#excitation_file: {wave.name}\n")
                f.write(f"#hertzian_dipole: x 0 {sx:.9g} {sz:.9g} src\n")
                f.write(f"#rx: 0 {sx:.9g} {DEPTH - zlim[0]:.9g}\n")
        print(f"running gprMax ({comp}) ...")
        r = subprocess.run([gpr_python, "-m", "gprMax", inp.name], cwd=d,
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(gpr_repo)})
        if r.returncode != 0:
            print(r.stdout[-2500:]); print(r.stderr[-2500:])
            raise SystemExit(f"gprMax {comp} failed")
        import h5py
        with h5py.File(inp.with_suffix(".out"), "r") as f:
            dt_g = float(f.attrs["dt"])
            field = "Ey" if comp == "y" else "Ex"
            tr = np.array(f[f"rxs/rx1/{field}"])
        got[f"gprMax E{comp}"] = np.interp(t, np.arange(tr.size) * dt_g, tr)

    print(f"\n{'trace':22s} {'peak (ns)':>11s} {'vs analytic':>12s}")
    expect = {"ours Ey (TM)": t_y, "ours Ex (TM sub)": t_x, "ours Ex (TE)": t_x,
              "gprMax Ey": t_y, "gprMax Ex": t_x}
    peaks = {}
    for k, tr in got.items():
        tp, _ = _peak_time(t, tr)
        peaks[k] = tp - t_wave
        print(f"{k:22s} {peaks[k]*1e9:11.3f} {(peaks[k]-expect[k])*1e9:+12.3f}")

    print(f"\n{'split (E||x - E||y)':22s} {'ns':>11s}")
    print(f"{'analytic':22s} {(t_x-t_y)*1e9:11.3f}")
    for a, b, lbl in (("ours Ex (TM sub)", "ours Ey (TM)", "radarwave TM-sub"),
                      ("ours Ex (TE)", "ours Ey (TM)", "radarwave TE"),
                      ("gprMax Ex", "gprMax Ey", "gprMax")):
        print(f"{lbl:22s} {subsample_lag(got[b], got[a], dt)*1e9:11.3f}")

    np.savez_compressed(OUT / "anisotropic_comparison.npz", t=t, t_wave=t_wave,
                        eps=eps, depth=DEPTH, **{k.replace(" ", "_"): v
                                                 for k, v in got.items()})
    print("\nsaved", OUT / "anisotropic_comparison.npz")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
