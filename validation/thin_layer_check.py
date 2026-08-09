"""Which code gets a thin permittivity layer right?  Both against exact theory.

One 2 m layer of eps 1.83 in a uniform eps 1.80 background -- the ex03 firn
numbers -- 40 m below a monostatic antenna.  The echo is isolated by
differencing against a layer-free twin, and referenced to the twin's direct
wave recorded at the image distance (80 m), which by the image-source identity
makes the ratio exactly the layer's plane-wave reflection response.  That
response is computed exactly for the actual pulse with a transfer matrix, so
each code gets a verdict, not just a disagreement.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from radarwave import (FDTD2D, PropertyGrid, blackharrispulse,   # noqa: E402
                       max_time_step)
from radarwave.constants import C0                                # noqa: E402
from radarwave.polarimetry import analytic                        # noqa: E402

OUT = REPO / "figures" / "validation"
FC = 60e6
EPS_BG, EPS_LAY = 1.80, 1.83
Z_LAY, THICK, DEPTH_REF = 40.0, 2.0, 80.0


def exact_ratio(pulse, dt):
    """Envelope-peak ratio of (reflected pulse) to (incident pulse), exact."""
    n_fft = 8 * len(pulse)
    f = np.fft.rfftfreq(n_fft, dt)
    r = np.zeros(f.size, dtype=complex)
    n1, n2 = np.sqrt(EPS_BG + 0j), np.sqrt(EPS_LAY + 0j)
    r12 = (n1 - n2) / (n1 + n2)
    for i, fq in enumerate(f):
        if fq == 0:
            continue
        phi = 2.0 * np.pi * fq * n2 * THICK / C0
        e = np.exp(-2j * phi)
        r[i] = (r12 - r12 * e) / (1.0 - r12 ** 2 * e)
    S = np.fft.rfft(pulse, n_fft)
    echo = np.fft.irfft(S * r, n_fft)
    inc = np.fft.irfft(S, n_fft)
    return (float(np.max(np.abs(analytic(echo))))
            / float(np.max(np.abs(analytic(inc)))))


def run_radarwave(dx):
    xlim, zlim = (-50.0, 50.0), (-5.0, 110.0)
    grid = PropertyGrid.uniform(xlim, zlim, dx, eps=EPS_BG)
    _, Z = grid.mesh()
    eps = {k: np.array(v, copy=True) for k, v in grid.eps.items()}
    for k in eps:
        eps[k][(Z >= Z_LAY) & (Z < Z_LAY + THICK)] = EPS_LAY
    layered = grid.with_properties(eps=eps)

    dt = 0.9 * max_time_step(grid.eps_min, grid.mu_min, grid.dx, grid.dz)
    t = np.arange(0.0, 1.35e-6, dt)
    pulse = blackharrispulse(FC, t)
    src = np.array([[0.0, 0.0]])
    rec = np.array([[0.0, 0.0], [0.0, DEPTH_REF]])

    tr_lay = FDTD2D(layered, dt, npml=12, mode="TM").run(
        src, pulse, rec, outstep=1).gather[:, :, 0]
    tr_bg = FDTD2D(grid, dt, npml=12, mode="TM").run(
        src, pulse, rec, outstep=1).gather[:, :, 0]

    echo = tr_lay[:, 0] - tr_bg[:, 0]
    v = C0 / np.sqrt(EPS_BG)
    t_arr = 2 * Z_LAY / v
    win = np.abs(t - t_arr) < 0.12e-6
    a_echo = float(np.max(np.abs(analytic(echo))[win]))
    a_ref = float(np.max(np.abs(analytic(tr_bg[:, 1]))))
    return a_echo / a_ref, pulse, dt, t


def run_gprmax(gpr_python, gpr_repo, pulse, dt, t, dx):
    import h5py

    d = OUT / "gprmax"
    d.mkdir(parents=True, exist_ok=True)
    wave = d / "thin_wave.txt"
    amps = np.concatenate([pulse, np.zeros(max(16, len(pulse) // 20))])
    with wave.open("w") as f:
        f.write("time src\n")
        for tt, a in zip(np.arange(len(amps)) * dt, amps):
            f.write(f"{tt:.9e} {a:.9e}\n")

    lat, dep = 100.0, 115.0
    sx, sz = 50.0, 5.0
    results = {}
    for tag, with_layer in (("lay", True), ("bg", False)):
        inp = d / f"thin_{tag}.in"
        with inp.open("w") as f:
            f.write(f"#title: thin_{tag}\n")
            f.write(f"#domain: {lat:.9g} {dx:.9g} {dep:.9g}\n")
            f.write(f"#dx_dy_dz: {dx:.9g} {dx:.9g} {dx:.9g}\n")
            f.write(f"#time_window: {len(t)*dt:.9g}\n")
            f.write("#pml_cells: 20 0 20 20 0 20\n")
            f.write(f"#material: {EPS_BG:.9g} 0 1 0 bg\n")
            f.write(f"#material: {EPS_LAY:.9g} 0 1 0 lay\n")
            f.write(f"#box: 0 0 0 {lat:.9g} {dx:.9g} {dep:.9g} bg\n")
            if with_layer:
                f.write(f"#box: 0 0 {sz + Z_LAY:.9g} {lat:.9g} {dx:.9g} "
                        f"{sz + Z_LAY + THICK:.9g} lay\n")
            f.write(f"#excitation_file: {wave.name}\n")
            f.write(f"#hertzian_dipole: y {sx:.9g} 0 {sz:.9g} src\n")
            f.write(f"#rx: {sx:.9g} 0 {sz:.9g}\n")
            f.write(f"#rx: {sx:.9g} 0 {sz + DEPTH_REF:.9g}\n")
        r = subprocess.run([gpr_python, "-m", "gprMax", inp.name], cwd=d,
                           capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(gpr_repo)})
        if r.returncode != 0:
            print(r.stdout[-2000:]); print(r.stderr[-2000:])
            raise SystemExit(f"gprMax {tag} failed")
        with h5py.File(inp.with_suffix(".out"), "r") as f:
            dt_g = float(f.attrs["dt"])
            results[tag] = (np.array(f["rxs/rx1/Ey"]), np.array(f["rxs/rx2/Ey"]),
                            dt_g)

    (lay0, _, dt_g) = results["lay"]
    (bg0, bg_ref, _) = results["bg"]
    t_g = np.arange(lay0.size) * dt_g
    echo = lay0 - bg0
    v = C0 / np.sqrt(EPS_BG)
    t_arr = 2 * Z_LAY / v
    win = np.abs(t_g - t_arr) < 0.12e-6
    a_echo = float(np.max(np.abs(analytic(echo))[win]))
    a_ref = float(np.max(np.abs(analytic(bg_ref))))
    return a_echo / a_ref


def main(gpr_python, gpr_repo, dx=0.2):
    rw, pulse, dt, t = run_radarwave(dx)
    exact = exact_ratio(pulse, dt)
    gp = run_gprmax(gpr_python, gpr_repo, pulse, dt, t, dx)
    db = lambda x: 20 * np.log10(x)
    print(f"\nthin layer ({THICK} m, eps {EPS_BG} -> {EPS_LAY}) at {Z_LAY} m, "
          f"dx = {dx} m")
    print(f"  exact (transfer matrix x image source): {db(exact):8.2f} dB")
    print(f"  radarwave                             : {db(rw):8.2f} dB "
          f"({db(rw) - db(exact):+.2f})")
    print(f"  gprMax                                : {db(gp):8.2f} dB "
          f"({db(gp) - db(exact):+.2f})")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], dx=float(a[2]) if len(a) > 2 else 0.2)
