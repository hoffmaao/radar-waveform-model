"""Run ex03's bright polarisation through gprMax and plot it against radarwave.

Usage:  compare_ex03_gprmax.py <gprmax-python> <gprmax-repo> [--quick]

Both codes are handed the identical permittivity map, the identical source
waveform, the same cell size and the same source/receiver position, so any
difference in the recorded trace is a difference in the solvers.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples"))
sys.path.insert(0, str(REPO / "validation"))

from gprmax_export import write_case                     # noqa: E402
from radarwave import FDTD2D, blackharrispulse, max_time_step  # noqa: E402
import ex03_dipping_fabric_transition as ex03            # noqa: E402

OUT = REPO / "figures" / "validation"


def main(gpr_python, gpr_repo, quick=False):
    dx = 0.6 if quick else 0.20
    xlim, zlim = (-110.0, 110.0), (-8.0, 300.0)
    column, grid, model, twin, boundary, layers, _exclude = ex03.build_models(
        xlim, zlim, dx)

    # The bright polarisation: the TM solver reads eps_yy, so ex03 feeds it
    # eps_xx.  That scalar map is exactly what gprMax will be given.
    mm = model.with_properties(
        eps={"xx": model.eps["xx"], "yy": model.eps["xx"], "zz": model.eps["zz"]})
    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, 1.95e-6, dt)
    pulse = blackharrispulse(ex03.FC, t)
    src = np.array([[0.0, ex03.Z_ANT]])

    print(f"grid {model.eps['xx'].shape}, dx = {dx} m, {len(t)} steps")
    ours = FDTD2D(mm, dt, npml=ex03.NPML, mode="TM").run(
        src, pulse, src, outstep=1).gather[:, 0, 0]

    # radarwave's PropertyGrid samples at HALF-cell resolution (staggered-grid
    # coefficients live between nodes), so the raw array is 0.1 m data for a
    # 0.2 m model -- export it as-is with dx and gprMax solves a domain
    # stretched 2x in both axes.  Take the cell-centre samples, and place the
    # antenna via the model's own axes rather than the nominal limits (the
    # array also carries the PML padding, which shifts the corner).
    eps = np.asarray(mm.eps["yy"], dtype=float)[1::2, 1::2]
    sig = np.asarray(mm.sig["yy"], dtype=float)[1::2, 1::2]
    xc, zc = mm.x[1::2], mm.z[1::2]
    assert abs((xc[1] - xc[0]) - dx) < 1e-9, "cell-centre stride is not dx"
    sx = 0.0 - float(xc[0])
    sz = ex03.Z_ANT - float(zc[0])
    case = write_case(OUT / "gprmax", "ex03_bright", eps=eps, sig=sig,
                      dx=dx, dz=dx, src_xz=(sx, sz), rec_xz=(sx, sz),
                      pulse=pulse, dt=dt, n_steps=len(t))
    print("wrote", case)

    env = {"PYTHONPATH": str(gpr_repo)}
    print("running gprMax ...")
    r = subprocess.run([gpr_python, "-m", "gprMax", case.name],
                       cwd=case.parent, capture_output=True, text=True,
                       env={**__import__("os").environ, **env})
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-3000:])
        raise SystemExit(f"gprMax failed ({r.returncode})")

    import h5py
    outfile = case.with_suffix(".out")
    with h5py.File(outfile, "r") as f:
        dt_g = float(f.attrs["dt"])
        theirs = np.array(f["rxs/rx1/Ey"])
    t_g = np.arange(theirs.size) * dt_g
    print(f"gprMax dt = {dt_g*1e12:.3f} ps vs ours {dt*1e12:.3f} ps")

    np.savez_compressed(OUT / "ex03_gprmax_comparison.npz",
                        t_ours=t, ours=ours, t_gpr=t_g, theirs=theirs, dx=dx)
    print("saved", OUT / "ex03_gprmax_comparison.npz")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(a[0], a[1], quick="--quick" in sys.argv)
