"""Figure for the three-way anisotropic check: radarwave vs gprMax vs analytic.

One panel per polarisation, every trace normalised to its own peak inside the
arrival window (the codes' source normalisations differ by orders of
magnitude, and amplitude is not what this case tests).  Dashed lines mark the
analytic arrivals; the residual panel shows each code's split against the
analytic value.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from radarwave.polarimetry import analytic          # noqa: E402
from radarwave.viz import use_talk_style            # noqa: E402

OUT = REPO / "figures" / "validation"


def main(bare=False):
    import matplotlib.pyplot as plt

    use_talk_style()
    d = np.load(OUT / "anisotropic_comparison.npz")
    t, t_wave, eps, depth = d["t"], float(d["t_wave"]), d["eps"], float(d["depth"])
    from radarwave.constants import C0
    t_x = depth / (C0 / np.sqrt(eps[0])) + t_wave
    t_y = depth / (C0 / np.sqrt(eps[1])) + t_wave

    series = {
        "E along y": [("radarwave TM", d["ours_Ey_(TM)"], "#b2182b", "-"),
                      ("gprMax", d["gprMax_Ey"], "#2166ac", "--")],
        "E along x": [("radarwave TM sub", d["ours_Ex_(TM_sub)"], "#b2182b", "-"),
                      ("radarwave TE", d["ours_Ex_(TE)"], "#ef8a62", "-"),
                      ("gprMax", d["gprMax_Ex"], "#2166ac", "--")],
    }
    marks = {"E along y": t_y, "E along x": t_x}

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.6), sharey=True)
    win = 60e-9
    centre = 0.5 * (t_x + t_y)
    for ax, (label, rows) in zip(axes, series.items()):
        keep = np.abs(t - centre) < win
        for name, tr, colour, ls in rows:
            env = np.abs(analytic(np.asarray(tr, dtype=float)))
            ax.plot(env[keep] / env[keep].max(), t[keep] * 1e6,
                    lw=1.4, color=colour, ls=ls, label=name)
        ax.axhline(marks[label] * 1e6, color="#111", ls=":", lw=1.1)
        ax.annotate("analytic", (0.03, marks[label] * 1e6),
                    xycoords=("axes fraction", "data"), fontsize=9,
                    va="bottom", color="#111")
        ax.set_title(label, loc="left")
        ax.set_xlabel("envelope (normalised)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("time (us)")
    axes[0].invert_yaxis()

    fig.suptitle(
        "A strong single-maximum fabric, source over receiver at "
        f"{depth:.0f} m: three solvers against the analytic arrivals.\n"
        "The birefringent split is 4.58 ns analytic, 4.58 ns radarwave "
        "(both modes), 4.75 ns gprMax.",
        x=0.008, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if bare:
        from radarwave.viz import strip_text
        strip_text(fig)
        fig.tight_layout(rect=(0, 0, 1, 1))
    path = OUT / "anisotropic_comparison.png"
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    main(bare="--bare" in sys.argv)
