"""Mechanism fingerprints for the Ridge A interface across the MCoRDS band.

Plots |R(f)| for every exported candidate from this repository's own transfer
matrix, and overlays the CReSIS ``em_model`` result wherever
``emmodel_results.mat`` exists (produced by ``emmodel_driver.m`` in the
fabric_anisotropy MATLAB container).  Agreement between the two is a
cross-validation of both; the spread BETWEEN candidates is the point: each
mechanism carries a distinct spectral signature inside 150-250 MHz, so the
observed feature's behaviour across the band identifies it.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from radarwave.constants import C0           # noqa: E402
from radarwave.viz import use_talk_style     # noqa: E402

OUT = REPO / "figures" / "validation"


def transfer_matrix_R(depth, er_of_f, freq):
    """|R(f)| of a thin-layer stack between half-spaces (normal incidence)."""
    thick = np.diff(np.asarray(depth, dtype=float).ravel())
    out = np.zeros(freq.size)
    for i, f in enumerate(freq):
        n = np.sqrt(er_of_f[:, i].astype(complex))
        k0 = 2 * np.pi * f / C0
        M = np.eye(2, dtype=complex)
        for nj, d in zip(n[1:-1], thick[1:-1]):
            ph = k0 * nj * d
            M = M @ np.array([[np.cos(ph), 1j * np.sin(ph) / nj],
                              [1j * nj * np.sin(ph), np.cos(ph)]])
        n0, ns = n[0], n[-1]
        num = n0 * (M[0, 0] + M[0, 1] * ns) - (M[1, 0] + M[1, 1] * ns)
        den = n0 * (M[0, 0] + M[0, 1] * ns) + (M[1, 0] + M[1, 1] * ns)
        out[i] = abs(num / den)
    return out


def main(bare=False):
    import matplotlib.pyplot as plt
    from scipy.io import loadmat

    use_talk_style()
    cand = loadmat(OUT / "emmodel_candidates.mat", squeeze_me=False)
    freq = np.asarray(cand["freq"]).ravel()
    names = [str(np.squeeze(n)) for n in np.asarray(cand["names"]).ravel()]

    results = None
    res_path = OUT / "emmodel_results.mat"
    if res_path.exists():
        results = loadmat(res_path, squeeze_me=True)

    fig, ax = plt.subplots(figsize=(11.5, 7.6))
    colours = plt.cm.tab10(np.linspace(0, 1, len(names)))
    for name, c in zip(names, colours):
        depth = cand[f"{name}__depth"]
        er = np.asarray(cand[f"{name}__er_x"])
        R = transfer_matrix_R(depth, er, freq)
        ax.plot(freq / 1e6, 20 * np.log10(np.maximum(R, 1e-16)),
                color=c, lw=1.8, label=name.replace("_", " "))
        if results is not None and f"{name}__R_x" in results:
            ax.plot(freq / 1e6,
                    20 * np.log10(np.maximum(
                        np.asarray(results[f"{name}__R_x"]).ravel(), 1e-16)),
                    color=c, lw=1.0, ls="--")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("|R| (dB), bright eigenpolarisation")
    ax.set_ylim(-130, -15)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower left")
    extra = ("solid: this repo's transfer matrix; dashed: CReSIS em_model"
             if results is not None else
             "this repo's transfer matrix (run emmodel_driver.m to overlay "
             "CReSIS em_model)")
    ax.set_title("Candidate mechanisms for a deep interface, 150-250 MHz\n"
                 + extra, loc="left")
    fig.tight_layout()
    if bare:
        from radarwave.viz import strip_text
        strip_text(fig)
        fig.tight_layout()
    path = OUT / "emmodel_fingerprints.png"
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    main(bare="--bare" in sys.argv)
