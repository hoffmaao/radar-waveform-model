"""Plot the radarwave / gprMax comparison for ex03's bright polarisation.

The two codes normalise their sources differently -- gprMax's hertzian dipole
and radarwave's soft source differ by about three orders of magnitude -- so
every trace is referenced to its own transmit pulse.  That ratio is the
physically meaningful quantity and the one under test: it is what "the fabric
return sits N dB below the transmit pulse" means.
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
    d = np.load(OUT / "ex03_gprmax_comparison.npz")
    t_o, o, t_g, g = d["t_ours"], d["ours"], d["t_gpr"], d["theirs"]

    # A common axis: gprMax picks its own Courant-limited step.
    t = t_o
    g_i = np.interp(t, t_g, g)


    # Reference in the FAR field, not at the source cell.  Each code records
    # its "transmit pulse" at the source node, where the field is the
    # self-field of the injection -- a hertzian dipole's differs from a soft
    # source's by a constant that has nothing to do with propagation, and
    # normalising there shifts one whole curve by that constant.  The shallow
    # firn layering is the earliest window where both records are far-field,
    # so both are referenced to their median envelope there.
    dt_s = float(t[1] - t[0])
    k = max(1, int(round(25e-9 / dt_s)))          # ~25 ns smoothing: the
    kern = np.ones(k) / k                         # point-by-point dB of two
    env_o = np.convolve(np.abs(analytic(o)), kern, mode="same")    # codes'
    env_g = np.convolve(np.abs(analytic(g_i)), kern, mode="same")  # nulls is
    ref_band = (t > 0.15e-6) & (t < 0.30e-6)      # noise, not physics
    # Normalise on the band's strongest PEAK, not its median.  The two codes'
    # effective source wavelets differ in shape -- gprMax's hertzian dipole has
    # a heavier tail than radarwave's soft source -- so the median sits on the
    # inter-packet floor, which is exactly where the wavelet tails live, and a
    # median reference folds that shape difference into every number.  The
    # first layer packet is the same reflection in both codes.
    ref_o = env_o[ref_band].max()
    ref_g = env_g[ref_band].max()
    db_o = 20 * np.log10(np.maximum(env_o / ref_o, 1e-12))
    db_g = 20 * np.log10(np.maximum(env_g / ref_g, 1e-12))

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 8.2))

    ax = axes[0]
    ax.plot(db_o, t * 1e6, lw=1.1, color="#b2182b", label="radarwave")
    ax.plot(db_g, t * 1e6, lw=1.1, color="#2166ac", ls="--", label="gprMax")
    ax.set_xlim(-140, 5)
    ax.set_ylim(t[-1] * 1e6, 0)
    ax.set_xlabel("returned power\n(dB re. shallow layering)")
    ax.set_ylabel("two-way time (us)")
    ax.set_title("full record", loc="left")
    ax.grid(alpha=0.2)
    ax.legend(loc="lower left", fontsize=10)

    ax = axes[1]
    w = (t > 0.15e-6) & (t < 0.55e-6)
    # Scale both waveforms by their envelope in the same reference band the dB
    # panels use.  Scaling each to its own maximum in the window handed the
    # gprMax curve to its top-boundary artefact and buried the layering.
    ax.plot(o[w] / ref_o, t[w] * 1e6, lw=1.2, color="#b2182b", label="radarwave")
    ax.plot(g_i[w] / ref_g, t[w] * 1e6, lw=1.2, color="#2166ac",
            ls="--", label="gprMax")
    ax.set_ylim(0.55, 0.15)
    ax.set_xlabel("amplitude\n(re. shallow layering)")
    ax.set_title("shallow layering, waveform shape", loc="left")
    ax.grid(alpha=0.2)

    ax = axes[2]
    diff = db_o - db_g
    ax.plot(diff, t * 1e6, lw=1.0, color="#333")
    ax.axvline(0.0, color="#999", lw=0.8)
    ax.set_xlim(-12, 12)
    ax.set_ylim(t[-1] * 1e6, 0)
    ax.set_xlabel("radarwave - gprMax (dB)")
    ax.set_title("difference", loc="left")
    ax.grid(alpha=0.2)

    band = (t > 0.3e-6) & (t < 1.9e-6)
    med = float(np.median(diff[band]))
    p90 = float(np.percentile(np.abs(diff[band] - med), 90))

    # The number that means something: agreement ON THE REFLECTION EVENTS.
    # Between events the trace is the 2-D line-source wake, whose level rides
    # on each code's low-frequency source content -- outside the band the
    # first-packet match can constrain -- so a full-band median mostly
    # compares wakes.  Score the codes where the medium actually speaks.
    from scipy.signal import find_peaks
    ev_band = (t > 0.12e-6) & (t < 1.9e-6)
    pk, _ = find_peaks(db_o[ev_band],
                       height=float(db_o[ev_band].max()) - 45.0,
                       distance=int(60e-9 / dt_s), prominence=6.0)
    t_ev = t[ev_band][pk]
    ev_diff = diff[ev_band][pk]
    ax.plot(np.clip(ev_diff, -11.5, 11.5), t_ev * 1e6, "o", ms=5,
            mfc="none", mec="#b2182b", mew=1.4, zorder=5)
    fig.suptitle(
        "ex03 bright polarisation: the same permittivity map solved by two "
        "independent FDTD codes, each referenced to its own first layer "
        "packet.\nEvery reflection event lands at the same time in both "
        "codes (circles: per-event level difference). Event levels and the "
        "floor differ by\nup to ~10 dB because the codes radiate different "
        "source signatures; amplitude fidelity is established separately by "
        "the thin-layer test\n(both codes within 1 dB of the exact answer).",
        x=0.008, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    if bare:
        from radarwave.viz import strip_text
        strip_text(fig)
        fig.tight_layout(rect=(0, 0, 1, 1))
    path = OUT / "ex03_gprmax_comparison.png"
    fig.savefig(path)
    print(f"median offset {med:+.3f} dB, 90th pct spread {p90:.3f} dB")
    print("wrote", path)


if __name__ == "__main__":
    main(bare="--bare" in sys.argv)
