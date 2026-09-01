"""Example 6 -- the same repeat series, in the EAGER traverse's accum3 band.

Example 5 puts an ApRES pulse over a deforming column and repeats it once a
year for four years.  This one repeats the identical experiment with the
CReSIS/OPR accumulation radar: the *same* column, the *same* deformation, the
*same* five visits, the *same* processing, all imported from
:mod:`ex05_apres_repeat`.  The only thing that differs is the band that is
transmitted, which is what makes the pair a comparison of waveforms rather than
of two experiments that happen to resemble each other.

THE BAND.  600-900 MHz in survey mode, from
``default_radar_params_2022_Antarctica_Ground_accum.m`` - the configuration run
on the 2022_Antarctica_Ground traverse at Windless Bight, whose repeat passes
are what the EAGER vertical-deformation product (``~/projects/radar_velocity``)
is built from.  Unlike ApRES it is a pulsed system: it transmits a linear-FM
chirp and matched-filters it rather than deramping on receive.  On a linear
medium the two chains produce the same band-limited complex range profile,
which is why both examples compress the same way here, and the tests check that
equivalence rather than asserting it.

WHAT THE BAND BUYS AND COSTS.  Two numbers, pulling against each other:

============================  =========================  =========================
                              ApRES                      accum3
============================  =========================  =========================
band                          200-400 MHz                600-900 MHz
range resolution in ice       0.42 m (0.70 m windowed)   0.28 m (0.46 m windowed)
one turn of phase             28.1 cm                    11.2 cm
============================  =========================  =========================

accum3 separates two reflectors half again as finely and reads a given motion
with two and a half times the phase - and wraps two and a half times as often
doing it.  Both effects are visible here.  The finer resolution splits some firn
bands into their top and bottom interfaces, which ApRES sees as one echo; the
tighter fringe means the envelope estimate that resolves the whole-fringe
ambiguity has to stay inside 5.6 cm rather than ApRES's 14 cm, and both examples
print how close it came.  More fringes is more signal only for as long as they
can still be counted.

THE DOMAIN IS SHALLOWER, AND HAS TO BE.  Resolving 900 MHz in ice needs a
1.6 cm cell against ApRES's 3.5 cm, so the same depth would cost about five
times the cell-updates.  The instrument does not reach that far anyway - an
accumulation radar is a firn instrument, and the EAGER strain result is quoted
over the top 100 m - so this example sounds 60 m rather than 140.  The
displacement profile's sign change sits at 72 m, below this domain, so unlike
example 5 every marker here is sinking: what accum3 sees is the top, steepest
part of the same curve, in finer detail.

Outputs land in ``figures/ex06/``.  ``--render-only`` re-renders from cache.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import StaleCache

import ex05_apres_repeat as ex05

from radarwave import ACCUM3
from radarwave.viz import use_talk_style

OUT = Path(__file__).resolve().parent.parent / "figures" / "ex06"


def main(quick=False, render_only=False, bare=False, out=None):
    use_talk_style()
    # 12 nodes per wavelength at the top of the band, matching how example 5
    # samples its own band edge, so the two runs carry the same numerical
    # dispersion budget.  A comparison of bands must not also be a comparison of
    # grid quality.
    dx = 0.045 if quick else 0.0156
    xlim = (-6.0, 6.0)
    zlim = (-2.0, 30.0) if quick else (-2.0, 60.0)
    n_epochs = 3 if quick else ex05.N_EPOCHS
    return ex05.run_series(
        ACCUM3, xlim, zlim, dx, n_epochs=n_epochs, render_only=render_only,
        bare=bare, out=out if out is not None else str(OUT),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write all outputs and caches under DIR "
                        "instead of figures/exNN (for sweeps and "
                        "cluster array jobs)")
    p.add_argument("--bare", action="store_true",
                   help="strip titles, notes, annotations and legends for slides")
    a = p.parse_args()
    try:
        main(quick=a.quick, render_only=a.render_only, bare=a.bare, out=a.out)
    except StaleCache as exc:
        raise SystemExit(str(exc))
