"""Example 2 -- two polarisations through ice with a preferred orientation fabric.

Ice with a horizontal crystal-orientation fabric is birefringent: the two
eigenpolarisations travel at slightly different speeds, and the delay between
them accumulates with depth.  That delay is what the polarimetric
interferograms in the SCAR figures measure.

The fabric here is the one inverted from Ridge A (Open Polar Radar frame
20250108_02_009): ``dlam = lam_perp - lam_par`` rising from about 0.02 at
250 m to 0.07 near 1500 m, with the two horizontal principal axes at 89 and
179 degrees east of north.

Two things are computed:

1. **A 2-D FDTD pair, run for two fabric strengths.**  Each eigenpolarisation
   is propagated with the out-of-plane solver fed its own permittivity
   component, so both sit on the same grid.  The movie uses a strong,
   shear-margin-like fabric, where the two polarisations visibly separate as
   they go down.  Receivers on the nadir axis give the one-way delay for both
   the strong fabric and Ridge A's own much weaker one, and both are checked
   against the traveltime model.

2. **A full-column synthetic interferogram.**  Ridge A is ~1850 m thick and
   MCoRDS runs near 195 MHz, so the ice column is some 4000 wavelengths deep --
   far beyond a 2-D FDTD grid.  The fringes are therefore synthesised from the
   same dielectric model, then pushed through the same processing applied to
   the real frames, and the recovered ``dlam`` is compared with the input.

A note on resolution: the delay being measured is a fraction of a time step,
so it is sensitive to numerical dispersion, which differs slightly between the
two runs because their wave speeds differ.  The error converges as the grid is
refined, and the full-resolution run sits at 11 nodes per wavelength, where it
is under 1 percent; ``--quick`` drops to 5.6 and is for checking the pipeline,
not for the numbers.  The convergence figures are in the README's "Resolution"
section.

Outputs land in ``figures/ex02/``.  ``--render-only`` re-renders the movie from
the cached snapshots without re-running the simulation, and refuses a cache that
was written from a different model.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    StaleCache,
    load_snapshots,
    save_figure,
    save_snapshots,
    stampable,
)

from radarwave import (
    C0,
    FDTD2D,
    IceColumn,
    PropertyGrid,
    blackharrispulse,
    dominant_frequency,
    envelope_peak_time,
    gabor,
    max_time_step,
    ridge_a_dlambda,
)
from radarwave.ice import RIDGE_A, RIDGE_A_DLAMBDA, RIDGE_A_DLAMBDA_DEPTH
from radarwave.polarimetry import (
    analytic,
    delay_from_phase,
    delayed_copy,
    dlambda_from_delay,
    interferogram,
    isolate_arrival,
    subsample_lag,
    unwrap_phase,
    volume_scattering,
)
from radarwave.scenes import IceModelBuilder, Layer
from radarwave.viz import crop_snapshots, use_talk_style, wavefield_movie

OUT = Path(__file__).resolve().parent.parent / "figures" / "ex02"
FC = 60e6  # FDTD source frequency
FC_MCORDS = 195e6  # centre frequency of the real Ridge A product
NPML = 12
PERP_AZ, PAR_AZ = 89, 179  # horizontal principal axes, degrees east of north
LAYER_DEPTHS = (60.0, 130.0, 210.0, 300.0, 420.0, 560.0, 700.0)  # acidity horizons
Z_SRC = 3.0  # antenna buried a few metres into the firn
REC_TOP, REC_STEP, REC_MARGIN = 40.0, 20.0, 25.0  # nadir receiver string


#: Essentially the strongest horizontal fabric ice can have: every c-axis lying
#: in the horizontal plane and aligned, lam ~ (0.97, 0.01, 0.02).  The grain
#: anisotropy deps = 0.034 caps the effect, so this is the largest speed
#: difference between two polarisations that is physically available, about
#: 0.5 percent.  Ridge A's own contrast is roughly twenty times weaker -- still
#: measurable (see the delay panel) but far too small to watch on a screen.
STRONG_DLAMBDA = 0.96
STRONG_LAM_Z = 0.02

#: The two columns as their constructor arguments, and the acidity reflector as
#: its own.  Kept as mappings rather than written out at the call site so the
#: cache stamp is taken from the same values the model is built from, and a
#: parameter added to either one cannot escape the stamp.
STRONG_COLUMN = dict(
    thickness=1850.0,
    depth_bco=100.0,
    sigma_ice=1.5e-5,
    lam_z_sfc=STRONG_LAM_Z,
    lam_z_bed=STRONG_LAM_Z,
    dlambda=lambda d: np.full_like(np.asarray(d, float), STRONG_DLAMBDA),
)
COLUMNS = {"ridge": RIDGE_A, "strong": STRONG_COLUMN}
REFLECTOR = dict(thickness=1.0, sigma_factor=3.0, undulations=((2.0, 260.0, 0.7),))


def make_column(kind):
    return IceColumn(**COLUMNS[kind])


def fdtd_geometry(quick):
    """Return ``(dx, xlim, zlim, t_end)`` for the FDTD pair.

    Narrow and deep.  Nothing here needs width - the wave goes straight down and
    the first Fresnel zone at 800 m is only ~30 m across - but depth is exactly
    what the demonstration needs, because the separation between the two
    eigenpolarisations grows linearly with distance travelled.

    ``t_end`` is long enough for the deepest layer's echo to come back, not just
    for the downgoing wave to cross the model: the received-waveform figure needs
    the reflections.  The movie only uses the first part, while the wave is on
    its way down.

    Split out of :func:`run_fdtd` so the cache stamp is read from the same place
    the run is, and cannot drift away from it.
    """
    if quick:
        return 0.5, (-45.0, 45.0), (-8.0, 150.0), 2.2e-6
    return 0.26, (-45.0, 45.0), (-8.0, 800.0), 8.6e-6


def cache_stamp(quick):
    """Everything the cached snapshots, traces and delays depend on.

    Re-rendering a cache written from a different model would put its measured
    delays and amplitudes beside the traveltime numbers computed here from the
    current constants, with nothing in the figure to say so.  Every value that
    changes the physics has to be in here or the guard gives false confidence:
    the columns and the reflector come straight from the mappings they are built
    from, and the two things their ``dlambda`` callables close over -- Ridge A's
    tabulated profile and ``STRONG_DLAMBDA`` -- are stamped beside them.
    """
    dx, xlim, zlim, t_end = fdtd_geometry(quick)
    return {
        "dx": dx,
        "xlim": xlim,
        "zlim": zlim,
        "t_end": t_end,
        "fc": FC,
        "npml": NPML,
        "layer_depths": LAYER_DEPTHS,
        "reflector": REFLECTOR,
        "z_src": Z_SRC,
        "receivers": (REC_TOP, REC_STEP, REC_MARGIN),
        "strong_dlambda": STRONG_DLAMBDA,
        "ridge_a_dlambda_depth": RIDGE_A_DLAMBDA_DEPTH,
        "ridge_a_dlambda": RIDGE_A_DLAMBDA,
        **{f"column_{kind}": stampable(params) for kind, params in COLUMNS.items()},
    }


def run_fdtd(quick, kind="ridge"):
    dx, xlim, zlim, t_end = fdtd_geometry(quick)

    column = make_column(kind)
    grid = PropertyGrid.uniform(xlim, zlim, dx)
    builder = IceModelBuilder(grid, column, surface=0.0, air=True)
    # A few reflectors so the movie shows returns as well as the direct wave.
    layer_depths = [d for d in LAYER_DEPTHS if d < zlim[1] - 20]
    builder.add_layers([Layer(depth=d, **REFLECTOR) for d in layer_depths])
    model = builder.finalize(npml=NPML)

    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, t_end, dt)
    pulse = blackharrispulse(FC, t)
    fpeak = dominant_frequency(pulse, dt)
    # The wavelet is not centred on t = 0, so an echo's envelope peaks this
    # much after its traveltime.  It is ~10 ns, comparable with the delays
    # being measured, so every predicted arrival below carries it.
    t_wave = envelope_peak_time(pulse, t)

    z_src = Z_SRC
    src = np.array([[0.0, z_src]])
    # Receiver 0 sits back at the source: that is the monostatic trace, the
    # thing an actual sounder measures.  The rest are strung down the nadir
    # axis to give the one-way delay directly.
    rec_z = np.arange(REC_TOP, zlim[1] - REC_MARGIN, REC_STEP)
    rec = np.vstack([[0.0, z_src], np.column_stack([np.zeros_like(rec_z), rec_z])])
    snap_every = max(1, len(t) // 240)

    # Both eigenpolarisations are run with the out-of-plane (TM) solver, each
    # given its own permittivity component.  For a fabric column that varies
    # only with depth this is exact: a wave polarised along x propagating in the
    # y-z plane obeys the same scalar equation as a wave polarised along y in
    # the x-z plane, with eps_xx in place of eps_yy.  Doing it this way puts
    # both runs on the identical grid with the identical source and receiver
    # nodes, so numerical dispersion -- which is far larger than the ~0.2 ns
    # delay being measured -- cancels exactly in their difference.  (The vector
    # TE solver is available and validated separately; it is not used here
    # because its fields are staggered half a cell away from the TM fields.)
    variants = {
        "perp": model.with_properties(
            eps={"xx": model.eps["xx"], "yy": model.eps["xx"], "zz": model.eps["zz"]}
        ),
        "par": model,
    }

    runs = {}
    for label, m in variants.items():
        sim = FDTD2D(m, dt, npml=NPML, mode="TM")
        if not runs:
            print(sim.summary())
            print(f"peak frequency {fpeak / 1e6:.1f} MHz, {len(t)} steps")
        t0 = time.time()
        runs[label] = sim.run(
            src, pulse, rec, outstep=1, snapshot_every=snap_every,
            snapshot_stride=1, progress=max(1, len(t) // 5),
        )
        print(f"  {label} eigenpolarisation finished in {time.time() - t0:.1f} s")

    return column, model, grid, dt, runs, src, rec, xlim, zlim, fpeak, z_src, t_wave


def received_waveforms(surface, column_of, layer_depths, measured, t_wave, out_path,
                      bare=False):
    """Eight monostatic traces: two fabrics x four transmit/receive pairings.

    The top row is the two eigenpolarisations.  Those are the fabric's own
    modes, so they do not interfere, and their amplitudes are identical to
    within the ~0.05 dB their reflection coefficients differ by -- the layers
    are isotropic, so both polarisations see the same reflector.  The fabric
    shows up in *when* they arrive, not in how strong they are.

    The bottom row is what an antenna at 45 degrees to the principal axes
    records, which is what a real survey usually measures.  A 45 degree launch
    is not an eigenmode: it is the two eigenmodes at equal amplitude.  They
    propagate independently -- the fabric orientation is fixed with depth here,
    so no energy is exchanged -- and recombine at the receiver with whatever
    differential phase they have accumulated.  Co-pol adds them and cross-pol
    differences them, so both beat through nulls as that phase passes multiples
    of pi.  That beating is the amplitude signature polarimetric sounding
    actually lives on, and it is absent from the eigenpolarisation panels by
    construction.

    Nothing here needs a new simulation: the superposition is exact whenever the
    eigenaxes do not rotate with depth.

    Plotted as returned power in dB against two-way time, the way a processed
    sounding is shown, on one shared reference so the panels are directly
    comparable in both timing and amplitude.
    """
    import matplotlib.pyplot as plt

    eigen = [("ridge", "perp"), ("ridge", "par"), ("strong", "perp"), ("strong", "par")]
    mixed = [("ridge", "co"), ("ridge", "cross"), ("strong", "co"), ("strong", "cross")]
    order = eigen + mixed
    names = {
        "perp": f"E along {PERP_AZ} deg (slow)",
        "par": f"E along {PAR_AZ} deg (fast)",
        "co": "45 deg launch, co-pol",
        "cross": "45 deg launch, cross-pol",
    }
    colours = {"perp": "#2166ac", "par": "#b2182b", "co": "#6a3d9a", "cross": "#1b7837"}
    titles = {"ridge": "Ridge A fabric", "strong": f"strong fabric (dlam = {STRONG_DLAMBDA:g})"}

    def trace_of(k, w):
        if w in ("perp", "par"):
            return np.asarray(surface[k][w], dtype=float)
        slow = np.asarray(surface[k]["perp"], dtype=float)
        fast = np.asarray(surface[k]["par"], dtype=float)
        return 0.5 * (slow + fast) if w == "co" else 0.5 * (slow - fast)

    env = {(k, w): np.abs(analytic(trace_of(k, w))) for k, w in order}
    ref = max(float(env[key].max()) for key in eigen)
    z_src_of = {k: float(v[2]) for k, v in measured.items()}

    def arrival(col, depth, axis, kind):
        """When the echo from ``depth`` peaks in the record.

        The antenna is buried a few metres down, so the path is from there and
        not from the surface -- about 26 ns through firn, which is half the
        width of the inset below.  ``t_wave`` puts the wavelet's own offset
        back in so the mark lands on the peak rather than beside it.
        """
        return 2.0 * col.traveltime(z_src_of[kind], depth, axis=axis) + t_wave

    # Reference arrival of the deepest usable layer, per fabric, for the fast
    # eigenpolarisation.  Drawn in both panels of a pair so the slow one can be
    # seen lagging it.
    def usable_layers(kind, col, t, margin=0.25e-6):
        return [d for d in layer_depths
                if arrival(col, d, None, kind) < t[-1] - margin]

    # One common zoom window and one common reference time for all four panels,
    # so peak positions can be compared straight across.  Anchor it on the
    # Ridge A fast axis.
    _ref_col = column_of("ridge")
    _t_ref = surface["ridge"]["t"]
    _use = usable_layers("ridge", _ref_col, _t_ref)
    zoom_depth = max(_use) if _use else max(layer_depths)
    t_ref = arrival(_ref_col, zoom_depth, 1, "ridge")

    curves = {}
    for k, w in order:
        with np.errstate(divide="ignore"):
            curves[(k, w)] = 20 * np.log10(np.maximum(env[(k, w)] / ref, 1e-12))

    # One inset window *and one inset amplitude range* for every panel.  The
    # caption claims the windows are comparable, so the axes have to actually be
    # shared -- letting each autoscale makes two panels look alike that are 10 dB
    # apart.
    ins_lo, ins_hi = np.inf, -np.inf
    for k, w in order:
        t = surface[k]["t"]
        band = np.abs(t - t_ref) < 65e-9
        if band.any():
            ins_lo = min(ins_lo, float(curves[(k, w)][band].min()))
            ins_hi = max(ins_hi, float(curves[(k, w)][band].max()))

    fig, axes = plt.subplots(2, 4, figsize=(17.5, 13.6), sharey=True)
    for ax, (k, w) in zip(axes.flat, order):
        t = surface[k]["t"]
        col = column_of(k)
        # The mixed pairings carry both eigenmodes, so mark their layers on the
        # mean axis rather than on either one.
        axis = {"par": 1, "perp": 0}.get(w, None)
        db = curves[(k, w)]
        colour = colours[w]
        ax.plot(db, t * 1e6, lw=1.0, color=colour)

        for d in layer_depths:
            ax.axhline(arrival(col, d, axis, k) * 1e6, color="#999", ls=":", lw=0.9)
        ax.set_xlim(-118, 5)
        ax.set_ylim((t[-1] - 0.25e-6) * 1e6, 0)
        ax.set_xlabel("returned power\n(dB re. transmit pulse)")
        ax.set_title(f"{titles[k]}\n{names[w]}", loc="left", fontsize=12)
        ax.grid(alpha=0.2)

        # Zoom on the deepest return: the shift is tens of ns against a record
        # thousands of ns long, so it is only readable magnified.
        ins = ax.inset_axes([0.40, 0.06, 0.57, 0.30])
        ins.plot(db, t * 1e6, lw=1.3, color=colour)
        ins.axhline(t_ref * 1e6, color="#111", ls="--", lw=1.1)
        ins.set_ylim((t_ref + 60e-9) * 1e6, (t_ref - 60e-9) * 1e6)
        ins.set_xlim(ins_lo - 3, ins_hi + 5)
        ins.tick_params(labelsize=8)
        ins.set_title(f"{zoom_depth:.0f} m return, same window in all eight",
                      fontsize=8.5, loc="left", pad=2)
        ins.grid(alpha=0.25)
    axes[0, 0].set_ylabel("two-way time (us)")
    axes[1, 0].set_ylabel("two-way time (us)")

    # Quantify both effects on the deepest layer: how much later the slow
    # polarisation arrives, and whether its amplitude differs at all.
    #
    # The delay is read off the reflection itself, but only after the window is
    # detrended and tapered.  These echoes stand ~10 dB above the wake the
    # transmit pulse leaves behind, which over a window several cycles wide is
    # still the larger share of the energy -- and the wake has travelled
    # nowhere, so correlating the raw window returns a delay near zero.
    #
    # The amplitude cannot be read there: 10 dB of headroom leaves a couple of
    # dB of scatter, far above the effect being tested.  It comes instead from
    # the downgoing pulse at the deepest nadir receiver, which is clean enough
    # to resolve a tenth of a dB.
    notes = []
    for k in ("ridge", "strong"):
        col = column_of(k)
        t = surface[k]["t"]
        rec_z, _lag, zs, amp_db = measured[k]
        usable = usable_layers(k, col, t, margin=0.15e-6)
        if not usable:
            continue
        deepest = max(usable)
        t0 = arrival(col, deepest, None, k)
        idx, (a_par, a_perp) = isolate_arrival(
            t, [surface[k]["par"], surface[k]["perp"]], t0, 0.12e-6
        )
        if idx.sum() < 16:
            continue
        shift = subsample_lag(a_par, a_perp, t[1] - t[0])
        model = float(col.birefringent_delay(deepest) - col.birefringent_delay(zs))
        # The eigenpolarisations carry no amplitude signature -- the layers are
        # isotropic, so both see the same reflector and the only difference is
        # the ~0.05 dB their reflection coefficients differ by.  The amplitude
        # signature appears once the launch is *not* an eigenmode: the same
        # delay becomes a differential phase, and co- and cross-pol beat through
        # a null every half cycle of it.
        cycles = abs(float(col.birefringent_delay(deepest))) * FC
        notes.append(
            f"{titles[k]}: the {PERP_AZ} deg return from {deepest:.0f} m arrives "
            f"{shift * 1e9:.2f} ns later than {PAR_AZ} deg (traveltime model "
            f"{model * 1e9:.2f} ns); down at {float(np.asarray(rec_z)[-1]):.0f} m the two "
            f"are still within {abs(float(np.asarray(amp_db)[-1])):.2f} dB in amplitude. "
            f"That delay is {cycles:.2f} cycles at {FC / 1e6:.0f} MHz, so a 45 deg launch "
            f"beats through {2 * cycles:.1f} nulls by {deepest:.0f} m."
        )

    header = [
        "Top row, the fabric's own eigenpolarisations: it changes when the reflections "
        "arrive, not how strong they are.",
        "Bottom row, a 45 deg launch, which is what a survey actually transmits: the two "
        "modes recombine and beat, so the same delay now shows up as amplitude.",
        "Dotted lines mark each layer; the inset magnifies the deepest return, dashed "
        "line at the fast-axis arrival.",
    ] + notes
    fig.suptitle("\n".join(header), x=0.008, ha="left", fontsize=12.5)
    # Reserve exactly the band the header needs.  A fixed fraction left a dead
    # stripe when the notes were shorter than the space set aside for them.
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.020 * len(header) - 0.010))
    save_figure(fig, out_path, bare)
    plt.close(fig)
    return notes


def main(quick=False, render_only=False, bare=False):
    use_talk_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "snapshots.npz"

    import matplotlib.pyplot as plt

    column = IceColumn(**RIDGE_A)
    stamp = cache_stamp(quick)

    if render_only and cache.exists():
        panels, sx, sz, stimes, extra = load_snapshots(cache, stamp=stamp)
        fpeak = float(extra["fpeak"])
        t_wave = float(extra["t_wave"])
        measured = {
            k: (extra[f"rec_z_{k}"], extra[f"lag_{k}"], float(extra[f"z_src_{k}"]),
                extra[f"amp_db_{k}"])
            for k in ("ridge", "strong")
        }
        surface = {
            k: {w: extra[f"surf_{k}_{w}"] for w in ("t", "perp", "par")}
            for k in ("ridge", "strong")
        }
    else:
        measured = {}
        surface = {}
        panels = None
        for kind in ("ridge", "strong"):
            (col, model, grid, dt, runs, src, rec, xlim, zlim, fpeak,
             z_src, t_wave) = run_fdtd(quick, kind=kind)

            # Identical grids and identical source/receiver nodes, so the lag
            # is the birefringent delay with no geometric correction needed.
            perp, par = runs["perp"], runs["par"]

            def _peak(res, k):
                return float(np.max(np.abs(analytic(res.gather[:, k, 0]))))

            measured[kind] = (
                par.rec[1:, 1],
                np.array([
                    subsample_lag(par.gather[:, k, 0], perp.gather[:, k, 0], dt)
                    for k in range(1, rec.shape[0])
                ]),
                par.src[0, 1],
                # Whether the fabric costs the slow polarisation any amplitude
                # on the way down, measured where the pulse is clean.
                np.array([
                    20.0 * np.log10(_peak(perp, k) / _peak(par, k))
                    for k in range(1, rec.shape[0])
                ]),
            )
            # Monostatic surface traces, one per eigenpolarisation.
            surface[kind] = {
                "t": par.t,
                "perp": perp.gather[:, 0, 0],
                "par": par.gather[:, 0, 0],
            }

            if kind == "strong":
                margin = 0.07 * (xlim[1] - xlim[0])
                panels, sx, sz = crop_snapshots(
                    [
                        (perp.snapshots, f"E along {PERP_AZ} deg  (slow axis)"),
                        (par.snapshots, f"E along {PAR_AZ} deg  (fast axis)"),
                    ],
                    par.snapshot_x,
                    par.snapshot_z,
                    xlim=(xlim[0] + margin, xlim[1] - margin),
                    zlim=(zlim[0], zlim[1] - 0.3 * margin),
                )
                stimes = par.snapshot_times

        save_snapshots(
            cache, panels, sx, sz, stimes, stamp=stamp, fpeak=fpeak, t_wave=t_wave,
            **{f"{n}_{k}": v for k in ("ridge", "strong")
               for n, v in zip(("rec_z", "lag", "z_src", "amp_db"), measured[k])},
            **{f"surf_{k}_{w}": surface[k][w] for k in ("ridge", "strong")
               for w in ("t", "perp", "par")},
        )

    # ---- movie ---------------------------------------------------------
    # Speed difference between the two eigenpolarisations of the strong fabric,
    # quoted in the subtitle so the figure states its own physics.
    _strong = make_column("strong")
    _eps = _strong.permittivity(np.array([200.0]))[0]
    dv_over_v = float((np.sqrt(_eps[0]) - np.sqrt(_eps[1]))
                      / np.sqrt(0.5 * (_eps[0] + _eps[1])))
    track_depth = 780.0
    offset_m = dv_over_v * track_depth

    wavefield_movie(
        OUT / "birefringence.mp4",
        panels,
        sx,
        sz,
        stimes,
        fps=25,
        gain_power=0.85,
        gamma=0.35,
        alpha_knee=0.02,
        share_scale=True,
        follow={"speed": C0 / np.sqrt(2.9), "window": 34.0, "start": -10.0},
        guide={"panel": 1, "label": "fast-axis wavefront", "halfwidth": 12.0},
        trace={
            "t": surface["strong"]["t"],
            "series": [
                (surface["strong"]["perp"], f"{PERP_AZ} deg (slow)", "#2166ac"),
                (surface["strong"]["par"], f"{PAR_AZ} deg (fast)", "#b2182b"),
            ],
            "db": True,
            # From the antenna, which is buried in the firn, and offset by the
            # wavelet -- otherwise every guide sits ~16 ns off its own echo.
            "guides": list(
                make_column("strong").two_way_time(
                    np.array(LAYER_DEPTHS), z0=float(measured["strong"][2])
                ) + t_wave
            ),
            "tlim": 8.4e-6,
            "xlim": (-125.0, 5.0),
            "title": "what the receiver records",
        },
        title="Two polarisations through a strong horizontal fabric",
        subtitle=(
            f"{fpeak / 1e6:.0f} MHz through the strongest horizontal fabric ice can have "
            f"(dlam = {STRONG_DLAMBDA:g}): the two eigenpolarisations travel "
            f"{100 * dv_over_v:.2f} percent apart in speed, so by {track_depth:.0f} m the "
            f"slow one has fallen {offset_m:.1f} m behind. Dashed line marks the fast "
            "wavefront in both panels. Right: what the receiver records - the two return "
            "pulses come back at the same amplitude, only shifted in time."
        ),
        bare=bare,
    )
    print(f"  wrote {OUT / 'birefringence.mp4'}")

    # ---- full-column synthetic interferogram ----------------------------
    # A section of independent traces sharing one fabric column, so the picture
    # has the speckle of a real product and the phase can be stacked the way
    # the processing chain stacks it.
    n_traces = 180
    trace_spacing = 25.0  # m along track
    zz = np.arange(2.0, column.thickness - 20.0, 3.0)
    eps = column.permittivity(zz)
    slow_mean = 0.5 * (np.sqrt(eps[:, 0]) + np.sqrt(eps[:, 1])) / C0
    t0 = 2.0 * np.concatenate([[zz[0] * slow_mean[0]],
                               zz[0] * slow_mean[0] + np.cumsum(np.diff(zz) * slow_mean[:-1])])
    dtau = column.birefringent_delay(zz)

    # A band-limited wavelet, matching the ~15 percent bandwidth of MCoRDS.
    # Interferometric phase only equals 2*pi*fc*delay for a narrowband signal.
    dt_s = 1.0 / (8.0 * FC_MCORDS)
    t_axis = np.arange(0.0, t0[-1] + 60e-9, dt_s)
    wavelet = gabor(FC_MCORDS, np.arange(0, 14.0 / FC_MCORDS, dt_s), bandwidth=0.16)

    # The delay expressed on the time axis, so the second polarisation can be
    # made by warping the first.
    dtau_t = np.interp(t_axis, t0, dtau, left=0.0, right=dtau[-1])

    rng = np.random.default_rng(11)
    ig = np.zeros((t_axis.size, n_traces), dtype=complex)
    for j in range(n_traces):
        hh = volume_scattering(t_axis, rng, wavelet, decay_time=14e-6)
        vv = delayed_copy(t_axis, hh, dtau_t)
        noise = 0.03 * np.std(hh)
        hh = hh + rng.normal(scale=noise, size=hh.size)
        vv = vv + rng.normal(scale=noise, size=vv.size)
        ig[:, j] = interferogram(hh, vv)

    depth_axis = np.interp(t_axis, t0, zz, left=0.0, right=zz[-1])
    keep = (depth_axis > 30.0) & (depth_axis < column.thickness - 40.0)

    # Stack across traces and multilook in range before unwrapping -- the same
    # order the real processing uses.
    from scipy.ndimage import uniform_filter1d

    stacked = ig[keep].mean(axis=1)
    stacked = uniform_filter1d(stacked.real, 101) + 1j * uniform_filter1d(stacked.imag, 101)
    phi_u = unwrap_phase(np.angle(stacked))
    z_valid = depth_axis[keep]
    tau_est = delay_from_phase(phi_u - phi_u[0], FC_MCORDS)
    dlam_est = dlambda_from_delay(z_valid, tau_est, column=column, smooth=401)

    # ---- figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 7.6), width_ratios=[1, 1, 1.35, 1])

    zfine = np.linspace(0, column.thickness, 600)
    ax = axes[0]
    ax.plot(ridge_a_dlambda(zfine), zfine, color="#b2182b", lw=2)
    ax.set_xlabel(r"$\Delta\lambda = \lambda_\perp - \lambda_\parallel$")
    ax.set_ylabel("depth (m)")
    ax.set_ylim(column.thickness, 0)
    ax.set_title(f"1.  Ridge A fabric\n({PERP_AZ} vs {PAR_AZ} deg E of N)",
                 loc="left", fontsize=12)

    # FDTD validation, on its own depth range so the points are readable.
    ax = axes[1]
    zmax = max(float(np.max(measured[k][0])) for k in measured) * 1.05
    styles = {"ridge": ("#b2182b", "o", "Ridge A fabric"),
              "strong": ("#2166ac", "s", f"strong fabric (dlam = {STRONG_DLAMBDA:g})")}
    for kind, (rec_z, lag, zs, _amp) in measured.items():
        col_, mk, lbl = styles[kind]
        c = make_column(kind)
        zf = np.linspace(0, zmax, 300)
        ax.plot(
            0.5 * (c.birefringent_delay(zf) - c.birefringent_delay(zs)) * 1e12,
            zf, color=col_, lw=2, alpha=0.8,
        )
        ax.plot(np.asarray(lag) * 1e12, rec_z, mk, ms=7, mfc="none", mew=1.8,
                color=col_, label=lbl)
    ax.set_xscale("symlog", linthresh=30)
    ax.set_xlabel("one-way delay between\neigenpolarisations (ps)")
    ax.set_ylabel("depth (m)  [note: FDTD range only]", fontsize=11)
    ax.set_ylim(zmax, 0)
    ax.legend(fontsize=9.5, loc="lower right")
    span = max(abs(float(np.asarray(v[1])[-1])) for v in measured.values()) / max(
        min(abs(float(np.asarray(v[1])[-1])) for v in measured.values()), 1e-30)
    ax.set_title(f"2.  FDTD matches the traveltime model\nover a {span:.0f}x range of "
                 "fabric strength", loc="left", fontsize=12)

    ax = axes[2]
    # Multilook in range for display, as a real interferogram product is.
    look = 41
    ig_disp = (uniform_filter1d(ig[keep].real, look, axis=0)
               + 1j * uniform_filter1d(ig[keep].imag, look, axis=0))
    coh = np.abs(ig_disp) / np.max(np.abs(ig_disp))
    hsv = np.zeros(coh.shape + (3,))
    hsv[..., 0] = (np.angle(ig_disp) + np.pi) / (2 * np.pi)
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(coh / np.percentile(coh, 85), 0, 1) ** 0.45
    from matplotlib.colors import hsv_to_rgb

    ax.imshow(
        hsv_to_rgb(hsv), aspect="auto", origin="upper",
        extent=(0, n_traces * trace_spacing / 1000.0, z_valid[-1], z_valid[0]),
    )
    ax.set_xlabel("distance along profile (km)")
    ax.set_ylim(column.thickness, 0)
    ax.set_title(
        f"3.  synthetic interferogram at {FC_MCORDS / 1e6:.0f} MHz\n"
        "(hue = phase, brightness = coherence)", loc="left", fontsize=12,
    )

    ax = axes[3]
    ax.plot(ridge_a_dlambda(z_valid), z_valid, color="#b2182b", lw=2.2, label="input")
    ax.plot(dlam_est, z_valid, lw=1.3, color="#2166ac", alpha=0.9, label="recovered")
    ax.set_xlim(-0.02, 0.11)
    ax.set_xlabel(r"$\Delta\lambda$")
    ax.set_ylim(column.thickness, 0)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title("4.  inverting the fringes\nrecovers the input", loc="left", fontsize=12)

    for a in axes[2:]:
        a.set_yticklabels([])
    fig.suptitle(
        "Ridge A fabric  ->  birefringent delay  ->  polarimetric fringes  ->  recovered fabric",
        x=0.008, ha="left", fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, OUT / "birefringence.png", bare)
    plt.close(fig)

    notes = received_waveforms(
        surface, make_column,
        list(LAYER_DEPTHS),
        measured, t_wave,
        OUT / "received.png",
        bare,
    )
    print(f"  wrote {OUT / 'received.png'}")
    for line in notes:
        print("    " + line)

    n_fringes = (phi_u[-1] - phi_u[0]) / (2 * np.pi)
    print(f"  wrote {OUT / 'birefringence.png'}")
    for kind, (rec_z, lag, zs, _amp) in measured.items():
        c = make_column(kind)
        model_ps = 0.5 * (c.birefringent_delay(float(rec_z[-1]))
                          - c.birefringent_delay(zs)) * 1e12
        print(f"  {kind:6s} fabric, one-way delay at {rec_z[-1]:.0f} m: "
              f"FDTD {lag[-1] * 1e12:7.1f} ps, model {model_ps:7.1f} ps "
              f"({100 * (lag[-1] * 1e12 / model_ps - 1):+.1f}%)")
    print(f"  full column: {column.birefringent_delay(column.thickness - 20) * 1e9:.2f} ns "
          f"two-way, {n_fringes:.2f} fringes at {FC_MCORDS / 1e6:.0f} MHz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--bare", action="store_true",
                   help="strip titles, notes, annotations and legends for slides")
    a = p.parse_args()
    try:
        main(quick=a.quick, render_only=a.render_only, bare=a.bare)
    except StaleCache as exc:
        raise SystemExit(str(exc))
