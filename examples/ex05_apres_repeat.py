"""Example 5 -- one ApRES pulse, then the same pulse once a year for four years.

Examples 1-4 ask what is down there.  This one asks what the ice is *doing*.  A
phase-sensitive sounder set up over the same Lagrangian site does not measure
the reflectors; it measures how far they moved between visits, and it does so
to about a centimetre, because it reads the *phase* of each echo and not its
arrival time.

The experiment is exactly that, five times over.  One chirp goes into a
firn/ice column carrying ordinary meteoric stratigraphy and the record is kept.
The column is deformed by a year of :class:`radarwave.VerticalStrain` and the
identical chirp is transmitted again.  Five visits, four years.  Each record is
range-compressed, and the difference between consecutive ones - a phase per
echo, and so an apparent motion per layer - is the measurement.

WHY A SERIES AND NOT A PAIR.  A pair shows that the phase carries the motion.  A
series shows the two things a pair cannot.  First, the motion *accumulates*: by
year four the shallowest markers have moved close to two metres, which is
several resolution cells and six or seven fringes, so following one marker means
tracking it through steps that each move it more than a resolution cell.
Second, the phase wraps repeatedly on the way, and the measurement only holds
together if every wrap is accounted for - which is what a real ApRES deployment,
recording for a season and processed pair by pair, has to do.  Consecutive pairs
rather than everything against year zero, for the same reason: against year zero
the interferogram would have decorrelated and the unwrap would have nothing to
hold on to.

THE DEFORMATION.  Two things move a marker in the depth-below-surface
coordinate a surface-referenced radar works in, and they pull opposite ways:

* **burial** - snow accumulates on top and the firn beneath compacts, so a
  marker sinks.  In steady state this is ``b / rho_rel(z)`` per year: 0.57 m/yr
  at a 0.35 relative-density surface, 0.22 m/yr in solid ice.  Entirely a firn
  effect, and strongly depth-dependent.
* **dynamic thinning** - a vertical strain rate shortens the column above the
  marker by ``eps_zz * z`` per year, lifting it back towards the surface.
  Linear in depth, and the term glaciology wants out of the measurement.

With this example's numbers they balance at about 72 m: shallow markers sink
half a metre a year, deep ones rise a third of one, and the apparent motion
changes sign in between.  That sign change is the signature.  A bulk timing
error, an antenna repositioned between visits, a tide lifting the whole column -
none of them can imitate it, because none of them is a function of depth.

WHY A CHIRP, AND NOT AN IMPULSE.  A displacement is read off the phase of the
compressed echo, ``-2 pi f_c dtau``, and that expression needs an ``f_c``: an
impulse whose instantaneous frequency wanders across an octave does not have
one.  So the source is a real linear-FM chirp across the real ApRES band,
200-400 MHz (:data:`radarwave.APRES`).  Two numbers follow from the band, and
they set everything here:

* range resolution ``c / (2 B sqrt(eps))`` = 0.42 m in ice (0.70 m once the
  Blackman band window is applied), which is how finely two layers can be told
  apart;
* fringe spacing ``c / (2 f_c sqrt(eps))`` = 28.1 cm, which is how far a layer
  must move to wind the phase one full turn.

The chirp simulated is a hundred nanoseconds rather than the instrument's
second, because deramp-on-receive needs a sweep far longer than the two-way time
to the deepest target and an FDTD record is microseconds long.  The *band* is
the instrument's, so the compressed wavelet, the resolution and the sidelobes
are too; what is not reproduced is the time-bandwidth product, and with it the
SNR gain the long sweep exists to provide.  These traces carry no noise for it
to improve.  The deramp chain itself is in :mod:`radarwave.waveform` and is
checked against this compression in the tests.

HOW THE MOTION IS RECOVERED.  Envelope first, then phase, per consecutive pair,
which is what a real ApRES chain does and what ``+vdef`` does to the EAGER
repeat passes:

1. every echo in the earlier record is found, and the later record's envelope is
   cross-correlated against it;
2. the interferometric phase at each echo is unwrapped along depth;
3. the whole-fringe ambiguity left in that unwrapped phase is fixed from the
   envelope estimates, and the phase then gives the range change;
4. echoes are chained from step to step into marker trajectories, and the
   per-step motions accumulate along them.

The envelope's job is step 3 and only step 3: to say which fringe.  It has to be
right to half a fringe to do it - 14 cm at 300 MHz, 5.6 cm at 750 - and both
examples print how close it came, because that margin is the thing that fails
first when a band is pushed up.  These traces carry no noise, so the envelope
does better here than it would in the field; the phase's precision is what does
not depend on that.

NO MOVIE.  The other examples end in one; this one does not, and deliberately.
The medium is identical at every epoch - the steady-state firn column means the
velocity structure and the snow surface do not move, only the material markers
in them do - so consecutive wavefields differ by less than a metre in 140, which
is under a pixel.  There would be nothing to watch.  The measurement is in the
compressed records, in how the phase turns with depth, and in how it winds with
time, so that is what the four figures show: ``pulse.png`` (the transmitted
chirp, its band and its compressed shape), ``record.png`` (the records and the
phase between them), ``motion.png`` (apparent motion against depth, every year)
and ``series.png`` (the trajectories, and the same thing as the radar reads it).

Outputs land in ``figures/ex05/``.  ``--render-only`` re-renders from cache.
:mod:`ex06_eager_repeat` runs the same deformation through the CReSIS/OPR
accum3 band, and is the comparison the pair exists for.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (
    StaleCache,
    echo_time,
    load_arrays,
    render_from_cache,
    save_arrays,
    save_figure,
    stampable,
)

from radarwave import (
    APRES,
    C0,
    FDTD2D,
    IceColumn,
    PropertyGrid,
    VerticalStrain,
    lfm_chirp,
    max_time_step,
    range_compress,
)
from radarwave.polarimetry import interferogram, upsampled_lag
from radarwave.scenes import IceModelBuilder, Layer, conformal_layering
from radarwave.viz import use_talk_style

OUT = Path(__file__).resolve().parent.parent / "figures" / "ex05"

NPML = 12
Z_ANT = -1.0  # antenna 1 m above the snow surface, at both epochs

#: The interval between visits.  A year is the baseline at which the
#: deformation is metres-per-thousand rather than the micrometres a tidal repeat
#: gives, and it is long enough that the firn burial term - which is what makes
#: the apparent motion change sign with depth - is not negligible.  It is also
#: long enough that the steady-state firn assumption is doing real work; see
#: :mod:`radarwave.deform`.
EPOCH_YEARS = 1.0

#: How many visits.  Five spans four years, over which the shallowest markers
#: move close to two metres - several resolution cells and six or seven ApRES
#: fringes - so the series exercises the two things a single pair cannot: that a
#: marker can be followed across steps that each move it more than a resolution
#: cell, and that the accumulated phase stays tied to the accumulated motion
#: rather than drifting a fringe at a time.  Each epoch is one more full FDTD
#: run, so this is also the parameter that sets the runtime.
N_EPOCHS = 5

#: The background column, kept as its constructor arguments rather than written
#: out at the call site so the cache stamp comes from the same place the model
#: does.  1200 m is a shelf-like thickness; neither example sounds anywhere near
#: it, but it sets the ``lam_z`` gradient.
COLUMN = dict(thickness=1200.0, rho_sfc=0.35, rho_bco=0.81, depth_bco=100.0,
              sigma_ice=1.5e-5)

#: Deformation.  0.20 m ice equivalent per year is an unremarkable coastal
#: Antarctic accumulation rate, and -4e-3 /yr a brisk but ordinary vertical
#: strain rate for ice under longitudinal extension.  Together they put the sign
#: change at ~72 m, inside example 5's reach and below example 6's, which is the
#: whole reason for these two numbers rather than any other pair.
STRAIN = dict(accumulation=0.20, strain_rate=-4.0e-3)

#: Stratigraphy.  Much denser than the other examples' - firn banding every
#: 3-6 m rather than every 15 - because this measurement needs an echo every few
#: metres to sample the motion profile with, and both bands resolve a 0.6 m band
#: at that spacing.  The undulations are small: a Lagrangian site is sounded at
#: nadir, and a layer that dips across the first Fresnel zone smears its own
#: echo without adding anything either example is about.
LAYERS = dict(top=4.0, firn_base=95.0, firn_spacing=(3.0, 6.0),
              ice_spacing=(6.0, 11.0), thickness_firn=(0.5, 0.9),
              thickness_ice=(0.5, 0.9), d_rho=(0.012, 0.026),
              sigma_factor=(2.5, 3.5), amplitude=(0.15, 0.4),
              wavelengths=(140.0, 45.0), seed=11)

#: Duration of the transmitted chirp.  ``tpd * bandwidth`` is the
#: time-bandwidth product actually simulated - 20 for ApRES, 30 for accum3 -
#: enough for the compressed wavelet to take its asymptotic shape.  Every extra
#: nanosecond is an extra nanosecond of record before the first echo can arrive,
#: and at these grid sizes that is minutes.
TPD = 100e-9

#: Layer edges are ramped over this many cells.  The number the whole example
#: depends on: a hard-edged layer can only sit on a grid node, so displacing it
#: by less than a cell displaces it by zero or by a whole cell, and at 3.5 cm
#: cells that quantisation is up to a radian - larger than much of the signal.
#: Ramping over a couple of cells lets the sampled profile's centroid follow the
#: requested depth continuously.  It costs reflectivity (the ramp is about a
#: fifth of a wavelength at the top of either band), which these density layers
#: can afford and the -51 dB fabric reflectors of examples 3 and 4 could not.
EDGE_CELLS = 2.5


def ice_column():
    return IceColumn(**COLUMN)


def strain_model(column=None):
    return VerticalStrain(column or ice_column(), **STRAIN)


def layering(max_depth, edge_width=0.0):
    return conformal_layering(max_depth, edge_width=edge_width, **LAYERS)


def deformed(layers, strain, years=EPOCH_YEARS):
    """The same stratigraphy ``years`` later.

    Each layer's centre is carried along the deformation trajectory and its
    thickness scaled by the local stretch, so a band in compacting firn thins as
    it sinks.  Over a year that is under a percent - far below what either band
    resolves - but it costs one evaluation and keeps a decade-long baseline from
    quietly accumulating an error.

    Undulation amplitudes and anomaly magnitudes carry across unchanged.  A
    vertical strain that is a function of depth alone does not fold a layer, and
    the density anomaly's own mass balance is a sub-percent correction on a
    sub-percent correction.
    """
    return [
        Layer(
            depth=float(strain.displaced_depth(layer.depth, years)),
            thickness=float(layer.thickness * strain.stretch(layer.depth, years)),
            d_rho=layer.d_rho,
            sigma_factor=layer.sigma_factor,
            dip_deg=layer.dip_deg,
            undulations=layer.undulations,
            edge_width=layer.edge_width,
        )
        for layer in layers
    ]


def build_model(grid, column, layers, npml=NPML):
    return IceModelBuilder(grid, column, surface=0.0, air=True).add_layers(layers).finalize(npml)


def compressed(trace, wavelet, dt, system):
    """Range-compressed complex profile, on the record's own time axis.

    Note what compression does to that axis, because every prediction here
    depends on it: correlating against a wavelet that starts at ``t = 0`` puts
    an echo's peak at its own two-way traveltime, with none of the envelope-peak
    offset the other examples must carry (:func:`_common.echo_time`'s
    ``t_wave``).  The chirp's hundred nanoseconds of extent are precisely what
    the compression removes.
    """
    return range_compress(trace, wavelet, dt, system)


def find_echoes(env_db, usable, dt, system, dyn_range=35.0, prominence=12.0,
                local_reach=6.0, local_depth=10.0):
    """Indices of the echoes in a compressed record.

    A compressed echo is a wavelet, not a spike: the Blackman window makes its
    main lobe 1.65 resolution cells wide and leaves sidelobes a couple of cells
    out, 20-26 dB down, on top of whatever the twin subtraction did not remove.
    Peak-finding on amplitude alone therefore returns half again to twice as
    many "echoes" as there are layers, and every extra one carries its parent's
    phase rather than a phase of its own - which is what turns a one-centimetre
    measurement into a five-centimetre one at 750 MHz.

    Three rules, and the last one is what does the work:

    * a peak must rise ``prominence`` dB out of its own surroundings;
    * it must stand within ``dyn_range`` of the brightest echo in the record;
    * it must be within ``local_depth`` dB of the strongest sample inside
      ``local_reach`` resolution cells of it.

    The third is deliberately conservative.  It throws away a genuine reflector
    that happens to be 10 dB weaker than a neighbour a metre or two away, and
    that is the intended trade: a lost echo costs one point on a profile of
    twenty, while a retained sidelobe puts a wrong point on it and nothing
    downstream can tell which it was.  Measured across both bands, 10 dB keeps
    22 echoes of 23 layers for ApRES and 10 for accum3's 8 - accum3 resolves
    some bands into their top and bottom interfaces, which is real and is what
    the extra bandwidth bought.
    """
    from scipy.ndimage import maximum_filter1d
    from scipy.signal import find_peaks

    env = np.where(usable, env_db, -300.0)
    cell = max(1, int(round(1.0 / (system.bandwidth * dt))))
    peaks, _ = find_peaks(
        env,
        distance=2 * cell,
        height=float(np.max(env)) - dyn_range,
        prominence=prominence,
    )
    local_max = maximum_filter1d(env, size=2 * int(local_reach * cell) + 1,
                                 mode="nearest")
    return np.asarray([k for k in peaks if env[k] >= local_max[k] - local_depth],
                      dtype=int)


def apparent_motion(prof_1, prof_2, t, dt, column, system, peaks, z_ant=Z_ANT):
    """Range change at every echo: envelope first, then phase.

    This is the ApRES chain, and the division of labour in it is the point.  The
    envelope cross-correlation is coarse - a few centimetres on a clean echo,
    which is a fifth of a fringe at 300 MHz and would be most of one at 750.  The phase is precise to a hundredth of a fringe but only modulo a
    whole turn.  So the phase carries the measurement and the envelope is used
    for one thing: choosing which turn.

    The phase is unwrapped along depth *before* that choice, so one integer is
    fixed for the whole profile rather than one per echo.  Choosing per echo
    lets a single noisy envelope estimate insert a fringe-sized step into an
    otherwise smooth profile, which reads as a layer that did something its
    neighbours did not.  The integer is the median over all echoes, so no one
    bad envelope estimate moves it.

    Returns a mapping with ``depth``, ``phase`` (wrapped, as measured),
    ``dtau``, ``displacement`` and ``displacement_envelope`` (m), one entry per
    echo, plus the ``fringes`` integer that was resolved.
    """
    env_1 = np.abs(prof_1)
    env_2 = np.abs(prof_2)
    cell = max(1, int(round(1.0 / (system.bandwidth * dt))))
    # Four resolution cells either side of the echo, which is a compromise with
    # a real floor and a real ceiling.  A Blackman-compressed wavelet is 1.65
    # cells across at -3 dB and several more at -20, so a window much narrower
    # than this truncates the shifted copy asymmetrically and drags the estimate
    # towards zero: at two cells it reads 12 cm short of a 40 cm motion.  A
    # window much wider reaches the next layer, 3-6 m away, and measures two
    # reflectors at once.  1.7 m of range sits between the two.
    half = 4 * cell

    igram = interferogram(prof_2, prof_1)
    phase = np.angle(igram[peaks])
    dtau_env = np.array([
        upsampled_lag(env_1[max(0, k - half): k + half + 1],
                      env_2[max(0, k - half): k + half + 1], dt)
        for k in peaks
    ])

    # Unwrap along depth.  Consecutive echoes are 3-6 m apart and the phase
    # turns by well under pi across that, so this is safe here; on a real record
    # it is the step that needs a coherence mask and a gap rule, which is what
    # vdef.differentialRange spends most of its length on.
    phase_unwrapped = np.unwrap(phase)
    turns = (-2.0 * np.pi * system.fc * dtau_env - phase_unwrapped) / (2.0 * np.pi)
    n0 = float(np.round(np.median(turns)))
    dtau = -(phase_unwrapped + 2.0 * np.pi * n0) / (2.0 * np.pi * system.fc)

    depth = column.depth_from_two_way_time(t[peaks], z0=z_ant)
    # The local refractive index, not a column average: the column between the
    # surface and the reflector changed length, and the ice that came or went is
    # at the reflector's own depth.  This is vdef.verticalDisplacement's
    # conversion, and through firn a column average would be a fifth wrong.
    eps = column.permittivity(depth)
    n_local = np.sqrt(0.5 * (eps[:, 0] + eps[:, 1]))
    to_metres = C0 / (2.0 * n_local)

    return {
        "peaks": np.asarray(peaks, dtype=int),
        "depth": depth,
        "phase": phase,
        "dtau": dtau,
        "displacement": dtau * to_metres,
        "displacement_envelope": dtau_env * to_metres,
        "fringes": n0,
    }


def _break_wraps(phase, valid):
    """``phase`` with a gap at every wrap, so a line plot does not join the ends.

    A wrapped profile jumps from +pi to -pi wherever it completes a turn, and a
    line plot draws that jump as a horizontal stroke straight across the panel.
    On a record carrying six turns that is six false features.
    """
    out = np.where(valid, np.asarray(phase, dtype=float), np.nan)
    jump = np.abs(np.diff(out)) > np.pi
    out[1:][jump] = np.nan
    return out


def track_markers(steps, tolerance):
    """Chain per-step echo lists into trajectories for individual markers.

    Each step measures the echoes it finds in its *own* first record, so the
    lists are not the same reflectors in the same order: an echo can drop below
    the detection threshold for a year and reappear, and the shallow ones move
    by more than a resolution cell per step.  A marker is followed by predicting
    where this step says it will be and taking the nearest echo in the next
    step's list, which is what a real time series does and what makes the
    accumulated displacement a property of one reflector rather than of one
    range bin.

    A marker whose predicted position has no echo within ``tolerance`` is
    dropped rather than matched to the nearest thing available.  Chaining onto
    the wrong reflector does not look like an error afterwards - it looks like a
    layer that moved several metres in a year.

    Parameters
    ----------
    steps : sequence of dict
        One :func:`apparent_motion` result per consecutive epoch pair.
    tolerance : float
        Largest depth mismatch (m) that still counts as the same marker.

    Returns
    -------
    list of dict
        One per surviving marker, with ``depth0`` and, per epoch from 1
        onwards, cumulative ``displacement``, ``dtau`` and the ``depth`` it was
        followed to.
    """
    markers = [{"depth0": float(z), "depth": [float(z)],
                "displacement": [0.0], "dtau": [0.0], "index": [i]}
               for i, z in enumerate(steps[0]["depth"])]

    for k, step in enumerate(steps):
        survivors = []
        for m in markers:
            if m["index"][-1] is None:
                continue
            i = m["index"][-1]
            here = float(step["depth"][i])
            moved = float(step["displacement"][i])
            m["displacement"].append(m["displacement"][-1] + moved)
            m["dtau"].append(m["dtau"][-1] + float(step["dtau"][i]))
            m["depth"].append(here + moved)
            if k + 1 < len(steps):
                candidates = steps[k + 1]["depth"]
                j = int(np.argmin(np.abs(candidates - (here + moved))))
                m["index"].append(j if abs(candidates[j] - (here + moved)) <= tolerance
                                  else None)
            survivors.append(m)
        markers = survivors
    return [m for m in markers if len(m["displacement"]) == len(steps) + 1]


def run_series(system, xlim, zlim, dx, *, tpd=TPD, n_epochs=N_EPOCHS,
               render_only=False, bare=False, out=None):
    """Simulate, cache and draw one repeat-pass time series.

    Shared by this example and :mod:`ex06_eager_repeat`, which differ only in
    the band they transmit and the domain that band can afford.  Keeping the
    body here rather than duplicating it is what guarantees the two examples are
    a comparison of waveforms and not of two experiments that merely resemble
    each other.
    """
    import matplotlib.pyplot as plt

    outdir = Path(out) if out is not None else OUT
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "records.npz"

    column = ice_column()
    strain = strain_model(column)
    epoch_years = np.arange(n_epochs) * EPOCH_YEARS
    layers_0 = layering(zlim[1], edge_width=EDGE_CELLS * dx)
    # Each epoch is displaced from the ORIGINAL stratigraphy by its own elapsed
    # time, not chained one step at a time: VerticalStrain integrates the
    # trajectory, so evaluating it at 4 years is the trajectory at 4 years and
    # not four Euler steps' worth of accumulated truncation error.
    layer_sets = [layers_0 if y == 0 else deformed(layers_0, strain, years=float(y))
                  for y in epoch_years]

    grid = PropertyGrid.uniform(xlim, zlim, dx)
    models = [build_model(grid, column, layers) for layers in layer_sets]
    # The same column with no layers in it at all.  Its record holds everything
    # every epoch has in common and that has not moved: the transmit pulse, the
    # surface echo, the response of the smooth firn velocity gradient, and - the
    # one that actually matters - the long, slowly decaying wake a 2-D line
    # source leaves behind itself.  The layer echoes run 85-92 dB below the
    # transmit pulse and that wake sits near -95 dB, so an echo stands only 5-10
    # dB above something that did not move; a phase read off the sum is a
    # weighted average of the two and reports the motion far short.  Measured on
    # a 42 m test column: subtracting the twin cut the error from 5.3 cm to
    # 0.7 cm.  One twin serves the whole series, because the background is
    # exactly what does not change across it.
    model_twin = build_model(grid, column, [])

    dt = 0.9 * max_time_step(models[0].eps_min, models[0].mu_min,
                             models[0].dx, models[0].dz)
    # Run until the deepest part of the domain has had time to return, plus the
    # chirp's own length so the last echo is recorded whole rather than clipped
    # mid-sweep.  Integrated through the real slowness profile: a solid-ice
    # velocity would over-run by a fifth, and every extra step is the whole grid
    # again, once per epoch.
    t_end = float(echo_time(column, zlim[1], Z_ANT, tpd)) * 1.04
    t = np.arange(0.0, t_end, dt)
    pulse = lfm_chirp(system, t, tpd)
    src = np.array([[0.0, Z_ANT]])

    stamp = {
        "system": system.name,
        "f_start": system.f_start,
        "f_stop": system.f_stop,
        "tpd": tpd,
        "dx": dx,
        "xlim": xlim,
        "zlim": zlim,
        "t_end": t_end,
        "npml": NPML,
        "z_ant": Z_ANT,
        "epoch_years": EPOCH_YEARS,
        "n_epochs": n_epochs,
        "edge_cells": EDGE_CELLS,
        "column": stampable(COLUMN),
        "strain": STRAIN,
        "layering": stampable(LAYERS),
    }

    ppw = system.grid_step(models[0].eps_max, 1.0) / dx
    span = float(epoch_years[-1])
    print(system.summary())
    print(f"  grid {models[0].nx} x {models[0].nz} at dx = {dx:.4f} m "
          f"({ppw:.1f} nodes per wavelength at {system.f_stop / 1e6:.0f} MHz), "
          f"{len(t)} steps to {t_end * 1e6:.2f} us")
    print(f"  {n_epochs} epochs {EPOCH_YEARS:g} yr apart plus one layerless twin; "
          f"{len(layers_0)} layers")
    print(f"  apparent motion over the whole {span:g} yr: "
          f"{float(strain.displacement(LAYERS['top'], span)):+.3f} m at "
          f"{LAYERS['top']:.0f} m, "
          f"{float(strain.displacement(zlim[1], span)):+.3f} m at {zlim[1]:.0f} m")

    if render_from_cache(cache, render_only):
        arrays = load_arrays(cache, stamp=stamp)
        t_rec = arrays["t_rec"]
        records = [arrays[f"rec_{k}"] for k in range(n_epochs)]
        rec_twin = arrays["rec_twin"]
    else:
        records, t_rec = [], None
        for label, model in ([(f"epoch {k} (+{y:g} yr)", m)
                              for k, (y, m) in enumerate(zip(epoch_years, models))]
                             + [("layerless twin", model_twin)]):
            t0 = time.time()
            res = FDTD2D(model, dt, npml=NPML, mode="TM").run(
                src, pulse, src, outstep=1, progress=max(1, len(t) // 4),
            )
            print(f"  {label} in {time.time() - t0:.1f} s", flush=True)
            records.append(res.gather[:, 0, 0])
            t_rec = res.t
        rec_twin = records.pop()
        # The raw records are cached, not the differences, so how they are
        # combined stays a rendering decision.
        save_arrays(cache, stamp=stamp, t_rec=t_rec, rec_twin=rec_twin,
                    **{f"rec_{k}": r for k, r in enumerate(records)})

    # ---- compression ---------------------------------------------------
    profiles = [compressed(r - rec_twin, pulse, dt, system) for r in records]
    prof_raw = compressed(records[0], pulse, dt, system)
    tx_compressed = compressed(pulse, pulse, dt, system)
    # Referenced to the compressed transmit chirp, so 0 dB is the transmitted
    # pulse and every echo is the two-way loss below it -- the same reference
    # the other examples use, taken after compression because that is what the
    # traces being plotted have had done to them.
    ref = float(np.max(np.abs(tx_compressed)))

    def db(a):
        return 20 * np.log10(np.maximum(np.abs(a), 1e-30) / ref)

    envs = [db(p) for p in profiles]
    env_raw = db(prof_raw)
    depth_axis = column.depth_from_two_way_time(t_rec, z0=Z_ANT)

    # The record is read between the shallowest and deepest markers and nowhere
    # else.  Above the first one there is nothing to measure and plenty to
    # measure it on: the transmit pulse and the surface echo are 7 ns apart,
    # closer than either band can resolve, so no peak up there is a single
    # reflector.  Below the last one there is no reflector at all, only the
    # model's lower boundary.  The bounds are the scene's own, not fitted.
    resolution = system.range_resolution()
    t_first = float(column.two_way_time(max(0.5, layers_0[0].depth - 2 * resolution),
                                        z0=Z_ANT))
    usable = (t_rec > t_first) & (depth_axis < layers_0[-1].depth + 2 * resolution)

    # ---- the measurement, step by step ---------------------------------
    # Consecutive pairs, not everything against epoch 0.  Over four years the
    # shallowest markers move close to two metres, which is several resolution
    # cells and many fringes: the interferogram against epoch 0 would have
    # decorrelated and the unwrap would have nothing to hold on to.  A real
    # ApRES series is processed the same way, pair by pair, and accumulated.
    steps = []
    for k in range(n_epochs - 1):
        peaks_k = find_echoes(envs[k], usable, dt, system)
        if peaks_k.size < 2:
            # Nothing downstream can work with this, and every symptom of it -
            # an empty profile, a NaN fringe count - would surface far from the
            # cause.  The usual reason is a domain or a band that puts every
            # echo under the detection threshold.
            raise RuntimeError(
                f"epoch {k} has {peaks_k.size} usable echoes between "
                f"{layers_0[0].depth:.0f} and {layers_0[-1].depth:.0f} m; "
                f"there is no measurement to make."
            )
        steps.append(apparent_motion(profiles[k], profiles[k + 1], t_rec, dt,
                                     column, system, peaks_k))
    peaks = steps[0]["peaks"]

    step_truth = [strain.displacement(step["depth"], EPOCH_YEARS) for step in steps]
    resid_cm = np.concatenate([(s["displacement"] - u) * 100
                               for s, u in zip(steps, step_truth)])
    resid_env_cm = np.concatenate([(s["displacement_envelope"] - u) * 100
                                   for s, u in zip(steps, step_truth)])
    rms_cm = float(np.sqrt(np.mean(resid_cm**2)))
    rms_env_cm = float(np.sqrt(np.mean(resid_env_cm**2)))

    # Half a resolution cell: close enough that the next echo along is not a
    # candidate, loose enough to absorb the depth-conversion error.
    markers = track_markers(steps, tolerance=0.5 * resolution)

    predicted = strain.phase(depth_axis, EPOCH_YEARS, system.fc, z0=Z_ANT, wrap=True)
    turns = float(np.ptp(strain.phase(depth_axis[usable], EPOCH_YEARS, system.fc,
                                      z0=Z_ANT, wrap=False))) / (2 * np.pi)

    z_first = float(steps[0]["depth"][0])
    z_last = float(steps[0]["depth"][-1])
    print(f"  {len(peaks)} echoes between {z_first:.0f} and {z_last:.0f} m; "
          f"{turns:.1f} turns of phase per epoch step; "
          f"{len(markers)} markers followed through all {n_epochs} epochs")
    print(f"  per-step motion recovered to {rms_cm:.2f} cm rms over "
          f"{len(resid_cm)} measurements")
    # How many individual envelope estimates fall outside half a fringe is the
    # honest margin, and it is not the same question as the rms: the fringe
    # integer is chosen once per profile from the MEDIAN of them, so a handful
    # of bad ones cost nothing and a biased majority costs a whole fringe.  At
    # 750 MHz half a fringe is 5.6 cm and individual estimates do cross it.
    half_fringe_cm = 50 * system.fringe_spacing()
    outside = int(np.count_nonzero(np.abs(resid_env_cm) > half_fringe_cm))
    print(f"  envelope alone {rms_env_cm:.2f} cm (worst "
          f"{float(np.max(np.abs(resid_env_cm))):.2f} cm); {outside} of "
          f"{len(resid_env_cm)} estimates fall outside the {half_fringe_cm:.1f} cm "
          f"half-fringe, which the median over each profile absorbs")

    cum_err_cm = np.array([
        abs(m["displacement"][-1]
            - float(strain.displacement(m["depth0"], span))) * 100
        for m in markers
    ]) if markers else np.array([np.nan])
    # Whether the per-step errors average down over the series or accumulate is
    # a property of where they come from, and it is worth measuring rather than
    # assuming.  Independent errors would grow as sqrt(N); a per-reflector
    # offset that repeats every step grows as N.
    cum_rms = float(np.sqrt(np.mean(cum_err_cm**2)))
    independent = rms_cm * np.sqrt(len(steps))
    systematic = rms_cm * len(steps)
    print(f"  cumulative motion after {span:g} yr recovered to {cum_rms:.2f} cm rms "
          f"(independent per-step errors would give {independent:.2f} cm, "
          f"perfectly correlated ones {systematic:.2f} cm)")

    epoch_colours = plt.cm.viridis(np.linspace(0.05, 0.85, n_epochs))

    # ---- 1. the pulse itself -------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8))

    ax = axes[0]
    inside = t <= 1.25 * tpd
    ax.plot(t[inside] * 1e9, pulse[inside], lw=0.8, color="#2166ac")
    ax.set_xlabel("time (ns)")
    ax.set_ylabel("transmitted amplitude")
    ax.set_title(f"the chirp: {system.f_start / 1e6:.0f} to "
                 f"{system.f_stop / 1e6:.0f} MHz in {tpd * 1e9:.0f} ns", loc="left")

    ax = axes[1]
    nfft = 1 << 16
    spec = np.abs(np.fft.rfft(pulse, nfft))
    freq = np.fft.rfftfreq(nfft, dt)
    show = freq < 1.6 * system.f_stop
    ax.plot(freq[show] / 1e6, 20 * np.log10(spec[show] / spec.max()),
            lw=1.0, color="#2166ac")
    for f_edge in (system.f_start, system.f_stop):
        ax.axvline(f_edge / 1e6, color="#111", ls="--", lw=1.0)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("amplitude (dB)")
    ax.set_title(f"in band: {100 * system.fractional_bandwidth:.0f} percent about "
                 f"{system.fc / 1e6:.0f} MHz", loc="left")

    ax = axes[2]
    # The compressed wavelet is what every echo in the record looks like, and
    # its width is the resolution the measurement actually has -- wider than the
    # Fourier limit, because the Blackman band window trades main-lobe width for
    # a far lower skirt.
    #
    # It is measured on a *delayed* copy rather than on the transmit
    # autocorrelation, which peaks at zero lag and therefore has its whole
    # negative half outside the record: the width read off that is half the
    # real one.  Rolling by a few metres of range puts the wavelet where an echo
    # would be, with both flanks in view.
    to_range = C0 / (2.0 * np.sqrt(3.171))
    n_delay = int(round(6.0 / (to_range * dt)))
    probe = compressed(np.roll(pulse, n_delay), pulse, dt, system)
    comp_db = db(probe)
    lag = (t - t[n_delay]) * to_range
    span_lag = np.abs(lag) < 12.0 * resolution
    ax.plot(lag[span_lag], comp_db[span_lag], lw=1.3, color="#b2182b")
    width = float(np.count_nonzero(comp_db > np.max(comp_db) - 3.0) * dt * to_range)
    ax.axhline(np.max(comp_db) - 3.0, color="#999", ls=":", lw=1.0)
    ax.set_ylim(-60, 3)
    ax.set_xlabel("range offset (m)")
    ax.set_ylabel("amplitude (dB)")
    ax.set_title(f"compressed: {width:.2f} m wide, one fringe every "
                 f"{100 * system.fringe_spacing():.0f} cm", loc="left")

    fig.suptitle(
        f"{system.name} transmits {system.bandwidth / 1e6:.0f} MHz of bandwidth about "
        f"{system.fc / 1e6:.0f} MHz. The bandwidth sets how finely two layers can be "
        f"separated ({width:.2f} m); the centre frequency sets how far one has to move to "
        f"turn the phase once ({100 * system.fringe_spacing():.0f} cm).",
        x=0.008, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    save_figure(fig, outdir / "pulse.png", bare)
    plt.close(fig)
    print(f"  wrote {outdir / 'pulse.png'}")

    # ---- 2. the records ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.2))

    z_shown = (max(0.0, z_first - 4.0), z_last + 4.0)

    ax = axes[0]
    ax.plot(env_raw, depth_axis, lw=0.9, color="#c4c4c4", label="epoch 0, as recorded")
    for k in (0, n_epochs - 1):
        ax.plot(envs[k], depth_axis, lw=1.0, color=epoch_colours[k],
                label=f"epoch {k} (+{epoch_years[k]:g} yr), layers only")
    ax.plot(envs[0][peaks], depth_axis[peaks], "o", ms=5, mfc="none", color="#111",
            label=f"{len(peaks)} echoes measured")
    # Follow the echoes rather than the search window: the record runs deeper
    # than the deepest reflector, and an axis opened to it is blank paper.
    floor = float(np.percentile(envs[0][(depth_axis > z_shown[0])
                                        & (depth_axis < z_shown[1])], 1.0))
    ax.set_xlim(floor, float(np.max(envs[0])) + 6.0)
    ax.set_ylim(z_shown[1], z_shown[0])
    ax.set_xlabel("returned power (dB)")
    ax.set_ylabel("depth (m)")
    ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=1.0)
    ax.set_title(f"the compressed record, {span:g} years apart", loc="left")
    ax.legend_.set_zorder(6)

    # A ten-metre window with every epoch in it: this is the panel where the
    # echoes are seen marching down the record, one step per year.
    ax = axes[1]
    z_zoom = (max(0.0, z_first - 1.0), max(0.0, z_first - 1.0) + 10.0)
    win = (depth_axis >= z_zoom[0]) & (depth_axis <= z_zoom[1])
    for k in range(n_epochs):
        ax.plot(envs[k][win], depth_axis[win], lw=1.3, color=epoch_colours[k],
                label=f"+{epoch_years[k]:g} yr")
    ax.set_xlim(float(np.max(envs[0][win])) - 32.0, float(np.max(envs[0][win])) + 6.0)
    ax.set_ylim(z_zoom[1], z_zoom[0])
    ax.set_xlabel("returned power (dB)")
    ax.set_ylabel("depth (m)")
    ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=1.0, ncols=2)
    ax.set_title(
        f"{z_zoom[0]:.0f}-{z_zoom[1]:.0f} m, one visit a year: sinking "
        f"{float(strain.displacement(0.5 * sum(z_zoom), EPOCH_YEARS)) * 100:.0f} cm/yr",
        loc="left",
    )

    ax = axes[2]
    ax.plot(_break_wraps(predicted, usable), depth_axis, lw=1.6, color="#2166ac",
            label=f"from the imposed deformation, one {EPOCH_YEARS:g} yr step")
    ax.plot(steps[0]["phase"], steps[0]["depth"], "o", ms=5.5, color="#b2182b",
            label="measured, epoch 0 to 1")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_ylim(z_shown[1], z_shown[0])
    ax.set_xlabel("differential phase (rad)")
    ax.set_ylabel("depth (m)")
    ax.legend(fontsize=9, loc="lower left", frameon=True, framealpha=1.0)
    # A point at +pi and a prediction at -pi are the same phase, and on a
    # wrapped axis they sit at opposite edges of the panel.  The agreement is
    # therefore quoted rather than left to the eye.
    phase_resid = np.angle(np.exp(1j * (
        steps[0]["phase"]
        - strain.phase(steps[0]["depth"], EPOCH_YEARS, system.fc, z0=Z_ANT,
                       wrap=True))))
    ax.set_title(f"{turns:.1f} turns of phase per year; measured within "
                 f"{float(np.sqrt(np.mean(phase_resid**2))):.2f} rad of the "
                 f"prediction", loc="left")

    fig.suptitle(
        f"{system.name}: the same chirp into the same column, once a year for "
        f"{span:g} years. The echoes march down the record and the phase between "
        f"consecutive years turns {turns:.1f} times across it.",
        x=0.008, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, outdir / "record.png", bare)
    plt.close(fig)
    print(f"  wrote {outdir / 'record.png'}")

    # ---- 3. apparent motion with depth ---------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.2), width_ratios=[0.8, 1.2, 0.85])

    ax = axes[0]
    zz = np.linspace(0.0, zlim[1], 400)
    ax.plot(column.density(zz), zz, color="#333", lw=1.8)
    for layer in layers_0:
        ax.axhline(layer.depth, color="#2166ac", lw=0.6, alpha=0.5)
    ax.set_xlabel("relative density")
    ax.set_ylabel("depth (m)")
    ax.set_ylim(zlim[1], 0)
    ax.set_title(f"the column and its {len(layers_0)} markers", loc="left")

    ax = axes[1]
    disp = strain.displacement(zz, EPOCH_YEARS)
    ax.plot(disp * 100, zz, lw=2.0, color="#111", zorder=1,
            label=f"imposed, per {EPOCH_YEARS:g} yr step")
    for k, (step, u) in enumerate(zip(steps, step_truth)):
        ax.plot(step["displacement_envelope"] * 100, step["depth"], "s", ms=5,
                mfc="none", color="#aaa", zorder=2,
                label="envelope only" if k == 0 else None)
    for k, step in enumerate(steps):
        ax.plot(step["displacement"] * 100, step["depth"], "o", ms=6,
                color=epoch_colours[k + 1], zorder=3,
                label=f"measured, epoch {k} to {k + 1}")
    ax.axvline(0.0, color="#999", lw=1.0, ls=":")
    ax.set_xlabel("apparent motion (cm/yr)")
    ax.set_ylim(zlim[1], 0)
    ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=1.0)
    ax.set_title("what the radar recovers each year; positive is deeper",
                 loc="left")
    sign_change = float(zz[np.argmin(np.abs(disp))])
    if 0.5 < sign_change < zlim[1] - 0.5:
        ax.axhline(sign_change, color="#2166ac", ls="--", lw=1.2)
        ax.annotate(f"burial and thinning\ncancel at {sign_change:.0f} m",
                    (0.62, sign_change), xycoords=("axes fraction", "data"),
                    fontsize=10, va="top", color="#2166ac")

    ax = axes[2]
    for k, (step, u) in enumerate(zip(steps, step_truth)):
        ax.plot((step["displacement_envelope"] - u) * 100, step["depth"], "s", ms=5,
                mfc="none", color="#aaa", label="envelope only" if k == 0 else None)
    for k, (step, u) in enumerate(zip(steps, step_truth)):
        ax.plot((step["displacement"] - u) * 100, step["depth"], "o", ms=6,
                color=epoch_colours[k + 1],
                label="envelope + phase" if k == 0 else None)
    ax.axvline(0.0, color="#999", lw=1.0, ls=":")
    lim = max(6.0, 1.3 * float(np.max(np.abs(resid_env_cm))))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(zlim[1], 0)
    ax.set_xlabel("residual (cm)")
    ax.legend(fontsize=9, loc="upper left", frameon=True, framealpha=1.0)
    ax.set_title("residual, every step: measured minus imposed", loc="left")

    fig.suptitle(
        f"{system.name} recovers each year's apparent motion to {rms_cm:.2f} cm rms "
        f"across {len(resid_cm)} measurements. The envelope alone reaches "
        f"{rms_env_cm:.1f} cm, which is not the measurement but is what picks the right "
        f"one of the {100 * system.fringe_spacing():.0f} cm fringes - it has to stay "
        f"inside {50 * system.fringe_spacing():.0f} cm to do that.",
        x=0.008, ha="left", fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, outdir / "motion.png", bare)
    plt.close(fig)
    print(f"  wrote {outdir / 'motion.png'}")

    # ---- 4. the time series --------------------------------------------
    # A handful of markers followed through every epoch, spread over the depth
    # range rather than taken in order, so the panel carries the sign change
    # rather than five neighbouring layers doing the same thing.
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6))
    # Four, not six: the right-hand panel draws a wrapping sawtooth per marker
    # and by four years each has turned six or seven times, so more lines stop
    # being more information.
    show_n = min(4, len(markers))
    chosen = ([markers[i] for i in
               np.linspace(0, len(markers) - 1, show_n).astype(int)]
              if markers else [])
    colours = plt.cm.plasma(np.linspace(0.0, 0.78, max(show_n, 1)))
    fine = np.linspace(0.0, span, 121)

    ax = axes[0]
    for m, colour in zip(chosen, colours):
        ax.plot(fine, strain.trajectory(m["depth0"], fine) * 100, lw=1.4,
                color=colour, alpha=0.5)
        ax.plot(epoch_years, np.array(m["displacement"]) * 100, "o", ms=7,
                color=colour, label=f"{m['depth0']:.0f} m")
    ax.axhline(0.0, color="#999", lw=1.0, ls=":")
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("cumulative motion (cm)")
    ax.legend(fontsize=9, title="marker depth\nat year 0", title_fontsize=9,
              loc="best", frameon=True, framealpha=1.0)
    ax.set_title("measured against the imposed trajectory; positive is deeper",
                 loc="left")

    # The same thing the radar actually reads.  Only the FIRST year is shown:
    # over the whole series accum3 turns eighteen times per marker, and a panel
    # with eighteen branches per colour is a picket fence in which no reader can
    # tell which one a measurement landed on.  One year is countable, makes the
    # same point - the measured quantity is a wrapped phase that turns several
    # times between visits - and the left panel already carries all four years.
    # The prediction is drawn densely so the winding is visible between visits,
    # and broken at every wrap: a line joining +pi to -pi is not part of it.
    ax = axes[1]
    first = np.linspace(0.0, EPOCH_YEARS, 121)
    total_turns = []
    for m, colour in zip(chosen, colours):
        traj = strain.trajectory(m["depth0"], first)
        dtau = (column.two_way_time(m["depth0"] + traj, z0=Z_ANT)
                - column.two_way_time(m["depth0"], z0=Z_ANT))
        wrapped = np.angle(np.exp(-2j * np.pi * system.fc * dtau))
        ax.plot(first, _break_wraps(wrapped, np.ones_like(first, dtype=bool)),
                lw=1.4, color=colour, alpha=0.6)
        measured = -2.0 * np.pi * system.fc * np.array(m["dtau"])
        shown = epoch_years <= EPOCH_YEARS + 1e-9
        ax.plot(epoch_years[shown], np.angle(np.exp(1j * measured[shown])), "o",
                ms=7, color=colour)
        total_turns.append(abs(measured[-1]) / (2 * np.pi))
    # A little air either side so the year-0 and year-1 points are not
    # half-clipped by the spines.
    ax.set_xlim(-0.03 * EPOCH_YEARS, 1.04 * EPOCH_YEARS)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel("time (yr)")
    ax.set_ylabel("cumulative phase (rad)")
    ax.set_title(
        f"as the radar reads it, first year only: one turn per "
        f"{100 * system.fringe_spacing():.0f} cm"
        + (f" ({min(total_turns):.0f}-{max(total_turns):.0f} turns over all "
           f"{span:g})" if total_turns else ""),
        loc="left",
    )

    if chosen:
        reach = [abs(m["displacement"][-1]) * 100 for m in chosen]
        headline = (
            f"{system.name}: {n_epochs} visits, {EPOCH_YEARS:g} year apart, "
            f"{len(markers)} markers followed through all of them. Cumulative motion "
            f"after {span:g} years is recovered to "
            f"{float(np.sqrt(np.mean(cum_err_cm**2))):.2f} cm rms out of "
            f"{min(reach):.0f} to {max(reach):.0f} cm - and the phase it is read from "
            f"wrapped {min(total_turns):.0f} to {max(total_turns):.0f} times getting "
            f"there."
        )
    else:
        headline = f"{system.name}: {n_epochs} visits; no marker survived the tracking."
    fig.suptitle(headline, x=0.008, ha="left", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, outdir / "series.png", bare)
    plt.close(fig)
    print(f"  wrote {outdir / 'series.png'}")

    return dict(t=t_rec, profiles=profiles, peaks=peaks, steps=steps,
                markers=markers, rms_cm=rms_cm, rms_env_cm=rms_env_cm,
                cum_err_cm=cum_err_cm, turns=turns, column=column, strain=strain,
                system=system, epoch_years=epoch_years)


def main(quick=False, render_only=False, bare=False, out=None):
    use_talk_style()
    dx = 0.09 if quick else 0.035
    xlim = (-12.0, 12.0)
    zlim = (-3.0, 60.0) if quick else (-3.0, 140.0)
    # --quick shortens the series as well as coarsening the grid: it is a
    # pipeline check, and three epochs exercise the tracking and the
    # accumulation that two would not.
    n_epochs = 3 if quick else N_EPOCHS
    return run_series(APRES, xlim, zlim, dx, n_epochs=n_epochs,
                      render_only=render_only, bare=bare, out=out)


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
