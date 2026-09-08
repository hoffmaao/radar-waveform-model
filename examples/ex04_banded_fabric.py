"""Example 4 -- a banded fabric package: how an off-nadir fabric gets BRIGHT.

Example 3 establishes the ceiling: a single fabric interface can never beat
``r = deps / (4 eps)``, about -51 dB, and gradational smoothing only takes it
further down.  This example shows the one mechanism that beats the ceiling:
coherent stacking.  Strain banding -- alternating fabric bands of the kind ice
cores actually show -- turns N weak interfaces into a Bragg mirror whose
primaries add in phase at

    f_B = v / (2 d),

for band period ``d``.  Twenty bands of the same contrast as ex03's single
transition lift the response by roughly ``20 log10(2N)`` at resonance, from
about -54 dB to about -22 dB, competitive with the brightest density layering
anywhere in the column.

The resonance is the fingerprint.  The same package is run at two pulse
frequencies: on resonance the return is bright; a quarter-octave away it
collapses by tens of dB.  A reflector that appears in one band and vanishes in
another is banded fabric -- nothing else in the reflectivity budget does that.
The measured on/off contrast is a lower bound rather than the whole collapse:
the off-resonance window does not sit on silence but on a floor set by the deep
layering's transmission residual, since the two models differ in velocity below
the package and their deep echoes do not subtract out.  It is quoted
uncompressed, matching the transfer-matrix prediction it is compared against.
The compressed record the figures display gains the coherent on-resonance echo
about 2.8 dB more than the off-resonance scatter -- the run measures that
differential rather than asserting it -- so a processed product shows a wider
contrast still.

Fabrics and processing are ex03's, imported from it, but the geometry is this
example's own: the package dips at 12 degrees and crosses 350 m at x = 0, and
the return is isolated the same way, by differencing against a package-free
twin.  The banding is periodic in the PERPENDICULAR distance to the dipping
plane, so the Bragg condition holds along the specular ray.

The stratigraphy is this example's own too, and it is half the picture.  The
package sits inside conformable meteoric layering that runs from the firn to
the floor of the domain: flat-lying horizons whose specular return comes
straight back from nadir.  They are the reference the banded package has to be
read against, because they are what a radargram is normally made of.  The
record then carries both kinds of reflector at once -- a train of nadir echoes
down the whole column, and one brighter arrival that comes from 71 m off to the
side -- and the movie draws the horizons over the wavefield so the geometry of
the two is visible rather than inferred.  See :data:`LAYERING` for what the
horizons are and why nothing here is excluded from them.

The bright polarisation is modelled the same way as ex03, by feeding the
out-of-plane solver ``eps_xx``, and carries the same caveat: that substitution
is validated at nadir, and its accuracy along a 12 degree ray is an open
question tracked in the validation suite.  The resonance mechanism itself does
not depend on it -- the band phasing is set by geometry, not by the tensor.

Outputs land in ``figures/ex04/``.  ``--render-only`` re-renders from cache.
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
    load_snapshots,
    render_from_cache,
    save_figure,
    save_snapshots,
    stampable,
)

import ex03_dipping_fabric_transition as ex03
from ex03_dipping_fabric_transition import (
    ABOVE,
    BELOW,
    COLUMN,
    NPML,
    Z_ANT,
    eigen_permittivity,
)

#: This example's own geometry: deeper and gentler than ex03's.  The two move
#: together, because the specular point sits at x = -h sin(d) cos(d): at
#: ex03's 35 degrees a 350 m package would reflect from 165 m off axis,
#: outside any affordable domain, while at 12 degrees it reflects from 71 m --
#: inside a modestly widened one.  The deep, gentle package is also the one
#: that reads like the Ridge A feature, and its return arrives ~4.4 us,
#: cleanly separated from the shallow layering in the received panel.
DIP_DEG = 12.0
DEPTH_AT_X0 = 350.0

#: The transmitter sits right of centre.  The specular point lies updip of the
#: antenna, so an offset stretches the ray across the frame and makes the
#: off-nadir origin of the return impossible to misread -- and it moves the
#: specular point toward the middle of the domain, well clear of the edge
#: taper.  DEPTH_AT_X0 keeps its meaning (package depth at x = 0); under the
#: antenna itself the interface is deeper by tan(dip) times the offset.
SRC_X = 55.0

# ex03's helpers default to ex03's own dip and depth, so every call below
# passes this example's instead.  Assigning to ex03.DIP_DEG here would be
# shorter, but it would rewrite ex03's geometry for anything else sharing the
# interpreter -- importing this module would silently move ex03's specular
# point -- and an import should not have that effect.

from radarwave import (
    C0,
    FDTD2D,
    IceColumn,
    PropertyGrid,
    envelope_peak_time,
    gabor,
    max_time_step,
)
from radarwave.polarimetry import analytic
from radarwave.scenes import IceModelBuilder, conformal_layering, dipping_depth
from radarwave.viz import crop_snapshots, use_talk_style, wavefield_movie

OUT = Path(__file__).resolve().parent.parent / "figures" / "ex04"

#: On- and off-resonance pulse frequencies.  The package period is tuned to
#: FC_ON; FC_OFF sits a quarter-octave below, where the transfer matrix puts
#: the response tens of dB down.  Nothing else about the two runs differs.
FC_ON = 60e6
FC_OFF = 45e6

N_BANDS = 20

#: Fractional bandwidth of the transmitted wavelet.  This is the parameter the
#: whole demonstration turns on: the stack's resonance is only about 1/N wide
#: (a few MHz), while an impulse like the Blackman-Harris pulse is nearly an
#: octave wide, so an "off-resonance" impulse still excites the resonance
#: through its spectral skirt and the on/off contrast collapses to a couple of
#: dB.  Distinguishing banded fabric requires a band-limited waveform -- which
#: is what a real chirped system delivers after pulse compression (MCoRDS is
#: about 15 percent).  0.12 here makes the pulse a little narrower than the
#: resonance, the regime a real product operates in.
NARROWBAND_BW = 0.12

#: Where the wavelet is centred in the record.  ``gabor`` defaults to the
#: middle of its time vector, which for a full-length record would launch the
#: pulse a microsecond in; 350 ns is over five envelope sigmas at either
#: frequency, so the truncation at t = 0 is negligible and the echo still
#: lands well inside the record.
WAVELET_T0 = 3.5e-7

#: Each band edge is smoothed over this width.  A hard edge on a rectangular
#: grid staircases wherever it cuts the cells obliquely, as this 12 degree
#: package does (the ex03 problem); 0.15 m is under a twentieth of a wavelength
#: and costs about 1.6 dB across the whole stack, which the transfer-matrix
#: prediction accounts for because it is computed from the same smoothed
#: profile the grid carries.
BAND_EDGE_WIDTH = 0.15

#: The conformable stratigraphy, given explicitly rather than left to
#: :func:`conformal_layering`'s defaults.  Those defaults were set for ex01 and
#: ex03, whose domains stop a couple of hundred metres down; this one runs to
#: 560 m, and the nadir layering has to stay legible the whole way or the
#: package has nothing to be read against over most of the record.  Two changes,
#: and both are about the ice section, because below firn close-off density
#: banding contributes essentially nothing and every deep horizon here is an
#: acidity (conductivity) one:
#:
#: * ``sigma_factor`` 4-9 times the background rather than 2.5-3.5.  A volcanic
#:   acid spike of that size is ordinary in an ECM profile.  A conductivity
#:   horizon reflects on its *excess* over the background, so this roughly
#:   triples the brightest horizons' contrast -- about 10 dB -- which is what
#:   carries them clear of the display floor for the whole record.  The example
#:   measures the brightest one and prints it against the package, so the margin
#:   between the two is a number off the run rather than an assertion.
#: * ``ice_spacing`` 16-26 m rather than 24-36 m, giving 21 horizons between
#:   close-off and the floor of a 560 m domain instead of 15: enough for the
#:   record to read as layering rather than as a handful of isolated events.
#:
#: Nothing is excluded.  ex03 clears a band of layers around its event because
#: there the fabric transition and the layering are within a few dB of each
#: other and a coincident layer would bury it; here neither margin the run
#: prints is anywhere near that close.  The package comes back 18.5 dB above the
#: brightest deep horizon anywhere in the record, which is what says the
#: layering never competes with it; and the horizon that does land at the
#: package's own two-way time -- unresolved from it inside the compressed
#: wavelet -- returns -107.2 dB, 33.7 dB down, which is what says a layer at the
#: same two-way time cannot hide it.  That second number is the one the
#: no-exclusion choice rests on: cutting a gap in the stratigraphy exactly where
#: the answer is would be the one thing a viewer is entitled to be suspicious
#: of.
LAYERING = dict(sigma_factor=(4.0, 9.0), ice_spacing=(16.0, 26.0))


def band_period():
    """Bragg period d = v / (2 f) for the bright polarisation's mean speed."""
    ea = eigen_permittivity(**ABOVE)[0]
    eb = eigen_permittivity(**BELOW)[0]
    v = C0 / np.sqrt(0.5 * (ea + eb))
    return float(v / (2.0 * FC_ON))


def package_thickness(period):
    """Vertical thickness of the whole package (m).

    The bands are periodic in the perpendicular distance to the dipping plane,
    so the package measures ``N * period`` across its normal and more than that
    down a vertical line through it.  Every drawing of the package base goes
    through here, so the geometry figure and the movie cannot end up disagreeing
    about where it is.
    """
    return N_BANDS * period / np.cos(np.deg2rad(DIP_DEG))


def band_fraction(s, period):
    """Blend fraction ABOVE -> BELOW as a function of perpendicular distance.

    N smoothed bands of thickness ``period/2`` separated by equal gaps: a sum
    of tanh pairs, one per band, which returns to zero above and below the
    package so the stack is the only scatterer the twin difference keeps.
    """
    frac = np.zeros_like(s, dtype=float)
    for k in range(N_BANDS):
        lo = k * period
        hi = lo + period / 2.0
        frac += 0.5 * (np.tanh((s - lo) / BAND_EDGE_WIDTH)
                       - np.tanh((s - hi) / BAND_EDGE_WIDTH))
    return np.clip(frac, 0.0, 1.0)


def build_models(xlim, zlim, dx):
    """The banded model and its package-free twin, sharing everything else."""
    column = IceColumn(**COLUMN)
    grid = PropertyGrid.uniform(xlim, zlim, dx)

    def boundary(x):
        return dipping_depth(x, DEPTH_AT_X0, DIP_DEG)

    layers = conformal_layering(zlim[1], **LAYERING)

    period = band_period()
    banded = IceModelBuilder(grid, column, surface=0.0, air=True)
    banded.add_layers(layers)

    # Perpendicular distance to the dipping plane, so the banding is periodic
    # along the specular ray and the Bragg condition holds at the dip.
    s = (banded.depth - boundary(grid.x)[:, None]) * np.cos(np.deg2rad(DIP_DEG))
    frac = band_fraction(s, period)

    span = 22.0
    ramp = (np.clip((grid.x - xlim[0]) / span, 0, 1)
            * np.clip((xlim[1] - grid.x) / span, 0, 1))
    frac = frac * (np.sin(0.5 * np.pi * ramp) ** 2)[:, None]

    banded.set_fabric(
        np.ones(frac.shape, dtype=bool),
        dlambda=ABOVE["dlambda"] + (BELOW["dlambda"] - ABOVE["dlambda"]) * frac,
        lam_z=ABOVE["lam_z"] + (BELOW["lam_z"] - ABOVE["lam_z"]) * frac,
    )

    twin = IceModelBuilder(grid, column, surface=0.0, air=True)
    twin.add_layers(layers)

    return (column, grid, banded.finalize(npml=NPML), twin.finalize(npml=NPML),
            boundary, layers, period)


def stack_reflectivity(period, f):
    """Complex reflection coefficient of the band stack at frequencies ``f``.

    The transfer matrix is run over the actual smoothed band profile, the same
    construction validated against the FDTD to 0.4 dB for the ex03 transition
    width.  The layer product is carried over the whole frequency axis at once:
    the profile is a couple of thousand 0.02 m slabs and the record is long
    enough that the in-band bin count runs into the thousands, so a bin-at-a-time
    Python loop would be the dominant cost of every ``--render-only`` re-render.
    The layers still multiply in their original order, so the result is the
    same to the last bit.
    """
    ea = eigen_permittivity(**ABOVE)[0]
    eb = eigen_permittivity(**BELOW)[0]
    dz = 0.02
    s = np.arange(-8.0, N_BANDS * period + 8.0, dz)
    prof = ea + (eb - ea) * band_fraction(s, period)

    r = np.zeros(f.size, dtype=complex)
    band = (f >= 5e6) & (f <= 200e6)
    k0 = 2 * np.pi * f[band] / C0
    n = np.sqrt(prof.astype(complex))
    M = np.zeros((k0.size, 2, 2), dtype=complex)
    M[:, 0, 0] = M[:, 1, 1] = 1.0
    L = np.empty_like(M)
    for nj in n:
        ph = k0 * nj * dz
        cos_ph, sin_ph = np.cos(ph), np.sin(ph)
        L[:, 0, 0] = cos_ph
        L[:, 0, 1] = 1j * sin_ph / nj
        L[:, 1, 0] = 1j * nj * sin_ph
        L[:, 1, 1] = cos_ph
        M = M @ L
    n0, ns = n[0], n[-1]
    num = n0 * (M[:, 0, 0] + M[:, 0, 1] * ns) - (M[:, 1, 0] + M[:, 1, 1] * ns)
    den = n0 * (M[:, 0, 0] + M[:, 0, 1] * ns) + (M[:, 1, 0] + M[:, 1, 1] * ns)
    r[band] = num / den
    return r


def predicted_response(pulse, r, n_fft):
    """Envelope-peak stack response vs a sharp single interface, broadband.

    The pulse is filtered through the stack's reflectivity and the envelope peak
    is referenced to the incident pulse.  ``r`` depends only on the package and
    the frequency axis, never on which pulse is sounding it, so both runs share
    one :func:`stack_reflectivity` call.
    """
    S = np.fft.rfft(pulse, n_fft)
    echo = np.fft.irfft(S * r, n_fft)
    inc = np.fft.irfft(S, n_fft)
    return float(np.max(np.abs(analytic(echo)))) / float(np.max(np.abs(analytic(inc))))


def matched_filter(trace, wavelet):
    """Correlate the trace with the transmit wavelet: pulse compression.

    Each echo then wears the wavelet's autocorrelation -- the symmetric,
    zero-phase shape every processed radar product shows, whatever the
    reflector.  That matters for comparing with real radargrams: a thin
    permittivity band reflects roughly the *derivative* of the incident
    wavelet and a thin conductivity horizon a *replica*, so raw simulated
    traces wear different shapes than the compressed products they are held
    against.  On these noise-free traces this is a display transform, not an
    SNR gain -- the gain of real pulse compression is the time-bandwidth
    product of a long chirp, which an already-compact wavelet does not have.
    """
    w = np.asarray(wavelet, dtype=float)
    w = w / np.sqrt(np.sum(w ** 2))
    return np.correlate(np.asarray(trace, dtype=float), w, mode="same")


def geometry_figure(model, boundary, layers, period, px, pz, bare):
    """The scene: where the bands are, where the return comes from, and the
    one-dimensional profile the resonance argument runs on.

    The bands are 1.4 m in a half-kilometre domain, invisible at figure scale,
    so an inset zooms on the specular point where the reflection actually
    forms.  The right panel is the perpendicular permittivity profile for both
    eigenpolarisations: the bright one swings band to band, the dim one barely
    moves -- the polarisation selectivity and the Bragg period in one picture.
    """
    import matplotlib.pyplot as plt

    dl = np.asarray(model.eps["xx"], dtype=float) - np.asarray(model.eps["yy"], dtype=float)
    x, z = model.x, model.z

    # The scene is drawn at true aspect -- the whole point of the panel is that
    # the dip and the ray angle are the real ones -- so the panel is sized from
    # the domain rather than the other way round: a half-kilometre-deep, quarter-
    # kilometre-wide domain in a slot picked for a landscape figure would draw a
    # narrow strip and leave the rest of the slot blank.  Axes are placed in
    # inches, so the box the equal-aspect scene needs is the box it gets.
    scene_aspect = float((x[-1] - x[0]) / (z[-1] - z[0]))
    m_left, m_gap, m_right = 0.95, 1.15, 0.30
    m_bottom, m_top = 0.80, 0.62
    scene_h = 7.1
    scene_w = scene_h * scene_aspect
    prof_w = 5.6
    fig_w = m_left + scene_w + m_gap + prof_w + m_right
    fig_h = m_bottom + scene_h + m_top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes((m_left / fig_w, m_bottom / fig_h,
                       scene_w / fig_w, scene_h / fig_h))
    ax2 = fig.add_axes(((m_left + scene_w + m_gap) / fig_w, m_bottom / fig_h,
                        prof_w / fig_w, scene_h / fig_h))
    lim = float(np.abs(dl).max()) or 1.0
    ax.imshow(dl[::2, ::2].T, extent=(x[0], x[-1], z[-1], z[0]),
              aspect="equal", cmap="PuOr", vmin=-lim, vmax=lim)
    # The conformable horizons carry no fabric contrast, so they are invisible
    # in the eps_xx - eps_yy field this panel draws; drawn as lines they show
    # what the package is discordant with, which is the whole reason the scene
    # is worth a panel.  Faint, and the package outline heavy over them.
    for layer in layers:
        ax.plot(x, layer.depth_at(x), color="#999", lw=0.5, alpha=0.55)
    top = boundary(x)
    ax.plot(x, top, color="#555", lw=1.2)
    ax.plot(x, top + package_thickness(period), color="#555", lw=1.2)
    ax.plot([SRC_X], [Z_ANT], marker="v", ms=11, color="k")
    ax.annotate("antenna", (SRC_X + 4.0, Z_ANT - 6.0), fontsize=10)
    ax.plot([SRC_X, px], [Z_ANT, pz], "r--", lw=1.4)
    ax.plot([px], [pz], "ro", ms=7)
    ax.annotate("specular point", (px + 6.0, pz - 8.0), fontsize=10, color="r")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("depth (m)")

    # The inset is where the figure earns its keep: at domain scale the
    # package is a featureless stripe, and the banding only exists here.  Its
    # window is square in metres, so its height fraction comes from the panel's
    # own proportions -- given a fraction of its own the inset would draw square
    # inside a taller slot and float free of the corner it is anchored to.  The
    # bottom left is the one corner of a downdip package that stays empty, so
    # the zoom sits clear of the bands and the ray.
    ins_w = 0.46
    ins = ax.inset_axes([0.14, 0.03, ins_w, ins_w * scene_w / scene_h])
    ins.imshow(dl.T, extent=(x[0], x[-1], z[-1], z[0]), aspect="equal",
               cmap="PuOr", vmin=-lim, vmax=lim)
    half = 16.0
    ins.set_xlim(px - half, px + half)
    ins.set_ylim(pz + half, pz - half)
    ins.plot([px], [pz], "ro", ms=5)
    ins.tick_params(labelsize=7)
    ax.indicate_inset_zoom(ins, edgecolor="#333")

    svec = np.arange(-4.0, N_BANDS * period + 4.0, 0.02)
    frac = band_fraction(svec, period)
    ea = eigen_permittivity(**ABOVE)
    eb = eigen_permittivity(**BELOW)
    for k, (name, colour) in enumerate((("eps_xx (bright)", "#b2182b"),
                                        ("eps_yy (dim)", "#2166ac"))):
        prof = ea[k] + (eb[k] - ea[k]) * frac
        ax2.plot(prof, svec, lw=1.3, color=colour, label=name)
    ax2.axhspan(0.0, N_BANDS * period, color="#eee", zorder=0)
    ax2.set_ylim(svec[-1], svec[0])
    # Relative permittivity is a ratio and has no unit to give it.  Every other
    # axis in these figures carries one; writing a placeholder on this one would
    # be worse than leaving it, so it stays dimensionless and says so.
    ax2.set_xlabel("relative permittivity")
    ax2.set_ylabel("normal distance (m)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="lower left")

    save_figure(fig, OUT / "geometry.png", bare)
    plt.close(fig)


def main(quick=False, render_only=False, bare=False, out=None):
    if out is not None:
        global OUT
        OUT = Path(out)
    use_talk_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "snapshots.npz"
    import matplotlib.pyplot as plt

    dx = 0.5 if quick else 0.2
    xlim = (-130.0, 130.0)
    # Deep column, long record: the package top crosses 350 m under the
    # antenna, the specular return arrives ~4.4 us, and the deep conformable
    # layering fills the record around it.  --quick keeps the same geometry at
    # coarse resolution, so it still exercises the deep arrival.
    zlim = (-8.0, 560.0) if not quick else (-8.0, 430.0)
    t_end = 6.2e-6 if not quick else 5.2e-6

    column, grid, banded, twin, boundary, layers, period = build_models(xlim, zlim, dx)
    stamp = {
        "dx": dx, "xlim": xlim, "zlim": zlim, "t_end": t_end,
        "fc_on": FC_ON, "fc_off": FC_OFF, "n_bands": N_BANDS,
        "narrowband_bw": NARROWBAND_BW, "wavelet_t0": WAVELET_T0,
        "band_period": round(period, 6), "band_edge_width": BAND_EDGE_WIDTH,
        "npml": NPML, "z_ant": Z_ANT, "src_x": SRC_X, "dip_deg": DIP_DEG,
        "depth_at_x0": DEPTH_AT_X0,
        # Covers what this example overrides, not the rest of
        # ``conformal_layering``'s signature (seed, top, firn_base,
        # firn_spacing, thicknesses, amplitude, wavelengths): editing one of
        # those still re-uses a cache built from different stratigraphy.  The
        # gap is pre-existing and repo-wide -- ex01 and ex03 stamp nothing of
        # their layering at all -- and this entry narrows it rather than
        # opening it.  The complete fix is to stamp the derived layer geometry
        # (rounded depths and sigma factors) instead of the call arguments,
        # which invalidates the 2.3 GB cache here and costs a ~2.5 h re-run; do
        # it whenever a rebuild is being paid for anyway.
        "layering": LAYERING,
        "fabric_above": ABOVE, "fabric_below": BELOW,
        "column": stampable(COLUMN),
    }
    dt = 0.9 * max_time_step(banded.eps_min, banded.mu_min, banded.dx, banded.dz)
    t = np.arange(0.0, t_end, dt)
    src = np.array([[SRC_X, Z_ANT]])

    slant, px, pz = ex03.specular_geometry(SRC_X, DIP_DEG, DEPTH_AT_X0)
    print(f"banded package: {N_BANDS} bands, period {period:.2f} m "
          f"(Bragg {FC_ON/1e6:.0f} MHz), dip {DIP_DEG:.0f} deg")
    print(f"  specular point at x = {px:.1f} m, z = {pz:.1f} m")

    if render_from_cache(cache, render_only):
        panels, sx, sz, stimes, extra = load_snapshots(cache, stamp=stamp)
        t_rec = extra["t_rec"]
        traces = {k: extra[f"tr_{k}"] for k in ("on", "off")}
        recorded = {k: extra[f"rec_{k}"] for k in ("on", "off")}
        t_wave = {k: float(extra[f"t_wave_{k}"]) for k in ("on", "off")}
    else:
        snap_every = max(1, len(t) // 240)
        traces, recorded, t_wave, shown = {}, {}, {}, None
        for key, fc in (("on", FC_ON), ("off", FC_OFF)):
            pulse = gabor(fc, t, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
            t_wave[key] = envelope_peak_time(pulse, t)
            fields = []
            for m in (banded, twin):
                mm = m.with_properties(
                    eps={"xx": m.eps["xx"], "yy": m.eps["xx"], "zz": m.eps["zz"]})
                snap = (dict(snapshot_every=snap_every, snapshot_stride=1)
                        if m is banded and key == "on" else {})
                t0 = time.time()
                res = FDTD2D(mm, dt, npml=NPML, mode="TM").run(
                    src, pulse, src, outstep=1, progress=max(1, len(t) // 4), **snap)
                fields.append(res)
                print(f"  {fc/1e6:.0f} MHz ({'banded' if m is banded else 'twin  '}) "
                      f"in {time.time()-t0:.1f} s")
            traces[key] = fields[0].gather[:, 0, 0] - fields[1].gather[:, 0, 0]
            recorded[key] = fields[0].gather[:, 0, 0]
            if key == "on":
                shown = fields[0]
        t_rec = t
        stacks = [(shown.snapshots, f"{FC_ON/1e6:.0f} MHz, banded fabric")]
        # Crop the wavefield to the depth the record can actually reach, so the
        # wavefield and trace panels end at the same physical depth.  Showing
        # deeper than that would need the trace axis to run past t_end, leaving
        # a large blank strip beside depths no echo can return from.
        margin = 8.0
        depth_reached = float(
            column.depth_from_two_way_time(t_rec[-1] - t_wave["on"], z0=Z_ANT)
        )
        panels, sx, sz = crop_snapshots(
            stacks, shown.snapshot_x, shown.snapshot_z,
            xlim=(xlim[0] + margin, xlim[1] - margin),
            zlim=(zlim[0], min(zlim[1] - margin, depth_reached)))
        stimes = shown.snapshot_times
        save_snapshots(cache, panels, sx, sz, stimes, stamp=stamp,
                       t_rec=t_rec, tr_on=traces["on"], tr_off=traces["off"],
                       rec_on=recorded["on"], rec_off=recorded["off"],
                       t_wave_on=t_wave["on"], t_wave_off=t_wave["off"])

    # Drawn only once the cache has been accepted, so a stale one aborts before
    # any figure is written: a geometry.png from the current constants sitting
    # beside a movie and a resonance figure from the previous run is a set of
    # artifacts nothing on disk marks as mixed.
    geometry_figure(banded, boundary, layers, period, px, pz, bare)

    # ---- pulse compression, shared by the movie, the figure and the margins -
    # Short centred kernels: 'same' correlation aligns a kernel by its middle
    # sample, so the wavelet must be centred in its own window rather than
    # sitting at WAVELET_T0 of a record-length vector.  One kernel per
    # frequency, built once here and shared by everything downstream.
    #
    # Compression is not free of the quantity being measured, which is why this
    # example works in two domains and keeps each comparison inside one of
    # them.  On resonance the stack's echo carries a long coherent ringing tail
    # that the matched filter sums; off resonance the scatter has no such tail,
    # so compression gains the on-resonance echo by several dB more than the
    # off-resonance one.  A contrast therefore moves with the domain it is read
    # in, and the two domains are used for different jobs below.  The run prints
    # that differential and the contrast in both domains, so the split is a
    # measurement rather than a claim about one.
    dt_s = float(t_rec[1] - t_rec[0])
    wt = np.arange(0.0, 2.0 * WAVELET_T0, dt_s)
    kernels = {key: gabor(fc, wt, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
               for key, fc in (("on", FC_ON), ("off", FC_OFF))}
    rec_mf = {key: matched_filter(recorded[key], kernels[key]) for key in kernels}
    pkg_env = {key: np.abs(analytic(matched_filter(traces[key], kernels[key])))
               for key in kernels}
    tx_ref = {key: float(np.max(np.abs(analytic(rec_mf[key])))) for key in kernels}

    # ---- measurement: on/off-resonance level against the prediction --------
    # UNCOMPRESSED, both sides: ``predicted_response`` is an envelope-peak ratio
    # of the raw pulse filtered through the stack reflectivity, so measuring the
    # echo the same way keeps the comparison against the transfer matrix
    # like-for-like.  A compressed measurement against an uncompressed
    # prediction is not well posed however carefully it is captioned, and the
    # figures a reader is looking at are compressed, so the domain is named
    # wherever these levels are quoted.
    t_pred = float(echo_time(column, pz, Z_ANT, t_wave["on"], dip_deg=DIP_DEG))
    win = np.abs(t_rec - t_pred) < 0.12e-6
    level = {}
    for key in kernels:
        env = np.abs(analytic(traces[key]))
        tx = float(np.max(np.abs(analytic(recorded[key]))))
        level[key] = 20 * np.log10(float(np.max(env[win])) / tx)
    comp = {key: 20 * np.log10(float(np.max(pkg_env[key][win])) / tx_ref[key])
            for key in kernels}
    gain_gap = (comp["on"] - level["on"]) - (comp["off"] - level["off"])
    pulse_on = gabor(FC_ON, t_rec, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
    pulse_off = gabor(FC_OFF, t_rec, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
    n_fft = 8 * len(pulse_on)
    f_tm = np.fft.rfftfreq(n_fft, float(t_rec[1] - t_rec[0]))
    r_tm = stack_reflectivity(period, f_tm)
    peak_on = predicted_response(pulse_on, r_tm, n_fft)
    peak_off = predicted_response(pulse_off, r_tm, n_fft)
    print(f"  on  resonance ({FC_ON/1e6:.0f} MHz): measured {level['on']:.1f} dB "
          f"below the transmit pulse, uncompressed")
    print(f"  off resonance ({FC_OFF/1e6:.0f} MHz): measured {level['off']:.1f} dB")
    print(f"  measured on/off contrast {level['on'] - level['off']:+.1f} dB; "
          f"transfer matrix predicts {20*np.log10(peak_on/peak_off):+.1f} dB")
    print(f"  compression gains the on-resonance echo {gain_gap:+.1f} dB more "
          f"than the off-resonance one, so the same contrast reads "
          f"{comp['on'] - comp['off']:+.1f} dB compressed")

    # COMPRESSED, all three: unlike the on/off contrast above, these compare
    # signals inside the same record and the same domain, so the reference
    # cancels and the compressed one is the right choice -- it is the domain the
    # movie and the resonance panel display.  Each side is measured on the
    # record that contains it alone.  ``traces['on']`` is banded minus twin, so
    # the package sits there with no layering; ``recorded['on']`` minus that
    # difference is the package-free twin's own gather, so every horizon sits
    # there with no package.  Measuring the layering on the twin means the
    # search needs no upper cut to keep the package's compressed sidelobes out,
    # which is what lets it run to the floor of the record and include the
    # horizons that coincide with the package.  Those are the ones a viewer
    # would suspect of hiding it, and they are 30-odd dB down: the direct
    # evidence for laying this column with no exclusion gap.
    twin_env = np.abs(analytic(
        matched_filter(recorded["on"] - traces["on"], kernels["on"])))
    deep = t_rec > 2.0e-6
    pkg_db = comp["on"]
    lay_db = 20 * np.log10(float(np.max(twin_env[deep])) / tx_ref["on"])
    coin_db = 20 * np.log10(float(np.max(twin_env[win])) / tx_ref["on"])
    print(f"  compressed: package alone {pkg_db:.1f} dB, brightest deep nadir "
          f"horizon {lay_db:.1f} dB, margin {pkg_db - lay_db:+.1f} dB")
    print(f"  horizon coincident with the package {coin_db:.1f} dB, "
          f"margin {pkg_db - coin_db:+.1f} dB")

    # ---- movie -------------------------------------------------------------
    # The trace panel shows the received record pulse-compressed, the way a
    # processed product presents it.  The compression is applied to the whole
    # record up front and revealed progressively; near the reveal edge that
    # implies half a kernel of lookahead, invisible at movie frame rates.
    #
    # Two kinds of reflector in one frame, so the frame has to tell them apart:
    # the conformable horizons faint, the package outline heavy and in the same
    # near-black the trace panel marks the package with.  Horizons below the
    # cropped panel are dropped rather than drawn off the axis.
    #
    # Drawn, not imaged: the horizons' own reflections cannot be shown in this
    # panel, and lowering ``alpha_knee`` to reach them does not work.  They come
    # back around 70 dB under the incident field, which is where the 2-D line
    # source's wake also sits, so every opacity that reveals a horizon reveals
    # the wake with it and the frame turns to speckle (tried at 0.006 and
    # 0.0015).  The wavefield panel carries the geometry and the trace panel,
    # which has the dynamic range for it, carries the returns.
    layer_lines = [(grid.x, layer.depth_at(grid.x)) for layer in layers
                   if layer.depth < sz[-1]]
    package_style = dict(color="#111", lw=1.4, alpha=0.95)
    layer_times = [float(echo_time(column, float(layer.depth_at(SRC_X)), Z_ANT,
                                   t_wave["on"]))
                   for layer in layers]
    wavefield_movie(
        OUT / "banded_fabric.mp4", panels[0][0], sx, sz, stimes,
        contours=layer_lines + [
            (grid.x, boundary(grid.x), package_style),
            (grid.x, boundary(grid.x) + package_thickness(period), package_style)],
        trace={"t": t_rec,
               "series": [(rec_mf["on"], "received", "#b2182b")],
               "db": True,
               # Every nadir horizon's two-way time as a faint guide, the
               # package's as the one labelled marker: the record then reads as
               # a train of layer echoes with a single arrival that belongs to
               # none of them.
               "guides": layer_times,
               "markers": [(t_pred, "banded fabric")],
               "tlim": float(t_rec[-1]),
               "xlim": (-115.0, 0.0),
               # Two words and a unit: a movie frame is grabbed from the canvas,
               # so an over-long label is cut off rather than fitted.  The
               # reference is the compressed transmit pulse, as in the
               # resonance panel and the margins printed against it.
               "xlabel": "compressed return (dB)",
               # No panel title: the axis label already says what the trace is,
               # and the frames are to carry no other words.
               "title": ""},
        # No heading and no caption: the frame goes on a slide that carries its
        # own words, and the panel titles already say what each side is.
        title="",
        bare=bare, trace_width=0.5,
    )

    # ---- resonance figure --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8))
    ax = axes[0]
    band = (f_tm > 20e6) & (f_tm < 120e6)
    ax.plot(f_tm[band] / 1e6, 20 * np.log10(np.maximum(np.abs(r_tm[band]), 1e-12)),
            color="#333", lw=1.6, label="transfer matrix")
    # The two sounding frequencies, carried by the legend rather than by
    # callouts on the curve: a legend entry is a label, and the figure is not to
    # hold prose.  No dB here: the right panel's axis is the one place in this
    # figure that states an absolute level, so nothing can contradict it.
    for key, fc, colour in (("on", FC_ON, "#b2182b"), ("off", FC_OFF, "#2166ac")):
        ax.axvline(fc / 1e6, color=colour, ls="--", lw=1.2,
                   label=f"{fc/1e6:.0f} MHz")
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("reflectivity (dB)")
    ax.grid(alpha=0.25)
    # Opaque frame: the transfer-matrix curve runs its deep nulls straight
    # through this corner, and at slide size a frameless legend reads the curve
    # as part of its own text ("60|MHz").
    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=1.0)

    ax = axes[1]
    for key, fc, colour in (("on", FC_ON, "#b2182b"), ("off", FC_OFF, "#2166ac")):
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(np.maximum(pkg_env[key] / tx_ref[key], 1e-12))
        ax.plot(db, t_rec * 1e6, lw=1.1, color=colour,
                label=f"{fc/1e6:.0f} MHz")
    ax.axhline(t_pred * 1e6, color="#111", ls=":", lw=1.0)
    ax.set_xlim(-140, -60)
    ax.set_ylim(t_rec[-1] * 1e6, 0)
    ax.set_xlabel("compressed return (dB)")
    ax.set_ylabel("two-way time (us)")
    # The caveat that used to sit in this corner -- the off-resonance level is a
    # floor set by the deep-layer transmission residual, so the measured
    # contrast is a lower bound -- is in the module docstring and the README.
    # The figure carries no prose.
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right", frameon=True, framealpha=1.0)
    fig.tight_layout()
    save_figure(fig, OUT / "resonance.png", bare)
    plt.close(fig)
    print(f"  wrote {OUT / 'geometry.png'}, {OUT / 'banded_fabric.mp4'} "
          f"and {OUT / 'resonance.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write all outputs and caches under DIR")
    p.add_argument("--bare", action="store_true",
                   help="strip titles, notes, annotations and legends for slides")
    a = p.parse_args()
    try:
        main(quick=a.quick, render_only=a.render_only, bare=a.bare, out=a.out)
    except StaleCache as exc:
        raise SystemExit(str(exc))
