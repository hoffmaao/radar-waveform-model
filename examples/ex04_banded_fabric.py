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

Layering, fabrics and processing are ex03's, imported from it, but the geometry
is this example's own: the package dips at 12 degrees and crosses 350 m at
x = 0, through the same conformable stratigraphy, and the return is isolated
the same way, by differencing against a package-free twin.  The banding is
periodic in the PERPENDICULAR distance to the dipping plane, so the Bragg
condition holds along the specular ray.

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
    LAYER_GAP_HALFWIDTH,
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

    d_event, exclude = ex03.layer_exclusion(column, zlim, SRC_X, DIP_DEG, DEPTH_AT_X0)
    layers = conformal_layering(zlim[1], exclude=exclude)

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
            boundary, exclude, period)


def predicted_response(period, pulse, dt):
    """Envelope-peak stack response vs a sharp single interface, broadband.

    The transfer matrix is run over the actual smoothed band profile, the
    pulse is filtered through it, and the envelope peak is referenced to the
    incident pulse -- the same construction validated against the FDTD to
    0.4 dB for the ex03 transition width.
    """
    ea = eigen_permittivity(**ABOVE)[0]
    eb = eigen_permittivity(**BELOW)[0]
    dz = 0.02
    s = np.arange(-8.0, N_BANDS * period + 8.0, dz)
    prof = ea + (eb - ea) * band_fraction(s, period)

    n_fft = 8 * len(pulse)
    f = np.fft.rfftfreq(n_fft, dt)
    r = np.zeros(f.size, dtype=complex)
    for i, fq in enumerate(f):
        if fq < 5e6 or fq > 200e6:
            continue
        k0 = 2 * np.pi * fq / C0
        n = np.sqrt(prof.astype(complex))
        M = np.eye(2, dtype=complex)
        for nj in n:
            ph = k0 * nj * dz
            M = M @ np.array([[np.cos(ph), 1j * np.sin(ph) / nj],
                              [1j * nj * np.sin(ph), np.cos(ph)]])
        n0, ns = n[0], n[-1]
        num = n0 * (M[0, 0] + M[0, 1] * ns) - (M[1, 0] + M[1, 1] * ns)
        den = n0 * (M[0, 0] + M[0, 1] * ns) + (M[1, 0] + M[1, 1] * ns)
        r[i] = num / den
    S = np.fft.rfft(pulse, n_fft)
    echo = np.fft.irfft(S * r, n_fft)
    inc = np.fft.irfft(S, n_fft)
    peak = float(np.max(np.abs(analytic(echo)))) / float(np.max(np.abs(analytic(inc))))
    return peak, f, np.abs(r)


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


def geometry_figure(model, boundary, period, px, pz, bare):
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

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.6),
                             gridspec_kw={"width_ratios": [1.55, 1.0]})
    ax = axes[0]
    lim = float(np.abs(dl).max()) or 1.0
    ax.imshow(dl[::2, ::2].T, extent=(x[0], x[-1], z[-1], z[0]),
              aspect="equal", cmap="PuOr", vmin=-lim, vmax=lim)
    top = boundary(x)
    ax.plot(x, top, color="#555", lw=0.9)
    ax.plot(x, top + package_thickness(period), color="#555", lw=0.9)
    ax.plot([SRC_X], [Z_ANT], marker="v", ms=11, color="k")
    ax.annotate("antenna", (SRC_X + 4.0, Z_ANT - 6.0), fontsize=10)
    ax.plot([SRC_X, px], [Z_ANT, pz], "r--", lw=1.4)
    ax.plot([px], [pz], "ro", ms=7)
    ax.annotate("specular point", (px + 6.0, pz - 8.0), fontsize=10, color="r")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title(f"{N_BANDS} bands, period {period:.2f} m, dip {DIP_DEG:.0f} deg",
                 loc="left")

    # The inset is where the figure earns its keep: at domain scale the
    # package is a featureless stripe, and the banding only exists here.
    ins = ax.inset_axes([0.58, 0.04, 0.40, 0.34])
    ins.imshow(dl.T, extent=(x[0], x[-1], z[-1], z[0]), aspect="equal",
               cmap="PuOr", vmin=-lim, vmax=lim)
    half = 16.0
    ins.set_xlim(px - half, px + half)
    ins.set_ylim(pz + half, pz - half)
    ins.plot([px], [pz], "ro", ms=5)
    ins.tick_params(labelsize=7)
    ins.set_title("the bands, at the specular point", fontsize=8.5, pad=2)
    ax.indicate_inset_zoom(ins, edgecolor="#333")

    ax2 = axes[1]
    svec = np.arange(-4.0, N_BANDS * period + 4.0, 0.02)
    frac = band_fraction(svec, period)
    ea = eigen_permittivity(**ABOVE)
    eb = eigen_permittivity(**BELOW)
    for k, (name, colour) in enumerate((("eps_xx (bright)", "#b2182b"),
                                        ("eps_yy (dim)", "#2166ac"))):
        prof = ea[k] + (eb[k] - ea[k]) * frac
        ax2.plot(prof, svec, lw=1.3, color=colour, label=name)
    ax2.axhspan(0.0, N_BANDS * period, color="#eee", zorder=0)
    ax2.annotate(f"period {period:.2f} m = half a wavelength at "
                 f"{FC_ON/1e6:.0f} MHz",
                 (0.03, 0.985), xycoords="axes fraction", va="top", fontsize=9)
    ax2.set_ylim(svec[-1], svec[0])
    ax2.set_xlabel("relative permittivity")
    ax2.set_ylabel("distance along the interface normal (m)")
    ax2.set_title("the profile the wave meets", loc="left")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=9, loc="lower left")

    fig.tight_layout()
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

    column, grid, banded, twin, boundary, exclude, period = build_models(xlim, zlim, dx)
    stamp = {
        "dx": dx, "xlim": xlim, "zlim": zlim, "t_end": t_end,
        "fc_on": FC_ON, "fc_off": FC_OFF, "n_bands": N_BANDS,
        "narrowband_bw": NARROWBAND_BW, "wavelet_t0": WAVELET_T0,
        "band_period": round(period, 6), "band_edge_width": BAND_EDGE_WIDTH,
        "npml": NPML, "z_ant": Z_ANT, "src_x": SRC_X, "dip_deg": DIP_DEG,
        "depth_at_x0": DEPTH_AT_X0,
        "layer_gap_halfwidth": LAYER_GAP_HALFWIDTH,
        "layer_exclusion_band": tuple(round(b, 3) for b in exclude),
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
    geometry_figure(banded, boundary, period, px, pz, bare)

    # ---- measurement: on/off-resonance level against the prediction --------
    t_pred = float(echo_time(column, pz, Z_ANT, t_wave["on"], dip_deg=DIP_DEG))
    win = np.abs(t_rec - t_pred) < 0.12e-6
    level = {}
    for key in ("on", "off"):
        env = np.abs(analytic(traces[key]))
        tx = float(np.max(np.abs(analytic(recorded[key]))))
        level[key] = 20 * np.log10(float(np.max(env[win])) / tx)
    pulse_on = gabor(FC_ON, t_rec, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
    peak_on, f_tm, r_tm = predicted_response(period, pulse_on, float(t_rec[1] - t_rec[0]))
    pulse_off = gabor(FC_OFF, t_rec, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
    peak_off, _, _ = predicted_response(period, pulse_off, float(t_rec[1] - t_rec[0]))
    print(f"  on  resonance ({FC_ON/1e6:.0f} MHz): measured {level['on']:.1f} dB "
          f"below the transmit pulse")
    print(f"  off resonance ({FC_OFF/1e6:.0f} MHz): measured {level['off']:.1f} dB")
    print(f"  measured on/off contrast {level['on'] - level['off']:+.1f} dB; "
          f"transfer matrix predicts {20*np.log10(peak_on/peak_off):+.1f} dB")

    # Short centred kernels, shared by the movie panel and the figure: 'same'
    # correlation aligns a kernel by its middle sample, so the wavelet must be
    # centred in its own window rather than sitting at WAVELET_T0 of a
    # record-length vector.
    dt_s = float(t_rec[1] - t_rec[0])
    wt = np.arange(0.0, 2.0 * WAVELET_T0, dt_s)
    kernels = {key: gabor(fc, wt, bandwidth=NARROWBAND_BW, t0=WAVELET_T0)
               for key, fc in (("on", FC_ON), ("off", FC_OFF))}

    # ---- movie -------------------------------------------------------------
    # The trace panel shows the received record pulse-compressed, the way a
    # processed product presents it.  The compression is applied to the whole
    # record up front and revealed progressively; near the reveal edge that
    # implies half a kernel of lookahead, invisible at movie frame rates.
    rec_mf = matched_filter(recorded["on"], kernels["on"])
    wavefield_movie(
        OUT / "banded_fabric.mp4", panels[0][0], sx, sz, stimes,
        contours=[(grid.x, boundary(grid.x)),
                  (grid.x, boundary(grid.x) + package_thickness(period))],
        trace={"t": t_rec,
               "series": [(rec_mf, "received", "#b2182b")],
               "db": True,
               "guides": [t_pred],
               "tlim": float(t_rec[-1]),
               "xlim": (-125.0, 5.0),
               "xlabel": "matched-filtered return\n(dB re. compressed transmit pulse)",
               "title": "received at the antenna, pulse-compressed"},
        title=f"A banded fabric package at {DIP_DEG:.0f} degrees",
        subtitle=(f"{N_BANDS} bands at the Bragg period for {FC_ON/1e6:.0f} MHz, "
                  f"sounded with a {NARROWBAND_BW*100:.0f} percent-bandwidth "
                  "wavelet (the post-compression band of a real system). "
                  "Bright at resonance, gone a quarter-octave away - the "
                  "fingerprint of banded fabric."),
        bare=bare, trace_width=0.5,
    )

    # ---- resonance figure --------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.8))
    ax = axes[0]
    band = (f_tm > 20e6) & (f_tm < 120e6)
    ax.plot(f_tm[band] / 1e6, 20 * np.log10(np.maximum(r_tm[band], 1e-12)),
            color="#333", lw=1.6, label="stack response (transfer matrix)")
    # Staggered, and the off-resonance one right-aligned, so the two labels
    # cannot collide across the 15 MHz between their markers.
    marks = (("on", FC_ON, "#b2182b", "left", 1.5, -27.0),
             ("off", FC_OFF, "#2166ac", "right", -1.5, -47.0))
    for key, fc, colour, ha, dx_lbl, y_lbl in marks:
        ax.axvline(fc / 1e6, color=colour, ls="--", lw=1.2)
        ax.annotate(f"{fc/1e6:.0f} MHz pulse\nmeasured {level[key]:.0f} dB",
                    (fc / 1e6 + dx_lbl, y_lbl), ha=ha, fontsize=10, color=colour)
    ax.set_xlabel("frequency (MHz)")
    ax.set_ylabel("|R| (dB)")
    ax.set_title("the package is a bandpass reflector", loc="left")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")

    ax = axes[1]
    for key, fc, colour in (("on", FC_ON, "#b2182b"), ("off", FC_OFF, "#2166ac")):
        wavelet = kernels[key]
        mf = matched_filter(traces[key], wavelet)
        ref = float(np.max(np.abs(analytic(matched_filter(recorded[key], wavelet)))))
        env = np.abs(analytic(mf))
        with np.errstate(divide="ignore"):
            db = 20 * np.log10(np.maximum(env / ref, 1e-12))
        ax.plot(db, t_rec * 1e6, lw=1.1, color=colour,
                label=f"{fc/1e6:.0f} MHz")
    ax.axhline(t_pred * 1e6, color="#111", ls=":", lw=1.0)
    ax.set_xlim(-140, -60)
    ax.set_ylim(t_rec[-1] * 1e6, 0)
    ax.set_xlabel("matched-filtered return\n(dB re. compressed transmit pulse)")
    ax.set_ylabel("two-way time (us)")
    ax.set_title("same package, two pulses, after pulse compression", loc="left")
    ax.annotate("off-resonance level is a floor set by the\n"
                "deep-layer transmission residual, so the\n"
                "measured contrast is a lower bound",
                (0.03, 0.985), xycoords="axes fraction", va="top",
                fontsize=8.5, color="#444")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        f"{N_BANDS} fabric bands, period {period:.2f} m, dipping "
        f"{DIP_DEG:.0f} deg: bright at {FC_ON/1e6:.0f} MHz, collapsed at "
        f"{FC_OFF/1e6:.0f} MHz.", x=0.008, ha="left", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
