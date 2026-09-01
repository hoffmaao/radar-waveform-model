"""Example 1 -- a radar wave propagating through meteoric ice.

A 60 MHz pulse is launched from a surface antenna into a firn/ice column and
reflects off the dielectric contrasts that make up meteoric internal layering:

* density (permittivity) contrasts in the firn, strongest in the top ~100 m and
  fading as the pore space closes;
* volcanic acid horizons below the firn/ice transition, which are conductivity
  contrasts and carry most of the deep internal stratigraphy.

The layers are conformable: every one shares a single undulation shape with the
amplitude growing downwards, matching the stratigraphy in the 350-450 m window
of the Ridge A radargram.  Example 3 uses this same layering, minus a band
cleared around its fabric transition's arrival, and adds the transition cutting
across it.

An ice/bedrock interface sits at the base of the column, and it is there to be
a yardstick.  Internal layering is a dielectric contrast of a few parts in a
thousand; a bed is a contrast of tens of percent.  Putting both in one record
turns "the layering is faint" from a word into a number - the bed comes back
about 25 dB above the brightest layer in it, and that ratio is the reason
internal stratigraphy needs the dynamic range it does.  See :data:`BED_DEPTH`.

Outputs (in ``figures/ex01/``)

* ``wavefield.mp4``   -- the propagating wavefield
* ``model.png``       -- permittivity and conductivity of the ice model
* ``trace.png``       -- the recorded trace, with the layer depths and the bed
  marked, and the layer echoes measured against the bed
* ``radargram.png``   -- a common-offset section (with ``--radargram``)
* ``radargram.npz``   -- the section's samples, saved before plotting so the
  figure can be redrawn without re-running the shots

Run with ``--quick`` for a small, fast version while iterating.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import echo_time, save_figure

from radarwave import (
    C0,
    EPS0,
    FDTD2D,
    IceColumn,
    MU0,
    PropertyGrid,
    blackharrispulse,
    dominant_frequency,
    envelope_peak_time,
    max_time_step,
    run_common_offset,
)
from radarwave.scenes import IceModelBuilder, conformal_layering
from radarwave.viz import (
    crop_snapshots,
    plot_model,
    plot_radargram,
    use_talk_style,
    wavefield_movie,
)

OUT = Path(__file__).resolve().parent.parent / "figures" / "ex01"
FC = 60e6  # source frequency (Hz); a classic deep-sounding band
NPML = 12
Z_ANT = -1.0  # antenna 1 m above the snow surface
# The record runs long enough for the deepest layer's echo to return.  The last
# stretch carries the wave interacting with the bottom of the model rather than
# with the ice, so the display is cropped there.
# Far enough past the bed echo at 5.69 us that it is not jammed against the
# bottom of every panel, and short of the 6.39 us the domain floor would take.
T_MAX = 6.05e-6

#: An ice/bedrock interface at the base of the column, and what it is for: a
#: yardstick.  Internal layering is a contrast of a few parts in a thousand; a
#: bed is a contrast of tens of percent, -16.0 dB at normal incidence.  Without
#: one in the same record the layer amplitudes have nothing to be read against.
#:
#: What the record then shows is not what the interface coefficients suggest,
#: and that is the point.  At the antenna the bed arrives *level with* the
#: brightest firn band - within a couple of tenths of a dB - because the 480 m
#: of extra two-way path costs it about as much as its interface gains it.  The
#: example prints that budget: spreading, absorption, and the layer
#: reflectivity the balance implies.
#:
#: 500 m puts the echo at 5.68 us, inside both the record and the T_MAX crop,
#: so the bed costs no extra simulation - only the 60 m of rock beneath it,
#: which the domain already had.
#:
#: eps = 6 is crystalline bedrock (granite and gneiss run 5-8), which gives
#: ``r = (n_ice - n_rock) / (n_ice + n_rock) = -16.0 dB``; the example measures
#: it and prints both.  sigma = 1 mS/m is likewise ordinary for shield rock,
#: and does almost nothing to the reflection - it is there so the wave
#: transmitted into the rock is absorbed rather than rattling around under the
#: interface.
#:
#: Flat, deliberately.  A dipping bed sends its specular return off nadir and
#: away from the antenna, which is example 3's subject; here the bed exists to
#: be measured against, so it is put where a monostatic antenna can see all of
#: it.
BED_DEPTH = 500.0
BED_EPS = 6.0
BED_SIGMA = 1.0e-3


def _plot_section(data, positions, t, column, layers, z_ant, t_wave, bare=False):
    """Draw the common-offset section the way a processed radargram is shown."""
    import matplotlib.pyplot as plt
    from radarwave.polarimetry import analytic

    # Reference the display to the strongest internal return, not to the
    # surface reflection, which is 40 dB above everything else and would leave
    # the layering invisible.
    gained = data * np.maximum(t, 1e-12)[:, None] ** 2
    env = np.abs(analytic(gained, axis=0))
    # Same crop as the trace figure: beyond T_MAX the section is imaging the
    # bottom of the model, not the ice.
    below = (t > 1.5e-7) & (t < T_MAX)
    ref = float(np.max(env[below]))

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    plot_radargram(
        ax, gained, positions, t, db=True, dyn_range=30, ref=ref,
        title=(f"Common-offset section through the modelled layering "
               f"({len(positions)} traces, {abs(positions[1] - positions[0]):.0f} m apart)"),
    )
    for l in layers:
        ax.plot(positions,
                echo_time(column, l.depth_at(np.asarray(positions)), z_ant, t_wave) * 1e6,
                color="#2166ac", lw=0.8, alpha=0.45)
    # Follow the record, clipped to T_MAX.  A fixed limit leaves most of the
    # panel blank whenever the record is shorter than T_MAX.
    ax.set_ylim(min(T_MAX, float(t[-1])) * 1e6, 0)
    ax.annotate("blue: layers put into the model", (0.015, 0.03),
                xycoords="axes fraction", fontsize=11, color="#2166ac")
    fig.tight_layout()
    save_figure(fig, OUT / "radargram.png", bare)
    plt.close(fig)


def main(quick=False, radargram=False, processes=None, bare=False, out=None):
    # An output override is what lets many runs share a machine: without it,
    # every invocation writes the same figures/exNN paths and a parameter sweep
    # destroys its own results.  The cache guard catches the mixing after the
    # fact; this prevents it.
    if out is not None:
        global OUT
        OUT = Path(out)
    use_talk_style()
    OUT.mkdir(parents=True, exist_ok=True)

    dx = 0.5 if quick else 0.30
    # Wide enough that the wavefront is not clipped by the boundary before it
    # reaches the deepest layers: at 450 m the front already spanned the old
    # +/-105 m domain, so the picture was a slice of the wave rather than the
    # wave.  Costs ~1.7x the cells, which the movie is worth.
    xlim = (-60.0, 60.0) if quick else (-180.0, 180.0)
    zlim = (-12.0, 160.0) if quick else (-12.0, 560.0)
    t_end = 2.2e-6 if quick else 6.2e-6

    column = IceColumn(thickness=1850.0, depth_bco=100.0, sigma_ice=1.5e-5)
    grid = PropertyGrid.uniform(xlim, zlim, dx)
    builder = IceModelBuilder(grid, column, surface=0.0, air=True)
    # The same conformable stratigraphy example 3 uses, so that example is
    # literally this one plus a fabric transition.
    # Layers stop well above the bottom of the model so their echoes land
    # inside the record, clear of anything the lower boundary does.
    layers = conformal_layering(min(zlim[1] - 60.0, 480.0))
    builder.add_layers(layers)
    # The bed goes in only when the domain is deep enough to hold it with room
    # for the rock beneath; --quick stops at 160 m and has no bed.
    bed_depth = BED_DEPTH if zlim[1] > BED_DEPTH + 40.0 else None
    if bed_depth is not None:
        builder.add_bed(lambda x: np.full_like(x, bed_depth),
                        eps=BED_EPS, sigma=BED_SIGMA)
    model = builder.finalize(npml=NPML)

    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, t_end, dt)
    pulse = blackharrispulse(FC, t)
    fpeak = dominant_frequency(pulse, dt)
    t_wave = envelope_peak_time(pulse, t)
    lam_ice = C0 / np.sqrt(3.17) / fpeak

    sim = FDTD2D(model, dt, npml=NPML, mode="TM")
    print(sim.summary())
    print(
        f"peak frequency {fpeak / 1e6:.1f} MHz, wavelength in ice {lam_ice:.2f} m "
        f"({lam_ice / dx:.1f} nodes), {len(t)} time steps, {len(layers)} layers"
    )

    src = np.array([[0.0, Z_ANT]])
    # The trace is recorded at one x, and the layers undulate: a layer sits a
    # couple of metres off its nominal depth under the antenna, which is 20 ns
    # of two-way time and as much as the wavelet and air-path corrections
    # together.  Mark the depth that is actually beneath the receiver.
    layer_depths_below_antenna = [float(l.depth_at(src[0, 0])) for l in layers]
    snap_every = max(1, len(t) // 260)
    t0 = time.time()
    res = sim.run(
        src,
        pulse,
        src,
        outstep=1,
        snapshot_every=snap_every,
        snapshot_stride=1,
        progress=max(1, len(t) // 8),
    )
    print(f"  shot finished in {time.time() - t0:.1f} s")

    # ---- movie ---------------------------------------------------------
    layer_lines = [
        (grid.x, l.depth_at(grid.x)) for l in layers if l.depth < zlim[1] - 8.0
    ]
    if bed_depth is not None:
        layer_lines.append((grid.x, np.full_like(grid.x, bed_depth)))
    t_bed = (float(echo_time(column, bed_depth, Z_ANT, t_wave))
             if bed_depth is not None else None)
    margin = 0.06 * (xlim[1] - xlim[0])
    # Crop the wavefield panel at the depth the displayed record reaches, so
    # its depth axis and the trace panel's time axis read across at the same
    # place -- and so neither shows the wave arriving at the bottom of the
    # model, which T_MAX exists to keep out of the picture.
    z_movie = min(
        zlim[1] - 0.5 * margin,
        float(column.depth_from_two_way_time(T_MAX - t_wave, z0=Z_ANT)),
    )
    panels, sx, sz = crop_snapshots(
        [(res.snapshots, "total electric field")],
        res.snapshot_x,
        res.snapshot_z,
        xlim=(xlim[0] + margin, xlim[1] - margin),
        zlim=(zlim[0], z_movie),
    )
    wavefield_movie(
        OUT / "wavefield.mp4",
        panels,
        sx,
        sz,
        res.snapshot_times,
        fps=25,
        gain_power=0.6,
        gamma=0.35,
        alpha_knee=0.03,
        background=None,
        contours=layer_lines,
        trace={
            "t": res.t,
            "series": [(res.gather[:, 0, 0], "received", "#b2182b")],
            "db": True,
            "guides": list(echo_time(column, layer_depths_below_antenna, Z_ANT, t_wave)),
            "markers": [] if t_bed is None else [(t_bed, "bed")],
            "tlim": float(echo_time(column, float(sz[-1]), Z_ANT, t_wave)),
            "xlim": (-125.0, 5.0),
            "title": "received at the transmit antenna",
        },
        title="Radar wave through meteoric ice",
        subtitle=(
            f"{fpeak / 1e6:.0f} MHz pulse. Left: the total electric field. Right: the trace "
            "the antenna records, filling in as the wave travels. Dotted lines mark the "
            "two-way time of each layer put into the model, the dashed line the bed at "
            f"{BED_DEPTH:.0f} m - which comes back tens of dB above any of them."
        ),
        bare=bare,
        trace_width=0.42,
    )
    print(f"  wrote {OUT / 'wavefield.mp4'}")

    # ---- model figure --------------------------------------------------
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), width_ratios=[1, 1, 0.75])
    plot_model(
        axes[0], model.eps["yy"], model.x, model.z, cmap="cividis",
        label="relative permittivity",
    )
    axes[0].set_title("permittivity", loc="left")
    plot_model(
        axes[1], model.sig["yy"] * 1e6, model.x, model.z, cmap="magma",
        label="conductivity (uS/m)",
    )
    axes[1].set_title("conductivity", loc="left")
    for ax in axes[:2]:
        ax.set_ylim(zlim[1], zlim[0])
        ax.set_xlim(*xlim)

    zz = np.linspace(0, zlim[1], 400)
    axes[2].plot(column.density(zz), zz, color="#1f77b4", lw=2)
    axes[2].set_xlabel("relative density")
    axes[2].set_ylabel("depth (m)")
    axes[2].set_ylim(zlim[1], 0)
    axes[2].set_title("background firn column", loc="left")
    axes[2].axhline(column.depth_bco, color="#888", ls="--", lw=1)
    axes[2].annotate("bubble close-off", (0.45, column.depth_bco - 6), fontsize=11, color="#222")
    fig.tight_layout()
    save_figure(fig, OUT / "model.png", bare)
    plt.close(fig)

    # ---- recorded trace ------------------------------------------------
    from radarwave.polarimetry import analytic

    trace = res.gather[:, 0, 0]
    env = np.abs(analytic(trace))
    twtt = res.t * 1e6

    # The monostatic antenna records its own transmitted field, which is 60 dB
    # above every internal reflection, so reference the display to the strongest
    # return below the surface rather than to the global peak.
    # Scale on the part of the record that is actually displayed: past T_MAX
    # the trace is showing the bottom of the model, and letting that set the
    # normalisation flattens everything else to nothing.
    #
    # The reference window starts at the shallowest layer rather than at a fixed
    # time, so it cannot pick up the tail of the surface return, and it ends
    # before the bed, so the layering is referenced to layering.
    below = (res.t > 1.2e-7) & (res.t < T_MAX)
    t_first_layer = float(echo_time(column, min(layer_depths_below_antenna),
                                    Z_ANT, t_wave))
    internal = below & (res.t > t_first_layer - 0.04e-6)
    if t_bed is not None:
        internal &= res.t < t_bed - 0.15e-6
    ref = np.max(env[internal])
    db = 20 * np.log10(np.maximum(env, 1e-30) / ref)
    z_ref = float(column.depth_from_two_way_time(
        float(res.t[internal][np.argmax(env[internal])]) - t_wave, z0=Z_ANT))

    if t_bed is not None:
        win = np.abs(res.t - t_bed) < 0.12e-6
        bed_db = float(np.max(db[win]))
        r_bed = 20 * np.log10(abs(
            (np.sqrt(3.171) - np.sqrt(BED_EPS)) / (np.sqrt(3.171) + np.sqrt(BED_EPS))))

        # Why a -16 dB interface arrives level with a -40-something dB one.
        # Spreading: a 2-D line source reflected off a plane is the direct field
        # of an image source at twice the range, so amplitude goes as
        # 1/sqrt(2z) and the ratio between two nadir reflectors is 10log10(z1/z2).
        # Absorption: two-way through the extra ice, at the local alpha =
        # (sigma/2) sqrt(mu0 / (eps0 eps)).
        spreading_db = 10.0 * np.log10(z_ref / bed_depth)
        zz_path = np.linspace(z_ref, bed_depth, 400)
        eps_path = np.mean(column.permittivity(zz_path)[:, :2], axis=1)
        alpha = 0.5 * column.conductivity(zz_path) * np.sqrt(
            MU0 / (EPS0 * eps_path))
        absorption_db = -8.686 * 2.0 * float(np.trapezoid(alpha, zz_path))
        path_db = spreading_db + absorption_db
        print(f"  bed at {bed_depth:.0f} m: echo at {t_bed * 1e6:.3f} us, "
              f"{bed_db:+.1f} dB against the brightest layer (at {z_ref:.0f} m); "
              f"interface coefficient {r_bed:.1f} dB")
        # bed_db = (r_bed + path_to_bed) - (r_layer + path_to_layer)
        #        = r_bed - r_layer + path_db,  so  r_layer = r_bed + path_db - bed_db.
        print(f"  the extra {bed_depth - z_ref:.0f} m of two-way path costs the bed "
              f"{abs(path_db):.1f} dB ({abs(spreading_db):.1f} spreading, "
              f"{abs(absorption_db):.1f} absorption), so that layer must be "
              f"reflecting about {r_bed + path_db - bed_db:.0f} dB - which is where "
              f"density banding sits")

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, width_ratios=[1, 1])
    # Both panels on the same reference, so the two halves of the figure agree
    # about what "the strongest layer" is -- and so the bed, which lands within
    # a decibel of it, is directly comparable in either panel.
    axes[0].plot(trace / np.max(np.abs(trace[internal])), twtt, lw=0.7, color="#222")
    axes[0].set_xlim(-6, 6)
    axes[0].set_xlabel("relative amplitude")
    axes[0].set_ylabel("two-way time (us)")
    axes[0].set_title("wiggle trace, scaled to the strongest layer",
                      loc="left", fontsize=12)

    axes[1].plot(db, twtt, lw=1.2, color="#d62728")
    axes[1].set_xlim(-45, 12 if t_bed is None else max(12, bed_db + 6))
    axes[1].set_xlabel("envelope (dB)")
    axes[1].set_title("envelope, dB re. the strongest layer", loc="left", fontsize=12)

    layer_times = echo_time(column, layer_depths_below_antenna, Z_ANT, t_wave) * 1e6
    for ax in axes:
        for lt in layer_times:
            ax.axhline(lt, color="#1f77b4", lw=0.7, alpha=0.4)
        if t_bed is not None:
            ax.axhline(t_bed * 1e6, color="#111", ls="--", lw=1.2)
        ax.set_ylim(T_MAX * 1e6, 0)
    if t_bed is not None:
        axes[1].annotate(f"bed, {bed_db:+.1f} dB", (0.04, t_bed * 1e6),
                         xycoords=("axes fraction", "data"),
                         textcoords="offset points", xytext=(0, -6),
                         fontsize=11, va="top")
    axes[0].annotate(
        "direct field\n(transmit antenna)", (0.0, 0.02), fontsize=10, color="#222",
        ha="center", va="top",
    )
    fig.suptitle(
        "Recorded trace: every peak lines up with a layer put into the model (blue)"
        + ("" if t_bed is None else
           f"; the bed (dashed) arrives within {abs(bed_db):.1f} dB of the brightest "
           f"of them, despite an interface {abs(r_bed):.0f} dB stronger"),
        x=0.01, ha="left", fontsize=14,
    )
    fig.tight_layout()
    save_figure(fig, OUT / "trace.png", bare)
    plt.close(fig)
    print(f"  wrote {OUT / 'model.png'}, {OUT / 'trace.png'}")

    # ---- optional common-offset section --------------------------------
    if radargram:
        # Wider trace spacing than you would use in the field: at this depth a
        # single shot is ~1.5e10 node-updates, so the section is the expensive
        # part of the example by a wide margin.  Budget hours, not minutes: the
        # wider domain costs ~1.7x the cells per shot *and* spans 23 shots at
        # this spacing rather than the 13 the old +/-105 m domain did, so the
        # section is about 3x what it was.  A full run measured ~2.4 hours at
        # --processes 10.
        #
        # The span and the spacing are deliberate: the point of the wider domain
        # is that the movie shows the whole wavefront, and narrowing the section
        # back would quietly undo that.  The cheap win is not to repeat the
        # shots -- the section is written to radargram.npz below, so retuning
        # only the figure wants a --render-only path that reads that back, the
        # way examples 2 and 3 do for their snapshots.  There is not one yet.
        step = 4.0 if quick else 14.0
        xs = np.arange(xlim[0] + 20, xlim[1] - 20 + 1e-9, step)
        shots = np.column_stack([xs, np.full_like(xs, Z_ANT)])
        print(f"  running {len(xs)} shots for the common-offset section...")
        t0 = time.time()
        co = run_common_offset(
            model, dt, shots, shots, pulse, npml=NPML, mode="TM", processes=processes
        )
        print(f"  done in {time.time() - t0:.1f} s")
        data = co.common_offset
        # Persist before plotting.  Anything matplotlib does wrong downstream --
        # including in the --bare path -- would otherwise throw away hours of
        # simulation for the sake of a figure that can be redrawn in seconds.
        np.savez_compressed(OUT / "radargram.npz", data=data, t=co.t, x=co.src[:, 0])
        _plot_section(data, co.src[:, 0], co.t, column, layers, Z_ANT, t_wave, bare)
        print(f"  wrote {OUT / 'radargram.npz'}, {OUT / 'radargram.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="small, fast version")
    p.add_argument("--radargram", action="store_true", help="also run a multi-shot section")
    p.add_argument("--processes", type=int, default=None, help="parallel shots")
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write all outputs and caches under DIR "
                        "instead of figures/exNN (for sweeps and "
                        "cluster array jobs)")
    p.add_argument("--bare", action="store_true",
                   help="strip titles, notes, annotations and legends for slides")
    a = p.parse_args()
    main(quick=a.quick, radargram=a.radargram, processes=a.processes, bare=a.bare,
         out=a.out)
