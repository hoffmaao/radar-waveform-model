"""Example 1 -- a radar wave propagating through meteoric ice.

A 60 MHz pulse is launched from a surface antenna into a firn/ice column and
reflects off the dielectric contrasts that make up meteoric internal layering:

* density (permittivity) contrasts in the firn, strongest in the top ~100 m and
  fading as the pore space closes;
* volcanic acid horizons below the firn/ice transition, which are conductivity
  contrasts and carry most of the deep internal stratigraphy.

The layers are conformable: every one shares a single undulation shape with the
amplitude growing downwards, matching the stratigraphy in the 350-450 m window
of the Ridge A radargram.  Example 3 uses exactly this layering and adds a
fabric transition cutting across it.

Outputs (in ``figures/ex01/``)

* ``wavefield.mp4``   -- the propagating wavefield
* ``model.png``       -- permittivity and conductivity of the ice model
* ``trace.png``       -- the recorded trace, with the layer depths marked
* ``radargram.png``   -- a common-offset section (with ``--radargram``)

Run with ``--quick`` for a small, fast version while iterating.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from radarwave import (
    C0,
    FDTD2D,
    IceColumn,
    PropertyGrid,
    blackharrispulse,
    dominant_frequency,
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
# The record runs long enough for the deepest layer's echo to return.  The last
# stretch carries the wave interacting with the bottom of the model rather than
# with the ice, so the display is cropped there.
T_MAX = 5.9e-6


def _plot_section(data, positions, t, column, layers):
    """Draw the common-offset section the way a processed radargram is shown."""
    import matplotlib.pyplot as plt
    from scipy.signal import hilbert

    # Reference the display to the strongest internal return, not to the
    # surface reflection, which is 40 dB above everything else and would leave
    # the layering invisible.
    gained = data * np.maximum(t, 1e-12)[:, None] ** 2
    env = np.abs(hilbert(gained, axis=0))
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
                column.two_way_time(l.depth_at(np.asarray(positions))) * 1e6,
                color="#2166ac", lw=0.8, alpha=0.45)
    ax.set_ylim(T_MAX * 1e6, 0)
    ax.annotate("blue: layers put into the model", (0.015, 0.03),
                xycoords="axes fraction", fontsize=11, color="#2166ac")
    fig.tight_layout()
    fig.savefig(OUT / "radargram.png")
    plt.close(fig)


def main(quick=False, radargram=False, processes=None):
    use_talk_style()
    OUT.mkdir(parents=True, exist_ok=True)

    dx = 0.5 if quick else 0.30
    xlim = (-60.0, 60.0) if quick else (-105.0, 105.0)
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
    model = builder.finalize(npml=NPML)

    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, t_end, dt)
    pulse = blackharrispulse(FC, t)
    fpeak = dominant_frequency(pulse, dt)
    lam_ice = C0 / np.sqrt(3.17) / fpeak

    sim = FDTD2D(model, dt, npml=NPML, mode="TM")
    print(sim.summary())
    print(
        f"peak frequency {fpeak / 1e6:.1f} MHz, wavelength in ice {lam_ice:.2f} m "
        f"({lam_ice / dx:.1f} nodes), {len(t)} time steps, {len(layers)} layers"
    )

    src = np.array([[0.0, -1.0]])  # antenna 1 m above the snow surface
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
    margin = 0.06 * (xlim[1] - xlim[0])
    # Crop the wavefield panel at the depth the displayed record reaches, so
    # its depth axis and the trace panel's time axis read across at the same
    # place -- and so neither shows the wave arriving at the bottom of the
    # model, which T_MAX exists to keep out of the picture.
    z_movie = min(zlim[1] - 0.5 * margin, float(column.depth_from_two_way_time(T_MAX)))
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
            "guides": list(column.two_way_time(np.array([l.depth for l in layers]))),
            "tlim": float(column.two_way_time(float(sz[-1]))),
            "xlim": (-125.0, 5.0),
            "title": "received at the transmit antenna",
        },
        title="Radar wave through meteoric ice",
        subtitle=(
            f"{fpeak / 1e6:.0f} MHz pulse. Left: the total electric field. Right: the trace "
            "the antenna records, filling in as the wave travels. Dotted lines mark the "
            "two-way time of each layer put into the model."
        ),
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
    fig.savefig(OUT / "model.png")
    plt.close(fig)

    # ---- recorded trace ------------------------------------------------
    from scipy.signal import hilbert

    trace = res.gather[:, 0, 0]
    env = np.abs(hilbert(trace))
    twtt = res.t * 1e6

    # The monostatic antenna records its own transmitted field, which is 60 dB
    # above every internal reflection, so reference the display to the strongest
    # return below the surface rather than to the global peak.
    # Scale on the part of the record that is actually displayed: past T_MAX
    # the trace is showing the bottom of the model, and letting that set the
    # normalisation flattens everything else to nothing.
    below = (res.t > 1.2e-7) & (res.t < T_MAX)
    ref = np.max(env[below])
    db = 20 * np.log10(np.maximum(env, 1e-30) / ref)

    fig, axes = plt.subplots(1, 2, figsize=(11, 8), sharey=True, width_ratios=[1, 1])
    axes[0].plot(trace / np.max(np.abs(trace[below])), twtt, lw=0.7, color="#222")
    axes[0].set_xlim(-6, 6)
    axes[0].set_xlabel("amplitude / strongest internal return")
    axes[0].set_ylabel("two-way time (us)")
    axes[0].set_title("wiggle trace", loc="left")

    axes[1].plot(db, twtt, lw=1.2, color="#d62728")
    axes[1].set_xlim(-45, 12)
    axes[1].set_xlabel("envelope (dB re. strongest internal return)")
    axes[1].set_title("envelope, log scale", loc="left")

    layer_times = column.two_way_time(np.array([l.depth for l in layers])) * 1e6
    for ax in axes:
        for lt in layer_times:
            ax.axhline(lt, color="#1f77b4", lw=0.7, alpha=0.4)
        ax.set_ylim(T_MAX * 1e6, 0)
    axes[0].annotate(
        "direct field\n(transmit antenna)", (0.0, 0.02), fontsize=10, color="#222",
        ha="center", va="top",
    )
    fig.suptitle(
        "Recorded trace: every peak lines up with a layer put into the model "
        "(blue)", x=0.01, ha="left", fontsize=15,
    )
    fig.tight_layout()
    fig.savefig(OUT / "trace.png")
    plt.close(fig)
    print(f"  wrote {OUT / 'model.png'}, {OUT / 'trace.png'}")

    # ---- optional common-offset section --------------------------------
    if radargram:
        # Wider trace spacing than you would use in the field: at this depth a
        # single shot is ~1.5e10 node-updates, so the section is the expensive
        # part of the example by a wide margin.
        step = 4.0 if quick else 14.0
        xs = np.arange(xlim[0] + 20, xlim[1] - 20 + 1e-9, step)
        shots = np.column_stack([xs, np.full_like(xs, -1.0)])
        print(f"  running {len(xs)} shots for the common-offset section...")
        t0 = time.time()
        co = run_common_offset(
            model, dt, shots, shots, pulse, npml=NPML, mode="TM", processes=processes
        )
        print(f"  done in {time.time() - t0:.1f} s")
        data = co.common_offset
        _plot_section(data, co.src[:, 0], co.t, column, layers)
        np.savez_compressed(OUT / "radargram.npz", data=data, t=co.t, x=co.src[:, 0])
        print(f"  wrote {OUT / 'radargram.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true", help="small, fast version")
    p.add_argument("--radargram", action="store_true", help="also run a multi-shot section")
    p.add_argument("--processes", type=int, default=None, help="parallel shots")
    a = p.parse_args()
    main(quick=a.quick, radargram=a.radargram, processes=a.processes)
