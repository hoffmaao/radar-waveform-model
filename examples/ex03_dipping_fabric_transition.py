"""Example 3 -- a fabric transition that is not at nadir but still returns energy.

The Ridge A radargram shows a faint, steeply dipping feature at 1000-1500 m
labelled "Deep Fabric Fold?".  This example models what produces it.

The scene is a sharp change in crystal-orientation fabric across a plane
dipping at 35 degrees.  Two things follow from that geometry, and both are the
point of the example:

1. **The energy that comes back to the antenna is not from beneath it.**  A
   monostatic antenna only records a specular return from the place where the
   interface normal points back at it.  For a plane dipping at ``delta``
   through depth ``h`` under the antenna, that place is offset sideways by
   ``h sin(delta) cos(delta)`` and the event is recorded at an apparent depth
   ``h cos(delta)`` -- shallower than the interface actually is.  Across a
   profile the feature therefore images with an apparent dip satisfying
   ``tan(dip_apparent) = sin(dip_true)``, which is the usual reason an
   unmigrated fabric feature looks shallower and gentler than it is.

2. **It is a polarisation-dependent reflector.**  A fabric contrast is a
   contrast in the permittivity *tensor*, so the two eigenpolarisations see
   different jumps.  Here the fabric changes so that the 179 deg polarisation
   sees a strong contrast and the 89 deg polarisation sees almost none: the
   same interface is bright in one polarisation and nearly invisible in the
   other.  A density or acidity layer cannot do that.

The scene also carries ordinary conformable meteoric layering, so the fabric
transition appears as a discordant feature cutting across a conformable
background, which is what the radargram looks like.

The reflection is only about -60 dB, so each polarisation is run twice -- with
and without the fabric contrast -- and the difference isolates it.  That also
cancels any residual boundary artefact, which matters at this amplitude.  The
cancellation is exact only until the transmitted wave starts returning from
reflectors *below* the transition: the two models have genuinely different
velocities down there, so those layers arrive at different times and do not
subtract out.  Everything before ~1.6 us is clean, which covers the specular
arrival; the movie's trace panel shows the raw recorded trace anyway.

Outputs land in ``figures/ex03/``.  ``--render-only`` re-renders from cache.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_snapshots, save_snapshots

from radarwave import (
    C0,
    FDTD2D,
    IceColumn,
    PropertyGrid,
    blackharrispulse,
    max_time_step,
    run_common_offset,
)
from radarwave.scenes import IceModelBuilder, conformal_layering, dipping_depth
from radarwave.viz import (
    crop_snapshots,
    plot_radargram,
    radargram_reference,
    use_talk_style,
    wavefield_movie,
)

OUT = Path(__file__).resolve().parent.parent / "figures" / "ex03"
FC = 60e6
NPML = 12

DIP_DEG = 35.0  # dip of the fabric transition
DEPTH_AT_X0 = 175.0  # depth of the transition directly under the antenna

# Fabric either side of the transition: a vertical single maximum over a
# horizontal one aligned with x.  Chosen so that lam_y barely changes across the
# boundary while lam_x jumps by 0.75, which makes the interface bright to one
# eigenpolarisation and nearly invisible to the other.
#
# With dlam = 0.8 below, the horizontal budget h = 1 - lam_z must be at least
# 0.8, so lam_z has to collapse across the transition and lam_x necessarily
# takes the large jump.  That fixes which polarisation is the bright one, so
# the example works it out from the permittivities rather than assuming.
ABOVE = dict(lam_z=0.80, dlambda=0.03)   # lam = (0.115, 0.085, 0.80)
BELOW = dict(lam_z=0.08, dlambda=0.80)   # lam = (0.860, 0.060, 0.08)


def eigen_permittivity(lam_z, dlambda):
    """Solid-ice eigenpermittivities for a given (lam_z, dlam) pair."""
    from radarwave.constants import DEPS_ICE, EPS_ICE_MEAN

    h = 1.0 - lam_z
    lam = np.array([(h + dlambda) / 2, (h - dlambda) / 2, lam_z])
    return EPS_ICE_MEAN + DEPS_ICE * (lam - 1.0 / 3.0)


def build_models(xlim, zlim, dx):
    """Return the model with the fabric transition and its contrast-free twin."""
    column = IceColumn(thickness=1850.0, depth_bco=90.0, sigma_ice=1.2e-5,
                       lam_z_sfc=ABOVE["lam_z"], lam_z_bed=ABOVE["lam_z"],
                       dlambda=lambda d: np.full_like(np.asarray(d, float), ABOVE["dlambda"]))
    grid = PropertyGrid.uniform(xlim, zlim, dx)

    def boundary(x):
        return dipping_depth(x, DEPTH_AT_X0, DIP_DEG)

    layers = conformal_layering(zlim[1])

    with_fabric = IceModelBuilder(grid, column, surface=0.0, air=True)
    with_fabric.add_layers(layers)

    # Blend across the transition rather than stepping across it.  A hard step
    # on a rectangular grid turns a 35 degree plane into a staircase, which
    # scatters more strongly than the fabric contrast itself.  The blend has to
    # stay well inside a wavelength though: smoothing over ~4 m (more than a
    # wavelength at 60 MHz) makes the transition gradational and drops its
    # reflectivity by tens of dB, which would hide the very return this example
    # is about.  0.6 m is a few grid cells and about a fifth of a wavelength.
    edge = np.tanh((with_fabric.depth - boundary(grid.x)[:, None]) / 0.6)
    frac = 0.5 * (1.0 + edge)

    # Taper the contrast away near the side boundaries so the transition ends
    # smoothly instead of being truncated by the replicated padding, which
    # would diffract.
    span = 22.0
    ramp = np.clip((grid.x - xlim[0]) / span, 0, 1) * np.clip((xlim[1] - grid.x) / span, 0, 1)
    taper = np.sin(0.5 * np.pi * ramp) ** 2
    frac = frac * taper[:, None]

    with_fabric.set_fabric(
        np.ones(frac.shape, dtype=bool),
        dlambda=ABOVE["dlambda"] + (BELOW["dlambda"] - ABOVE["dlambda"]) * frac,
        lam_z=ABOVE["lam_z"] + (BELOW["lam_z"] - ABOVE["lam_z"]) * frac,
    )

    # The twin keeps the background fabric everywhere and the same layering:
    # identical in every respect except the fabric transition, so subtracting it
    # isolates the transition and leaves the layer reflections cancelled.
    without = IceModelBuilder(grid, column, surface=0.0, air=True)
    without.add_layers(layers)

    return (column, grid, with_fabric.finalize(npml=NPML),
            without.finalize(npml=NPML), boundary, layers)


def specular_geometry(x_antenna=0.0):
    """Where the return recorded at ``x_antenna`` actually comes from."""
    d = np.deg2rad(DIP_DEG)
    h = DEPTH_AT_X0 + np.tan(d) * x_antenna  # interface depth under the antenna
    slant = h * np.cos(d)  # perpendicular distance = apparent depth
    px = x_antenna - slant * np.sin(d)
    pz = slant * np.cos(d)
    return slant, px, pz


def main(quick=False, render_only=False, radargram=False, processes=None):
    use_talk_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "snapshots.npz"
    import matplotlib.pyplot as plt

    dx = 0.5 if quick else 0.32
    xlim = (-110.0, 110.0)
    zlim = (-8.0, 300.0) if not quick else (-8.0, 260.0)
    t_end = 1.95e-6

    column, grid, model, twin, boundary, layers = build_models(xlim, zlim, dx)
    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, t_end, dt)
    pulse = blackharrispulse(FC, t)
    src = np.array([[0.0, -1.0]])

    slant, px, pz = specular_geometry(0.0)
    eps_a = eigen_permittivity(**ABOVE)
    eps_b = eigen_permittivity(**BELOW)
    r_perp = abs((np.sqrt(eps_a[0]) - np.sqrt(eps_b[0]))
                 / (np.sqrt(eps_a[0]) + np.sqrt(eps_b[0])))
    r_par = abs((np.sqrt(eps_a[1]) - np.sqrt(eps_b[1]))
                / (np.sqrt(eps_a[1]) + np.sqrt(eps_b[1])))

    # Whichever eigenpolarisation sees the bigger jump is the one worth showing.
    bright, dim = (("89 deg", "179 deg") if r_perp >= r_par else ("179 deg", "89 deg"))

    print(f"transition dips {DIP_DEG:.0f} deg, {DEPTH_AT_X0:.0f} m below the antenna")
    print(f"  specular point at x = {px:.1f} m, z = {pz:.1f} m; "
          f"event images at {slant:.1f} m apparent depth")
    print(f"  apparent dip {np.degrees(np.arctan(np.sin(np.deg2rad(DIP_DEG)))):.1f} deg")
    print(f"  normal-incidence reflection: 89 deg pol {20 * np.log10(r_perp):.1f} dB, "
          f"179 deg pol {20 * np.log10(r_par):.1f} dB")

    if render_only and cache.exists():
        panels, sx, sz, stimes, extra = load_snapshots(cache)
        t_rec, tr_par, tr_perp = extra["t_rec"], extra["tr_par"], extra["tr_perp"]
        rec_par, rec_perp = extra["rec_par"], extra["rec_perp"]
    else:
        snap_every = max(1, len(t) // 240)
        scattered = {}
        total = {}
        traces = {}
        recorded = {}
        for axis, label in (("yy", "179 deg"), ("xx", "89 deg")):
            fields = []
            for m in (model, twin):
                # The out-of-plane solver samples eps_yy, so feed it whichever
                # component this eigenpolarisation sees.
                mm = m.with_properties(
                    eps={"xx": m.eps["xx"], "yy": m.eps[axis], "zz": m.eps["zz"]}
                )
                t0 = time.time()
                res = FDTD2D(mm, dt, npml=NPML, mode="TM").run(
                    src, pulse, src, outstep=1, snapshot_every=snap_every,
                    snapshot_stride=1, progress=max(1, len(t) // 4),
                )
                fields.append(res)
                print(f"  {label} ({'fabric' if m is model else 'twin'}) "
                      f"in {time.time() - t0:.1f} s")
            scattered[label] = fields[0].snapshots - fields[1].snapshots
            total[label] = fields[0].snapshots
            traces[label] = fields[0].gather[:, 0, 0] - fields[1].gather[:, 0, 0]
            recorded[label] = fields[0].gather[:, 0, 0]
            last = fields[0]

        # Left panel: the wave as it actually propagates.  Right panel: the
        # same run minus a transition-free twin, which leaves only the energy
        # the fabric contrast sent back.  Both are needed -- the reflection is
        # ~60 dB down, so it simply cannot be seen beside the incident wave,
        # which is exactly why the real feature is faint in the radargram.
        #
        # Below the interface the two models genuinely differ, so their
        # difference there is the transmitted wave picking up a different phase
        # -- large, but not the reflection.  Blank that out on the scattered
        # panel so it shows only upgoing energy a surface antenna could record.
        margin = 8.0
        panels, sx, sz = crop_snapshots(
            [(total[bright], f"E along {bright}")],
            last.snapshot_x, last.snapshot_z,
            xlim=(xlim[0] + margin, xlim[1] - margin),
            zlim=(zlim[0], zlim[1] - margin),
        )
        stimes = last.snapshot_times
        t_rec = last.t
        tr_par, tr_perp = traces[bright], traces[dim]
        rec_par, rec_perp = recorded[bright], recorded[dim]
        save_snapshots(cache, panels, sx, sz, stimes,
                       t_rec=t_rec, tr_par=tr_par, tr_perp=tr_perp,
                       rec_par=rec_par, rec_perp=rec_perp)

    # ---- movie ---------------------------------------------------------
    # The specular arrival, needed for the marker on the trace panel.
    t_pred_movie = 2.0 * column.traveltime(0.0, pz, axis=None) / np.cos(np.deg2rad(DIP_DEG))

    xline = np.linspace(xlim[0], xlim[1], 200)
    layer_lines = [(grid.x, l.depth_at(grid.x)) for l in layers]
    wavefield_movie(
        OUT / "dipping_fabric.mp4",
        panels,
        sx,
        sz,
        stimes,
        fps=25,
        gain_power=0.6,
        gamma=0.35,
        alpha_knee=0.03,
        contours=layer_lines + [
            (xline, boundary(xline)),
            (np.array([0.0, px]), np.array([0.0, pz])),  # the specular ray
        ],
        annotations=[
            (0.0, 18.0, "antenna"),
            (px + 26.0, pz + 16.0, "specular point"),
        ],
        trace={
            "t": t_rec,
            "series": [(rec_par, f"E along {bright}", "#b2182b")],
            "db": True,
            "markers": [(t_pred_movie, "fabric transition")],
            # Run the trace's time axis out to the two-way time of the deepest
            # depth on the wavefield panel, so the two panels read across at the
            # same physical depth instead of on unrelated scales.
            "tlim": float(column.two_way_time(float(sz[-1]))),
            "xlim": (-125.0, 5.0),
            "title": "what the receiver records",
        },
        title="A dipping fabric transition returns energy from off nadir",
        subtitle=(
            f"Conformable layering cut by a fabric transition dipping {DIP_DEG:.0f} deg, "
            f"{DEPTH_AT_X0:.0f} m below the antenna. Its return comes from {abs(px):.0f} m off "
            f"to the side and arrives as if the reflector were at {slant:.0f} m. A fabric "
            "contrast can never exceed about -51 dB, so it sits well below the layering."
        ),
    )
    print(f"  wrote {OUT / 'dipping_fabric.mp4'}")

    # ---- model / geometry figure ---------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), width_ratios=[1.15, 1])

    dl = model.eps["xx"] - model.eps["yy"]
    im = axes[0].imshow(dl.T, extent=model.extent(), aspect="equal", cmap="PuOr",
                        vmin=-np.max(np.abs(dl)), vmax=np.max(np.abs(dl)))
    axes[0].plot(xline, boundary(xline), color="k", lw=1.5)
    axes[0].plot([0], [0], marker="v", ms=13, color="#111")
    axes[0].plot([0, px], [0, pz], color="#d62728", lw=2, ls="--")
    axes[0].plot([px], [pz], marker="o", ms=9, color="#d62728")
    axes[0].annotate("specular point", (px, pz), textcoords="offset points",
                     xytext=(14, 26), color="#d62728", fontsize=11, ha="left")
    axes[0].annotate("antenna", (0, 0), textcoords="offset points", xytext=(14, -4),
                     fontsize=11, va="top")
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(zlim[1], zlim[0])
    axes[0].set_xlabel("distance (m)")
    axes[0].set_ylabel("depth (m)")
    axes[0].set_title("fabric contrast across the transition", loc="left")
    cb = fig.colorbar(im, ax=axes[0], pad=0.02, shrink=0.85)
    cb.set_label(r"$\epsilon_{xx} - \epsilon_{yy}$")

    # Recorded scattered trace: the arrival time is the test of the geometry.
    # Predict the arrival along the specular ray.  Using the solid-ice velocity
    # would be 15 percent too slow: most of this path is in firn, where the
    # wave travels much faster, so the traveltime has to be integrated through
    # the actual velocity profile.
    t_pred = 2.0 * column.traveltime(0.0, pz, axis=None) / np.cos(np.deg2rad(DIP_DEG))
    t_naive = 2.0 * slant / (C0 / np.sqrt(3.17))
    from scipy.signal import hilbert

    env_par = np.abs(hilbert(tr_par))
    env_perp = np.abs(hilbert(tr_perp))
    win = np.abs(t_rec - t_pred) < 0.09e-6
    t_meas = t_rec[win][int(np.argmax(env_par[win]))]
    ratio = 20 * np.log10(np.max(env_par[win]) / max(np.max(env_perp[win]), 1e-30))
    print(f"  scattered return: predicted {t_pred * 1e6:.3f} us (firn-corrected; "
          f"{t_naive * 1e6:.3f} us at the solid-ice velocity), "
          f"measured {t_meas * 1e6:.3f} us")
    predicted_contrast = 20 * np.log10(max(r_perp, r_par) / max(min(r_perp, r_par), 1e-12))
    print(f"  bright polarisation is {bright}; measured contrast {ratio:.1f} dB "
          f"(predicted {predicted_contrast:.1f} dB)")
    tx_peak = float(np.max(np.abs(hilbert(rec_par))))
    print(f"  fabric return sits {20 * np.log10(np.max(env_par[win]) / tx_peak):.0f} dB "
          f"below the transmit pulse; the layering peaks at "
          f"{20 * np.log10(np.max(np.abs(hilbert(rec_par))[(t_rec > 0.2e-6) & (t_rec < 1.2e-6)]) / tx_peak):.0f} dB")

    ax = axes[1]
    # Normalise on the arrival itself.  The very end of the record carries the
    # transmitted-field difference arriving back off the lower boundary, which
    # is not a reflection from the transition and would otherwise set the scale.
    scale = np.max(env_par[win])
    ax.plot(t_rec * 1e6, tr_par / scale, lw=1.0, color="#b2182b",
            label=f"E along {bright}")
    ax.plot(t_rec * 1e6, tr_perp / scale, lw=1.0, color="#2166ac",
            label=f"E along {dim}")
    ax.axvline(t_pred * 1e6, color="#111", ls="--", lw=1.2)
    ax.annotate(
        f"predicted arrival\nslant range {slant:.0f} m\n(interface is {DEPTH_AT_X0:.0f} m"
        f" below)",
        (t_pred * 1e6, 0.85), textcoords="offset points", xytext=(10, 0),
        fontsize=11, va="top",
    )
    ax.set_xlim(0.9, 1.9)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("two-way time (us)")
    ax.set_ylabel("scattered amplitude (normalised)")
    ax.legend(fontsize=11, loc="lower right")
    ax.set_title(
        f"recorded return at the antenna\n"
        f"polarisation contrast {ratio:.0f} dB", loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "geometry.png")
    plt.close(fig)
    print(f"  wrote {OUT / 'geometry.png'}")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    xs = np.array([0, 1, 2])
    width = 0.35
    ax.bar(xs - width / 2, eps_a, width, label="above transition", color="#2166ac")
    ax.bar(xs + width / 2, eps_b, width, label="below transition", color="#b2182b")
    ax.set_xticks(xs)
    ax.set_xticklabels([r"$\epsilon_{xx}$ (89 deg)", r"$\epsilon_{yy}$ (179 deg)",
                        r"$\epsilon_{zz}$"])
    lo = min(eps_a.min(), eps_b.min())
    hi = max(eps_a.max(), eps_b.max())
    ax.set_ylim(lo - 0.15 * (hi - lo), hi + 0.25 * (hi - lo))
    ax.set_ylabel("solid-ice eigenpermittivity")
    ax.legend(fontsize=11)
    ax.set_title(
        f"the jump is polarisation dependent\n"
        f"89 deg: {20 * np.log10(r_perp):.0f} dB    "
        f"179 deg: {20 * np.log10(r_par):.0f} dB",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "permittivity_jump.png")
    plt.close(fig)
    print(f"  wrote {OUT / 'permittivity_jump.png'}")

    # ---- optional common-offset section --------------------------------
    if radargram:
        step = 6.0
        xs_shot = np.arange(xlim[0] + 25, xlim[1] - 25 + 1e-9, step)
        shots = np.column_stack([xs_shot, np.full_like(xs_shot, -1.0)])
        secs = {}
        for axis, label in (("yy", "179 deg"), ("xx", "89 deg")):
            traces = []
            for m in (model, twin):
                mm = m.with_properties(
                    eps={"xx": m.eps["xx"], "yy": m.eps[axis], "zz": m.eps["zz"]}
                )
                print(f"  section {label} ({'fabric' if m is model else 'twin'}): "
                      f"{len(xs_shot)} shots")
                co = run_common_offset(mm, dt, shots, shots, pulse, npml=NPML,
                                       mode="TM", processes=processes)
                traces.append(co.common_offset)
                t_out = co.t
            secs[label] = traces[0] - traces[1]
        np.savez_compressed(OUT / "sections.npz", t=t_out, x=xs_shot,
                            **{k.replace(" ", "_"): v for k, v in secs.items()})

        # Two-way time to depth through the actual slowness profile.  A single
        # solid-ice velocity is about 25 percent too slow through the firn
        # column, which would image the event ~15 percent shallow while the
        # predicted-position overlay below is drawn in true metres.
        gain_power = 1.0
        depth_axis = column.depth_from_two_way_time(t_out)
        # One reference for both panels, so the polarisation contrast in the
        # section is the real amplitude difference and not a scaling artefact.
        ref = radargram_reference(secs.values(), t_out, gain_power=gain_power)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.6), sharey=True)
        for ax, (label, data) in zip(axes, secs.items()):
            plot_radargram(ax, data, xs_shot, t_out, db=True, dyn_range=40,
                           gain_power=gain_power, depth_axis=depth_axis, ref=ref,
                           title=f"scattered section, E along {label}")
            ax.plot(xline, boundary(xline), "k--", lw=1.2, label="true interface")
            ax.plot(xs_shot, (DEPTH_AT_X0 + np.tan(np.deg2rad(DIP_DEG)) * xs_shot)
                    * np.cos(np.deg2rad(DIP_DEG)), color="#d62728", lw=2,
                    label="predicted image position")
            # Do not open the axis below the depth the record actually reaches;
            # the extra would be blank paper the section says nothing about.
            ax.set_ylim(min(0.85 * zlim[1], float(depth_axis[-1])), 0)
            ax.legend(fontsize=10, loc="lower left")
        fig.suptitle(
            "The event images shallower and gentler than the interface really is  "
            f"(true dip {DIP_DEG:.0f} deg, apparent "
            f"{np.degrees(np.arctan(np.sin(np.deg2rad(DIP_DEG)))):.0f} deg)",
            x=0.008, ha="left", fontsize=15,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(OUT / "radargram.png")
        plt.close(fig)
        print(f"  wrote {OUT / 'radargram.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--radargram", action="store_true")
    p.add_argument("--processes", type=int, default=None)
    a = p.parse_args()
    main(quick=a.quick, render_only=a.render_only, radargram=a.radargram,
         processes=a.processes)
