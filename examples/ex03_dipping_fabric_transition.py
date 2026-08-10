"""Example 3 -- a fabric transition that is not at nadir but still returns energy.

The Ridge A radargram shows a faint, steeply dipping feature at 1000-1500 m
labelled "Deep Fabric Fold?".  This example models what produces it.

The scene is a strong change in crystal-orientation fabric across a plane
dipping at 35 degrees, blended over ``TRANSITION_WIDTH`` rather than stepped.  Two things follow from that geometry, and both are the
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
   different jumps.  Here the fabric changes so that the 89 deg polarisation
   (``eps_xx``) sees a strong contrast and the 179 deg one (``eps_yy``) sees
   almost none: the same interface is bright in one polarisation and nearly
   invisible in the other.  A density or acidity layer cannot do that.

The scene also carries ordinary conformable meteoric layering, so the fabric
transition appears as a discordant feature cutting across a conformable
background, which is what the radargram looks like.

The reflection is faint: about -54 dB at normal incidence for the bright
eigenpolarisation, and the 1.2 m blend across the transition costs a further
25.5 dB (see ``TRANSITION_WIDTH``), so roughly -80 dB before any spreading or
absorption.  Each polarisation is therefore run twice -- with
and without the fabric contrast -- and the difference isolates it.  That also
cancels any residual boundary artefact, which matters at this amplitude.  The
cancellation is exact only until the transmitted wave starts returning from
reflectors *below* the transition: the two models have genuinely different
velocities down there, so those layers arrive at different times and do not
subtract out.  Everything before ~1.6 us is clean, which covers the specular
arrival; the movie's trace panel shows the raw recorded trace anyway.

Outputs land in ``figures/ex03/``.  ``--render-only`` re-renders from cache, and
refuses a cache that was written from a different model.
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
    envelope_peak_time,
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
Z_ANT = -1.0  # antenna 1 m above the snow surface

DIP_DEG = 35.0  # dip of the fabric transition
DEPTH_AT_X0 = 175.0  # depth of the transition below the surface at the antenna

# Width of the blend across the transition, and what it costs.  A fabric
# interface is a gradational thing in real ice -- fabric evolves with strain,
# not in a step -- and the reflectivity is acutely sensitive to how gradational:
# filtering the pulse through the interface gives -6.5 dB at 0.3 m, -14.5 dB at
# 0.6 m, -25.5 dB at 1.2 m and -40 dB at 2.5 m, against a sharp step.  Roughly
# 10-15 dB per doubling, so this constant is not cosmetic.  1.2 m is a little
# under half a wavelength at 60 MHz: clearly gradational, still measurable.
TRANSITION_WIDTH = 1.2

# Layers whose two-way time coincides with the transition's would land their own
# wavelet on top of it, and nothing downstream can separate two arrivals at the
# same time.  This clears a band around the equivalent nadir depth; it has to
# exceed the wavelet length plus the layer undulation amplitude (~4 m here).
LAYER_GAP_HALFWIDTH = 15.0

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

# The background column as its constructor arguments, kept as a mapping rather
# than written out at the call site so the cache stamp is taken from the same
# values the column is built from and a parameter added later cannot escape it.
# The ``dlambda`` callable closes over ABOVE, which is stamped in its own right.
COLUMN = dict(
    thickness=1850.0,
    depth_bco=90.0,
    sigma_ice=1.2e-5,
    lam_z_sfc=ABOVE["lam_z"],
    lam_z_bed=ABOVE["lam_z"],
    dlambda=lambda d: np.full_like(np.asarray(d, float), ABOVE["dlambda"]),
)


def eigen_permittivity(lam_z, dlambda):
    """Solid-ice eigenpermittivities for a given (lam_z, dlam) pair."""
    from radarwave.constants import DEPS_ICE, EPS_ICE_MEAN

    h = 1.0 - lam_z
    lam = np.array([(h + dlambda) / 2, (h - dlambda) / 2, lam_z])
    return EPS_ICE_MEAN + DEPS_ICE * (lam - 1.0 / 3.0)


def layer_exclusion(column, zlim, x_antenna=0.0):
    """The nadir depth band whose two-way time collides with the event.

    Which nadir depth would a layer have to sit at to arrive when the
    transition does?  Not the transition's own depth: the specular ray leaves
    at the dip angle, so it images at its perpendicular range.  Invert the
    column's own two-way time rather than dividing by a nominal velocity, so
    the firn is accounted for.  Shared with ex04, whose banded package images
    from the same specular point.

    Returns ``(d_event, (lo, hi))``.
    """
    _, _, pz = specular_geometry(x_antenna)
    t_event = column.two_way_time(pz, z0=Z_ANT) / np.cos(np.deg2rad(DIP_DEG))
    probe = np.linspace(0.0, zlim[1], 4001)
    # Rounded to the millimetre: the band is 30 m wide, so nothing in the model
    # can tell the difference, and it keeps a numpy release whose interpolation
    # differs in the last digits from rejecting a cache as stale.
    d_event = round(float(np.interp(t_event, column.two_way_time(probe, z0=Z_ANT), probe)), 3)
    return d_event, (d_event - LAYER_GAP_HALFWIDTH, d_event + LAYER_GAP_HALFWIDTH)


def build_models(xlim, zlim, dx):
    """Return the model with the fabric transition and its contrast-free twin."""
    column = IceColumn(**COLUMN)
    grid = PropertyGrid.uniform(xlim, zlim, dx)

    def boundary(x):
        return dipping_depth(x, DEPTH_AT_X0, DIP_DEG)

    d_event, exclude = layer_exclusion(column, zlim)
    layers = conformal_layering(zlim[1], exclude=exclude)

    with_fabric = IceModelBuilder(grid, column, surface=0.0, air=True)
    with_fabric.add_layers(layers)

    # Blend across the transition rather than stepping across it.  A hard step
    # on a rectangular grid turns a 35 degree plane into a staircase, which
    # scatters more strongly than the fabric contrast itself.  The blend is not
    # free: see TRANSITION_WIDTH for what each width costs.  It is a physical
    # parameter of the model, not a numerical detail -- how sharp a fabric
    # transition has to be before a radar can see it is the question this
    # example is really asking.
    edge = np.tanh((with_fabric.depth - boundary(grid.x)[:, None]) / TRANSITION_WIDTH)
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
            without.finalize(npml=NPML), boundary, layers, exclude)


def specular_geometry(x_antenna=0.0):
    """Where the return recorded at ``x_antenna`` actually comes from.

    Returns ``(slant_range, px, pz)``.  ``slant_range`` is the perpendicular
    distance from the *antenna* to the plane -- the whole ray, its short air leg
    included, and the range the event images at.  ``(px, pz)`` is the specular
    point in model coordinates, so ``pz`` is a depth below the surface and the
    ice portion of the ray is ``pz / cos(delta)`` of that range.

    Everything is measured from the antenna rather than from the surface,
    because the ray :func:`_common.echo_time` times starts at ``Z_ANT`` and runs
    at the dip angle the whole way.  Dropping the perpendicular from ``z = 0``
    instead leaves the air leg pointing somewhere else and quietly lengthens the
    path.
    """
    d = np.deg2rad(DIP_DEG)
    h = DEPTH_AT_X0 - Z_ANT + np.tan(d) * x_antenna  # interface depth below the antenna
    slant_range = h * np.cos(d)
    px = x_antenna - slant_range * np.sin(d)
    pz = Z_ANT + slant_range * np.cos(d)
    return slant_range, px, pz


def main(quick=False, render_only=False, radargram=False, processes=None, bare=False,
         out=None):
    # An output override is what lets many runs share a machine: without it,
    # every invocation writes the same figures/exNN paths and a parameter sweep
    # destroys its own results.  The cache guard catches the mixing after the
    # fact; this prevents it.
    if out is not None:
        global OUT
        OUT = Path(out)
    use_talk_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "snapshots.npz"
    import matplotlib.pyplot as plt

    dx = 0.5 if quick else 0.32
    xlim = (-110.0, 110.0)
    zlim = (-8.0, 300.0) if not quick else (-8.0, 260.0)
    t_end = 1.95e-6

    column, grid, model, twin, boundary, layers, exclude = build_models(xlim, zlim, dx)
    # Everything the cached wavefield and traces depend on.  A cache written
    # from a different model must not be re-rendered: its measured numbers would
    # be printed beside predictions computed from the constants above.  The
    # column goes in as the mapping it is built from, so a parameter of it that
    # never reaches the derived band -- sigma_ice, which sets the recorded
    # amplitudes but not any traveltime -- is still caught, and is named as
    # itself rather than as the band having moved.
    stamp = {
        "dx": dx,
        "xlim": xlim,
        "zlim": zlim,
        "t_end": t_end,
        "fc": FC,
        "npml": NPML,
        "z_ant": Z_ANT,
        "dip_deg": DIP_DEG,
        "depth_at_x0": DEPTH_AT_X0,
        "transition_width": TRANSITION_WIDTH,
        "layer_gap_halfwidth": LAYER_GAP_HALFWIDTH,
        "layer_exclusion_band": exclude,
        "fabric_above": ABOVE,
        "fabric_below": BELOW,
        "column": stampable(COLUMN),
    }
    dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
    t = np.arange(0.0, t_end, dt)
    pulse = blackharrispulse(FC, t)
    t_wave = envelope_peak_time(pulse, t)
    src = np.array([[0.0, Z_ANT]])

    slant_range, px, pz = specular_geometry(0.0)
    eps_a = eigen_permittivity(**ABOVE)
    eps_b = eigen_permittivity(**BELOW)
    r_perp = abs((np.sqrt(eps_a[0]) - np.sqrt(eps_b[0]))
                 / (np.sqrt(eps_a[0]) + np.sqrt(eps_b[0])))
    r_par = abs((np.sqrt(eps_a[1]) - np.sqrt(eps_b[1]))
                / (np.sqrt(eps_a[1]) + np.sqrt(eps_b[1])))

    # Whichever eigenpolarisation sees the bigger jump is the one worth showing.
    bright, dim = (("89 deg", "179 deg") if r_perp >= r_par else ("179 deg", "89 deg"))

    print(f"transition dips {DIP_DEG:.0f} deg, {DEPTH_AT_X0:.0f} m below the surface")
    print(f"  specular point at x = {px:.1f} m, z = {pz:.1f} m; "
          f"event images at {slant_range:.1f} m range from the antenna")
    print(f"  apparent dip {np.degrees(np.arctan(np.sin(np.deg2rad(DIP_DEG)))):.1f} deg")
    print(f"  normal-incidence reflection: 89 deg pol {20 * np.log10(r_perp):.1f} dB, "
          f"179 deg pol {20 * np.log10(r_par):.1f} dB")

    if render_only and cache.exists():
        panels, sx, sz, stimes, extra = load_snapshots(cache, stamp=stamp)
        t_rec, tr_par, tr_perp = extra["t_rec"], extra["tr_par"], extra["tr_perp"]
        rec_par, rec_perp = extra["rec_par"], extra["rec_perp"]
    else:
        snap_every = max(1, len(t) // 240)
        traces = {}
        recorded = {}
        shown = None
        for axis, label in (("yy", "179 deg"), ("xx", "89 deg")):
            fields = []
            for m in (model, twin):
                # The out-of-plane solver samples eps_yy, so feed it whichever
                # component this eigenpolarisation sees.
                mm = m.with_properties(
                    eps={"xx": m.eps["xx"], "yy": m.eps[axis], "zz": m.eps["zz"]}
                )
                # Only the movie consumes a wavefield, and it shows one panel:
                # the bright polarisation propagating through the real model.
                # A stack here is ~700 MB, so the other three runs record their
                # trace and nothing else.
                snap = (
                    dict(snapshot_every=snap_every, snapshot_stride=1)
                    if m is model and label == bright else {}
                )
                t0 = time.time()
                res = FDTD2D(mm, dt, npml=NPML, mode="TM").run(
                    src, pulse, src, outstep=1, progress=max(1, len(t) // 4), **snap,
                )
                fields.append(res)
                print(f"  {label} ({'fabric' if m is model else 'twin'}) "
                      f"in {time.time() - t0:.1f} s")
            traces[label] = fields[0].gather[:, 0, 0] - fields[1].gather[:, 0, 0]
            recorded[label] = fields[0].gather[:, 0, 0]
            if label == bright:
                shown = fields[0]

        # The movie shows one wavefield panel -- the wave as it actually
        # propagates, in the bright polarisation -- beside the trace the
        # antenna records.  The transition's own return is ~80 dB down and
        # cannot be seen beside the incident wave at all, which is exactly why
        # the real feature is faint in the radargram; it is isolated instead by
        # subtracting the transition-free twin, and that difference is what the
        # geometry figure below measures.
        #
        # Crop the wavefield to the depth the record can actually reach, so the
        # wavefield and trace panels end at the same physical depth.  Showing
        # deeper than that would need the trace axis to run past t_end, leaving
        # a large blank strip beside depths no echo can return from.
        margin = 8.0
        depth_reached = float(
            column.depth_from_two_way_time(shown.t[-1] - t_wave, z0=Z_ANT)
        )
        panels, sx, sz = crop_snapshots(
            [(shown.snapshots, f"E along {bright}")],
            shown.snapshot_x, shown.snapshot_z,
            xlim=(xlim[0] + margin, xlim[1] - margin),
            zlim=(zlim[0], min(zlim[1] - margin, depth_reached)),
        )
        stimes = shown.snapshot_times
        t_rec = shown.t
        tr_par, tr_perp = traces[bright], traces[dim]
        rec_par, rec_perp = recorded[bright], recorded[dim]
        save_snapshots(cache, panels, sx, sz, stimes, stamp=stamp,
                       t_rec=t_rec, tr_par=tr_par, tr_perp=tr_perp,
                       rec_par=rec_par, rec_perp=rec_perp)

    # ---- movie ---------------------------------------------------------
    # When the specular return peaks in the record: the two-way path down to
    # the specular point, timed from the antenna and along the inclined ray
    # rather than straight down, plus the wavelet's own offset.  Using the
    # solid-ice velocity instead would be 15 percent too slow -- most of this
    # path is in firn, where the wave travels much faster -- so the traveltime
    # is integrated through the actual velocity profile.  Marked on the movie's
    # trace panel and measured against the recording in the figure below.
    t_pred = float(echo_time(column, pz, Z_ANT, t_wave, dip_deg=DIP_DEG))

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
            "markers": [(t_pred, "fabric transition")],
            # The wavefield panel is cropped to the depth the record reaches,
            # so the two panels now end at the same physical depth and the
            # trace axis is simply the record.
            "tlim": float(t_rec[-1]),
            "xlim": (-125.0, 5.0),
            "title": "what the receiver records",
        },
        title="A dipping fabric transition returns energy from off nadir",
        subtitle=(
            f"Conformable layering cut by a fabric transition dipping {DIP_DEG:.0f} deg, "
            f"{DEPTH_AT_X0:.0f} m below the surface. Its return comes from {abs(px):.0f} m off "
            f"to the side and arrives as if the reflector were {slant_range:.0f} m from the "
            f"antenna. A fabric "
            "contrast can never exceed about -51 dB, so it sits well below the layering."
        ),
        bare=bare,
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
    # The same ray at a single solid-ice velocity, for contrast.  Only the ice
    # portion of it -- ``pz / cos(delta)`` of the slant range -- is what a
    # velocity assumption applies to; the air leg and the wavelet offset come
    # from the same place as they do for the firn-corrected prediction, so each
    # is counted exactly once and both times are comparable with the recording.
    ice_range = pz / np.cos(np.deg2rad(DIP_DEG))
    t_naive = 2.0 * ice_range / (C0 / np.sqrt(3.17)) + float(
        echo_time(column, 0.0, Z_ANT, t_wave, dip_deg=DIP_DEG)
    )
    from radarwave.polarimetry import analytic

    env_par = np.abs(analytic(tr_par))
    env_perp = np.abs(analytic(tr_perp))
    win = np.abs(t_rec - t_pred) < 0.09e-6
    t_meas = t_rec[win][int(np.argmax(env_par[win]))]
    ratio = 20 * np.log10(np.max(env_par[win]) / max(np.max(env_perp[win]), 1e-30))
    print(f"  scattered return: predicted {t_pred * 1e6:.3f} us (firn-corrected; "
          f"{t_naive * 1e6:.3f} us at the solid-ice velocity), "
          f"measured {t_meas * 1e6:.3f} us")
    predicted_contrast = 20 * np.log10(max(r_perp, r_par) / max(min(r_perp, r_par), 1e-12))
    print(f"  bright polarisation is {bright}; measured contrast {ratio:.1f} dB "
          f"(predicted {predicted_contrast:.1f} dB)")
    tx_peak = float(np.max(np.abs(analytic(rec_par))))
    print(f"  fabric return sits {20 * np.log10(np.max(env_par[win]) / tx_peak):.0f} dB "
          f"below the transmit pulse; the layering peaks at "
          f"{20 * np.log10(np.max(np.abs(analytic(rec_par))[(t_rec > 0.2e-6) & (t_rec < 1.2e-6)]) / tx_peak):.0f} dB")

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
        f"predicted arrival\nslant range {slant_range:.0f} m from the antenna\n"
        f"(interface is {DEPTH_AT_X0:.0f} m below the surface)",
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
    save_figure(fig, OUT / "geometry.png", bare)
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
    save_figure(fig, OUT / "permittivity_jump.png", bare)
    plt.close(fig)
    print(f"  wrote {OUT / 'permittivity_jump.png'}")

    # ---- optional common-offset section --------------------------------
    if radargram:
        step = 6.0
        xs_shot = np.arange(xlim[0] + 25, xlim[1] - 25 + 1e-9, step)
        shots = np.column_stack([xs_shot, np.full_like(xs_shot, Z_ANT)])
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

        # Two-way time to depth through the actual slowness profile, from the
        # antenna and less the wavelet offset, so a return drawn at its own
        # depth lands there.  A single solid-ice velocity is about 25 percent
        # too slow through the firn column, which would image the event ~15
        # percent shallow while the predicted-position overlay below is drawn
        # in true metres.
        gain_power = 1.0
        depth_axis = column.depth_from_two_way_time(t_out - t_wave, z0=Z_ANT)
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
        save_figure(fig, OUT / "radargram.png", bare)
        plt.close(fig)
        print(f"  wrote {OUT / 'radargram.png'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--render-only", action="store_true")
    p.add_argument("--radargram", action="store_true")
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--out", default=None, metavar="DIR",
                   help="write all outputs and caches under DIR "
                        "instead of figures/exNN (for sweeps and "
                        "cluster array jobs)")
    p.add_argument("--bare", action="store_true",
                   help="strip titles, notes, annotations and legends for slides")
    a = p.parse_args()
    try:
        main(quick=a.quick, render_only=a.render_only, radargram=a.radargram,
             bare=a.bare, processes=a.processes, out=a.out)
    except StaleCache as exc:
        raise SystemExit(str(exc))
