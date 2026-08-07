"""Talk-quality figures and wavefield movies.

Snapshots are written straight to MP4 (and optionally GIF) through
``imageio-ffmpeg``; no intermediate PNG sequence is produced.  Frames are
rendered by mutating a single Matplotlib figure and grabbing its canvas
buffer, which is far faster than re-drawing from scratch.
"""

from pathlib import Path
import numpy as np

from .polarimetry import analytic

__all__ = [
    "use_talk_style",
    "wavefield_movie",
    "plot_radargram",
    "radargram_reference",
    "plot_model",
    "spreading_gain",
    "crop_snapshots",
    "to_db",
]


def crop_snapshots(stacks, x, z, xlim=None, zlim=None):
    """Trim snapshot stacks to a display window.

    Snapshots cover the padded grid, so they include the absorbing layer and
    the replicated properties just outside the physical model.  Layers that are
    extruded horizontally into the padding leave a kink that scatters, and with
    an amplitude-compressed colour scale that shows up clearly.  Cropping a
    margin off each side keeps the picture to the part of the model that was
    actually specified.

    ``stacks`` is a single array or a list of ``(array, label)`` pairs; the
    return type matches.
    """
    x = np.asarray(x)
    z = np.asarray(z)
    ix = slice(None) if xlim is None else slice(
        int(np.searchsorted(x, xlim[0])), int(np.searchsorted(x, xlim[1], side="right"))
    )
    iz = slice(None) if zlim is None else slice(
        int(np.searchsorted(z, zlim[0])), int(np.searchsorted(z, zlim[1], side="right"))
    )
    if isinstance(stacks, (list, tuple)) and not isinstance(stacks, np.ndarray):
        out = [(np.asarray(s)[:, ix, iz], lbl) for s, lbl in stacks]
    else:
        out = np.asarray(stacks)[:, ix, iz]
    return out, x[ix], z[iz]

#: Colour used for annotation overlays on the wavefield movies.
ANNOT = "#111111"


def use_talk_style():
    """Matplotlib defaults tuned for projected slides."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 128,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "image.interpolation": "nearest",
        }
    )


def spreading_gain(t, power=0.5, t0=None):
    """Cylindrical-spreading compensation ``(t / t0) ** power``.

    A two-dimensional line source spreads as ``1/sqrt(r)``, so without this the
    later part of a movie fades to nothing and the deep reflections a talk is
    about become invisible.
    """
    t = np.asarray(t, dtype=float)
    t0 = np.max(t) if t0 is None else t0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (np.maximum(t, 1e-12) / t0) ** power
    return np.where(np.isfinite(g), g, 0.0)


def to_db(a, floor=-120.0, ref=None):
    """Amplitude to dB relative to ``ref`` (default: the array maximum)."""
    a = np.abs(np.asarray(a, dtype=float))
    ref = np.max(a) if ref is None else ref
    if ref <= 0:
        return np.full_like(a, floor)
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(a / ref)
    return np.maximum(db, floor)


def _rgb(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def _auto_figsize(x, z, width=11.0, max_height=10.0, min_width=4.6):
    """Figure size that renders an equal-aspect domain without huge margins.

    A deep, narrow model would otherwise produce a figure too skinny to fit its
    own title, so the width has a floor and the height is allowed to grow.
    """
    xs = float(x[-1] - x[0]) or 1.0
    zs = float(z[-1] - z[0]) or 1.0
    height = width * (zs / xs) * 0.82 + 1.6
    if height > max_height:
        width = max(min_width, width * max_height / height)
        height = max_height
    return (round(width, 2), round(height, 2))


def wavefield_movie(
    path,
    snapshots,
    x,
    z,
    times,
    *,
    fps=25,
    clip=99.9,
    gain_power=0.5,
    gamma=0.35,
    alpha_knee=0.03,
    share_scale=False,
    cmap="RdBu_r",
    background=None,
    background_cmap="Greys",
    background_alpha=0.22,
    contours=None,
    title="",
    subtitle=None,
    annotations=None,
    xlabel="distance (m)",
    ylabel="depth (m)",
    figsize=None,
    follow=None,
    guide=None,
    trace=None,
    dpi=112,
    crf=24,
    also_gif=False,
    progress=True,
):
    """Render a wavefield time series to an MP4 movie.

    Parameters
    ----------
    path : str or Path
        Output ``.mp4`` path.
    snapshots : (n_t, nx, nz) array
        Field snapshots, as returned in :attr:`radarwave.Result.snapshots`.
    x, z : arrays
        Coordinates matching the snapshot axes (m).
    times : (n_t,) array
        Snapshot times (s).
    clip : float
        Percentile of the gained field used to set the symmetric colour limits.
    gain_power : float
        Exponent of the spreading compensation applied per frame; 0 disables it.
    gamma : float
        Amplitude compression exponent.  Internal reflections run 40-70 dB below
        the direct wave, so a linear colour scale shows the direct arrival and
        nothing else; ``gamma = 0.4`` brings a 1-in-200 reflection up to about
        17 percent of full scale while keeping the sign of the field.
    share_scale : bool
        Give every panel the same colour limit.  Essential when the point of
        the comparison is that one panel is weaker than another -- with
        per-panel limits a 20 dB weaker field is rescaled until it looks
        identical.
    alpha_knee : float
        Physical amplitude (as a fraction of the colour limit) above which the
        wavefield is fully opaque.  Below it the overlay fades out, which hides
        the slowly decaying wake of the line source without touching the colour
        of anything that matters.
    background : (nx, nz) array, optional
        A static field (typically permittivity) drawn underneath the wave.
    contours : sequence of (x_array, z_array) pairs, optional
        Polylines drawn over every frame, e.g. layer or interface geometry.
    annotations : sequence of (x, z, text) tuples, optional
    follow : dict, optional
        ``{"speed": m/s, "window": m, "start": m}``.  Scrolls the depth axis
        with the wavefront instead of showing the whole model.  A phase shift
        between two panels is often a metre or two, which is a couple of pixels
        when 400 m of ice is on screen and plainly visible when 40 m is; this
        is how to see one.
    guide : dict, optional
        ``{"panel": index, "label": str, "halfwidth": m}``.  Tracks the
        wavefront depth in one panel and draws it as a horizontal line across
        every panel, so a lag between two panels is read against a fixed
        reference instead of by eye.
    trace : dict, optional
        Draw the trace the surface receiver is recording, filling in as the
        simulation runs.  Keys: ``t`` (s), ``series`` as a list of
        ``(amplitude, label, colour)``, and optionally ``gain_power`` (time
        gain applied before display, default 2), ``markers`` as
        ``[(t_seconds, label), ...]``, ``db`` (default True: show
        ``20 log10`` of the analytic envelope, i.e. returned power, referenced
        to the transmit pulse -- this is how a processed sounding is displayed
        and it keeps a 60 dB range readable), ``norm_after`` (linear mode only),
        ``xlim``, ``title`` and ``note``.
    crf : int
        x264 constant rate factor; lower is better quality and a bigger file.
    also_gif : bool
        Additionally write a GIF beside the MP4.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(times, dtype=float)
    gains = (
        np.ones(len(times))
        if not gain_power
        else spreading_gain(times, power=gain_power)
    )

    # ``snapshots`` is either one stack or a list of (stack, panel title) pairs.
    if isinstance(snapshots, (list, tuple)) and not isinstance(snapshots, np.ndarray):
        panels = [(np.asarray(s, dtype=np.float32), lbl) for s, lbl in snapshots]
    else:
        panels = [(np.asarray(snapshots, dtype=np.float32), None)]

    colours = plt.get_cmap(cmap)
    sample = np.linspace(0, len(times) - 1, min(len(times), 40)).astype(int)

    vmaxes = []
    for stack, _ in panels:
        scaled = np.concatenate([np.abs(stack[k].ravel()) * gains[k] for k in sample])
        nz = scaled[scaled > 0]
        vmaxes.append(float(np.percentile(nz, clip)) if nz.size else 1.0)
    vmaxes = [v if v > 0 else 1.0 for v in vmaxes]
    if share_scale:
        vmaxes = [max(vmaxes)] * len(vmaxes)

    def rgba_frame(stack, vmax, k):
        u = np.clip(stack[k] * gains[k] / vmax, -1.0, 1.0)
        # Colour uses the compressed amplitude so weak reflections are still
        # boldly coloured; opacity gates on the *physical* amplitude so the
        # faint wake behind a 2-D line source stays out of the way.  Keeping
        # the two separate is what makes both the direct wave and a reflection
        # 40 dB down readable in the same frame.
        s = np.sign(u) * np.abs(u) ** gamma
        img = colours((s.T + 1.0) / 2.0)
        img[..., 3] = np.clip(np.abs(u.T) / alpha_knee, 0.0, 1.0)
        return img

    extent = (x[0], x[-1], z[-1], z[0])
    if figsize is None and follow:
        figsize = (5.2 * len(panels) + 1.0, 7.0)
    if figsize is None:
        w, h = _auto_figsize(x, z)
        total = w * (1.0 if len(panels) == 1 else 0.62 * len(panels) + 0.4)
        figsize = (max(total, 9.0 if len(panels) > 1 else 6.5), h)
    if trace:
        figsize = (figsize[0] + 4.2, figsize[1])

    n_field = len(panels)
    widths = [1.0] * n_field + ([0.72] if trace else [])
    fig, grid_axes = plt.subplots(
        1, n_field + (1 if trace else 0), figsize=figsize, dpi=dpi,
        squeeze=False, width_ratios=widths,
    )
    axes = list(grid_axes[0][:n_field])
    trace_ax = grid_axes[0][n_field] if trace else None
    for a in axes[1:]:
        a.sharey(axes[0])
    fig.patch.set_facecolor("white")

    images = []
    for ax, (stack, label), vmax in zip(axes, panels, vmaxes):
        if background is not None:
            bg = np.asarray(background, dtype=float)
            # Stretch to the 2-98 percentile so the tint stays light and the
            # wavefield overlay keeps its colour.
            lo, hi = np.percentile(bg, [2, 98])
            ax.imshow(
                bg.T, extent=extent, aspect="equal", cmap=background_cmap,
                alpha=background_alpha, zorder=0,
                vmin=lo, vmax=hi + 0.35 * (hi - lo or 1.0),
            )
        for line in contours or []:
            ax.plot(line[0], line[1], color=ANNOT, lw=0.7, alpha=0.35, zorder=1)
        images.append(ax.imshow(rgba_frame(stack, vmax, 0), extent=extent,
                                aspect="auto" if follow else "equal", zorder=2))
        for ax_, az_, text in annotations or []:
            ax.annotate(
                text, (ax_, az_), color=ANNOT, fontsize=11, zorder=4,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
            )
        ax.set_xlabel(xlabel)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(z[-1], z[0])
        if label:
            ax.set_title(label, loc="left", fontsize=12)
    axes[0].set_ylabel(ylabel)

    guide_state = {"z": None, "stalled": 0}
    guide_lines = []
    if guide:
        for ax in axes:
            guide_lines.append(ax.axhline(np.nan, color="#111", ls="--", lw=1.2, zorder=6))
        guide_text = axes[0].annotate(
            guide.get("label", "wavefront"), (0.02, 0.0),
            xycoords=("axes fraction", "data"), textcoords="offset points",
            xytext=(0, -7), fontsize=10.5, va="top", zorder=7,
            annotation_clip=True,
        )

    trace_lines = []
    if trace:
        t_tr = np.asarray(trace["t"], dtype=float)
        gp = trace.get("gain_power", 0.0)
        gain = np.maximum(t_tr, 1e-12) ** gp if gp else np.ones_like(t_tr)

        if trace.get("db", True):
            # Returned power in dB, the way a processed sounding is displayed.
            # The envelope of the analytic signal is used so the trace reads as
            # a power profile rather than a dense oscillation, and everything is
            # referenced to the transmit pulse, which is the natural 0 dB.

            # One reference for every series, not one per series: the panel
            # exists to compare them, and normalising each to its own peak is
            # exactly the rescaling ``share_scale`` warns about.
            envs = [np.abs(analytic(np.asarray(a, dtype=float))) * gain
                    for a, _, _ in trace["series"]]
            ref = trace.get("ref") or max(float(np.max(e)) for e in envs) or 1.0
            prepared = []
            for env, (_, lbl, col) in zip(envs, trace["series"]):
                with np.errstate(divide="ignore"):
                    prepared.append((20.0 * np.log10(np.maximum(env / ref, 1e-12)), lbl, col))
            floor = trace.get("xlim", (-100.0, 5.0))[0]
            series = [(np.maximum(v, floor - 5.0), lbl, col) for v, lbl, col in prepared]
        else:
            series = [(np.asarray(a, dtype=float) * gain, lbl, col)
                      for a, lbl, col in trace["series"]]
            # A monostatic antenna records its own transmit pulse ~60 dB above
            # every reflection; scaling to that leaves the trace a flat line.
            after = trace.get("norm_after", 0.0)
            keep_n = t_tr > after
            norm = max(float(np.max(np.abs(a[keep_n]))) for a, _, _ in series) or 1.0
            series = [(a / norm, lbl, col) for a, lbl, col in series]

        for amp, lbl, col in series:
            (ln,) = trace_ax.plot([], [], lw=1.2, color=col, label=lbl)
            trace_lines.append((ln, amp, t_tr))
        for t_guide in trace.get("guides", []):
            trace_ax.axhline(t_guide * 1e6, color="#999", ls=":", lw=0.8, zorder=1)
        for t_mark, lbl in trace.get("markers", []):
            trace_ax.axhline(t_mark * 1e6, color="#111", ls="--", lw=1.1)
            trace_ax.annotate(lbl, (0.03, t_mark * 1e6), xycoords=("axes fraction", "data"),
                              textcoords="offset points", xytext=(0, 7),
                              fontsize=10, va="bottom")
        trace_ax.set_xlim(*trace.get("xlim", (-100.0, 5.0) if trace.get("db", True)
                                     else (-1.15, 1.15)))
        trace_ax.set_ylim(trace.get("tlim", t_tr[-1]) * 1e6, 0.0)
        trace_ax.set_xlabel(trace.get(
            "xlabel",
            "returned power (dB re. transmit pulse)" if trace.get("db", True)
            else "amplitude",
        ))
        trace_ax.set_ylabel("two-way time (us)")
        trace_ax.set_title(trace.get("title", "recorded at the surface"),
                           loc="left", fontsize=12)
        trace_ax.grid(alpha=0.25)
        if len(series) > 1:
            trace_ax.legend(fontsize=10, loc="lower center")

    fig.suptitle(title, x=0.012, ha="left", fontsize=15, y=0.995)
    if subtitle:
        # Wrap by hand: a long single line runs off the canvas and is clipped.
        import textwrap

        wrapped = "\n".join(textwrap.wrap(subtitle, width=max(60, int(figsize[0] * 10.5))))
        fig.text(0.012, 0.955, wrapped, ha="left", va="top", fontsize=10.5, color="#222")
    clock = axes[-1].text(
        0.985, 0.03, "", transform=axes[-1].transAxes, ha="right", va="bottom",
        fontsize=12, family="monospace", color=ANNOT, zorder=5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93 if subtitle else 0.96))

    # Constant-rate-factor encoding rather than a fixed quality level: the
    # amplitude-compressed wavefield is full of fine detail that a qscale
    # encode turns into a very large file (tens of MB), which is awkward to
    # drop into a slide deck.  crf 24 keeps these under about 10 MB.
    writer = imageio.get_writer(
        path, fps=fps, codec="libx264", quality=None, macro_block_size=16,
        ffmpeg_params=["-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p"],
    )
    gif_frames = []
    try:
        for k, t in enumerate(times):
            for im, (stack, _), vmax in zip(images, panels, vmaxes):
                im.set_data(rgba_frame(stack, vmax, k))
            z_front = None
            if guide:
                src_stack = panels[guide.get("panel", 0)][0][k]
                hw = guide.get("halfwidth", 6.0)
                cols = np.abs(x - 0.5 * (x[0] + x[-1])) <= hw
                if not cols.any():
                    cols = np.ones_like(x, dtype=bool)
                prof = np.abs(src_stack[cols]).mean(axis=0)
                z_front = float(z[int(np.argmax(prof))])
                # Never let it walk back up: once the wave has left the bottom
                # of the model the strongest thing on the axis is a reflection
                # heading the other way, and the window would lurch upwards.
                if guide_state["z"] is not None:
                    if z_front <= guide_state["z"]:
                        guide_state["stalled"] += 1
                    else:
                        guide_state["stalled"] = 0
                    z_front = max(z_front, guide_state["z"])
                guide_state["z"] = z_front
                for ln in guide_lines:
                    ln.set_ydata([z_front, z_front])
                guide_text.xy = (0.02, z_front)

            if follow:
                # Centre on the measured wavefront when there is one: a fixed
                # tracking speed drifts off it, because the wave moves faster
                # through firn than through solid ice.  Once the wavefront stops
                # advancing it has left the model, and there is nothing left to
                # follow -- pull back to the whole domain so the returns coming
                # back up are visible instead of an empty window.
                if guide and guide_state["stalled"] > 3:
                    for ax in axes:
                        ax.set_ylim(z[-1], z[0])
                else:
                    centre = (z_front if z_front is not None
                              else follow.get("start", 0.0) + follow["speed"] * t)
                    half = 0.5 * follow["window"]
                    lo = min(max(centre - half, z[0]), max(z[0], z[-1] - follow["window"]))
                    for ax in axes:
                        ax.set_ylim(lo + follow["window"], lo)

            for ln, amp, t_tr in trace_lines:
                n = int(np.searchsorted(t_tr, t, side="right"))
                ln.set_data(amp[:n], t_tr[:n] * 1e6)
            clock.set_text(f"{t * 1e6:7.2f} us")
            frame = _rgb(fig)
            writer.append_data(frame)
            if also_gif and k % 2 == 0:
                gif_frames.append(frame[::2, ::2])
            if progress and (k % 25 == 0 or k == len(times) - 1):
                print(f"    frame {k + 1}/{len(times)}", end="\r", flush=True)
    finally:
        writer.close()
        plt.close(fig)
    if progress:
        print()

    if also_gif and gif_frames:
        imageio.mimsave(path.with_suffix(".gif"), gif_frames, fps=max(fps // 2, 1), loop=0)
    return path


def plot_model(ax, field, x, z, *, cmap="viridis", label="", **kw):
    """Draw a property field with the standard depth-down orientation."""
    im = ax.imshow(
        np.asarray(field).T,
        extent=(x[0], x[-1], z[-1], z[0]),
        aspect="auto",
        cmap=cmap,
        **kw,
    )
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("depth (m)")
    if label:
        cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(label)
    return im


def radargram_reference(datasets, t, *, gain_power=0.0):
    """Shared 0 dB reference for several :func:`plot_radargram` panels.

    The gain has to be applied before the reference is taken.  Measuring it on
    ungained data and then asking ``plot_radargram`` to gain the display puts
    every sample tens of dB off the reference -- with ``gain_power = 1`` and a
    2 us record that is +114 dB, so the whole section clips to the top of the
    colour scale.  Going through one function keeps the two in step.
    """

    t = np.asarray(t, dtype=float)
    gain = spreading_gain(t, power=gain_power)[:, None] if gain_power else 1.0
    return max(
        float(np.max(np.abs(analytic(np.asarray(d, dtype=float) * gain, axis=0))))
        for d in datasets
    )


def plot_radargram(
    ax,
    data,
    positions,
    t,
    *,
    db=True,
    dyn_range=60.0,
    gain_power=0.0,
    cmap="Greys",
    depth_axis=None,
    title="",
    ref=None,
):
    """Plot a common-offset section the way a processed radargram is shown.

    ``data`` has shape ``(n_t, n_trace)``.  With ``db=True`` the envelope is
    shown in dB below its peak, which is how the Open Polar Radar products in
    the SCAR figures are displayed.

    Pass ``ref`` to fix the 0 dB reference across several panels -- build it
    with :func:`radargram_reference` so it carries the same gain.  Without it
    each panel is scaled to its own maximum, which hides exactly the amplitude
    difference a polarisation comparison is meant to show.

    ``depth_axis`` gives the depth of every time sample.  Converted through a
    real slowness profile it is not evenly spaced, so the rows are resampled
    onto an even depth grid before display; ``imshow`` can only stretch an
    image linearly between the extent limits, and feeding it a curved axis
    would misplace everything between the two ends.

    Returns the image and the reference level actually used.
    """

    data = np.asarray(data, dtype=float)
    if gain_power:
        data = data * spreading_gain(t, power=gain_power)[:, None]

    if db:
        env = np.abs(analytic(data, axis=0))
        ref = float(np.max(env)) if ref is None else float(ref)
        img = to_db(env, floor=-dyn_range, ref=ref)
        vmin, vmax = -dyn_range, 0.0
        label = "returned power (dB)"
    else:
        img = data
        lim = np.percentile(np.abs(data), 99.5)
        vmin, vmax = -lim, lim
        label = "amplitude"

    if depth_axis is None:
        yaxis = np.asarray(t, dtype=float) * 1e6
    else:
        yaxis = np.asarray(depth_axis, dtype=float)
        even = np.linspace(yaxis[0], yaxis[-1], yaxis.size)
        span = abs(yaxis[-1] - yaxis[0]) or 1.0
        if np.max(np.abs(yaxis - even)) > 1e-6 * span:
            img = np.column_stack([np.interp(even, yaxis, col) for col in img.T])
            yaxis = even

    im = ax.imshow(
        img,
        extent=(positions[0], positions[-1], yaxis[-1], yaxis[0]),
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("depth (m)" if depth_axis is not None else "two-way time (us)")
    if title:
        ax.set_title(title, loc="left")
    cb = ax.figure.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(label)
    return im, ref
