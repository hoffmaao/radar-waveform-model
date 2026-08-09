"""Helpers shared by the example scripts.

Caching the snapshot stacks means the look of a movie can be re-tuned for a
talk without paying for the simulation again -- run once, then iterate with
``--render-only``.

A cache carries a stamp of the parameters that define the model it came from,
and :func:`load_snapshots` refuses one whose stamp does not match what the
caller is asking for.  Without that, changing a model constant and re-rendering
draws the old wavefield and the old trace underneath the predicted numbers
computed from the new constants, and nothing in the figure says so.
"""

from pathlib import Path

import numpy as np


class StaleCache(RuntimeError):
    """A cache was written from a different model than the one asked for."""


def echo_time(column, depth, z_ant, t_wave, dip_deg=0.0):
    """When the echo from ``depth`` peaks in the record (s).

    Everything these examples mark on a time axis or compare with a recorded
    arrival goes through here, because a traveltime on its own is not the same
    quantity as the envelope peak that gets measured.  Two corrections separate
    them, each tens of ns and so each wider than a reflection:

    * the path starts at the antenna, not at the snow surface -- for a
      transmitter flown a metre up that is the two-way air leg, and for one
      buried in the firn it is a leg that never happened;
    * ``t_wave`` is the offset of the wavelet's own envelope peak from the
      start of the record (:func:`radarwave.envelope_peak_time`).

    ``dip_deg`` inclines the ray away from nadir: a specular return off a plane
    dipping at ``delta`` travels the same depth interval along a path
    ``1 / cos(delta)`` longer, air leg included.
    """
    sec = 1.0 / np.cos(np.deg2rad(dip_deg))
    return column.two_way_time(np.asarray(depth, dtype=float), z0=z_ant) * sec + t_wave


def save_figure(fig, path, bare=False):
    """Write ``fig``, stripping its words first when ``bare``.

    Every example saves through here so ``--bare`` cannot be wired into some
    figures and forgotten in others.
    """
    if bare:
        from radarwave.viz import strip_text

        strip_text(fig)
    fig.savefig(path)
    return path


def stampable(params):
    """Constructor arguments with the callables dropped.

    A stamp is compared as text and a function object has no stable text, so a
    callable argument cannot go in one.  Passing the whole argument mapping
    through here rather than hand-listing its keys at the call site is what keeps
    a parameter added later from escaping the guard; whatever the callables close
    over is stamped separately alongside.
    """
    return {k: v for k, v in params.items() if not callable(v)}


def _stamp_text(value):
    """Canonical text for one stamp entry.

    Text rather than the raw value so a stamp can hold floats, tuples, arrays
    and nested dictionaries side by side and still be compared key by key, and
    so a mismatch can be reported in a form a reader recognises.
    """
    if value is None:
        return "None"
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{float(value):.10g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_stamp_text(v)}"
                               for k, v in sorted(value.items())) + "}"
    try:
        items = list(value)
    except TypeError:
        return str(value)
    return "(" + ", ".join(_stamp_text(v) for v in items) + ")"


def _stamp_pairs(stamp, prefix=""):
    """Flatten a stamp to ``{dotted name: text}``.

    A mapping of constructor arguments is flattened rather than stamped whole, so
    a mismatch inside one names the parameter that moved -- ``column.sigma_ice``
    -- instead of printing two mappings and leaving the reader to find the
    difference.
    """
    pairs = {}
    for key, value in stamp.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            pairs.update(_stamp_pairs(value, prefix=f"{name}."))
        else:
            pairs[name] = _stamp_text(value)
    return pairs


_STAMP_KEYS = "_stamp_keys"
_STAMP_VALUES = "_stamp_values"
_RESERVED = ("x", "z", "times", "labels", _STAMP_KEYS, _STAMP_VALUES)


def _check_stamp(path, cached, stamp):
    """Raise :class:`StaleCache` unless ``cached`` matches ``stamp``."""
    want = _stamp_pairs(stamp)
    advice = (f"Rendering from it would draw the cached wavefield and trace under "
              f"the numbers predicted for the current model.  Delete {path} or "
              f"re-run without --render-only.")
    if cached is None:
        raise StaleCache(
            f"{path} carries no model stamp, so there is no way to tell which "
            f"model it came from.  {advice}"
        )
    changed = [k for k in sorted(set(want) | set(cached)) if want.get(k) != cached.get(k)]
    if changed:
        detail = "\n".join(
            f"  {k}: cache has {cached.get(k, '<absent>')}, model now has "
            f"{want.get(k, '<absent>')}" for k in changed
        )
        raise StaleCache(
            f"{path} was written from a different model:\n{detail}\n{advice}"
        )


def save_snapshots(path, panels, x, z, times, stamp=None, **extra):
    """Cache snapshot stacks and their axes to a compressed ``.npz``.

    ``stamp`` is a mapping of the parameters that define the model these
    snapshots came from.  It is written alongside them so a later
    :func:`load_snapshots` can tell whether the cache still describes the model
    being asked about.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"x": x, "z": z, "times": times, "labels": np.array([lbl for _, lbl in panels])}
    for k, (stack, _) in enumerate(panels):
        data[f"panel{k}"] = np.asarray(stack, dtype=np.float32)
    data.update({k: np.asarray(v) for k, v in extra.items()})
    if stamp is not None:
        pairs = _stamp_pairs(stamp)
        keys = sorted(pairs)
        data[_STAMP_KEYS] = np.array(keys)
        data[_STAMP_VALUES] = np.array([pairs[k] for k in keys])
    np.savez_compressed(path, **data)
    return path


def load_snapshots(path, stamp=None):
    """Return ``(panels, x, z, times, extra)`` from a cache written above.

    When ``stamp`` is given it is compared with the one the cache carries, and a
    :class:`StaleCache` naming every parameter that differs is raised rather
    than handing back snapshots from a different model.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        if stamp is not None:
            cached = None
            if _STAMP_KEYS in f.files and _STAMP_VALUES in f.files:
                cached = {str(k): str(v) for k, v in zip(f[_STAMP_KEYS], f[_STAMP_VALUES])}
            _check_stamp(path, cached, stamp)
        labels = [str(s) for s in f["labels"]]
        panels = [(f[f"panel{k}"], labels[k]) for k in range(len(labels))]
        extra = {
            k: f[k] for k in f.files
            if not k.startswith("panel") and k not in _RESERVED
        }
        return panels, f["x"], f["z"], f["times"], extra
