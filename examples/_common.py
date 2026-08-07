"""Helpers shared by the example scripts.

Caching the snapshot stacks means the look of a movie can be re-tuned for a
talk without paying for the simulation again -- run once, then iterate with
``--render-only``.
"""

from pathlib import Path

import numpy as np


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


def save_snapshots(path, panels, x, z, times, **extra):
    """Cache snapshot stacks and their axes to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"x": x, "z": z, "times": times, "labels": np.array([lbl for _, lbl in panels])}
    for k, (stack, _) in enumerate(panels):
        data[f"panel{k}"] = np.asarray(stack, dtype=np.float32)
    data.update({k: np.asarray(v) for k, v in extra.items()})
    np.savez_compressed(path, **data)
    return path


def load_snapshots(path):
    """Return ``(panels, x, z, times, extra)`` from a cache written above."""
    with np.load(Path(path), allow_pickle=False) as f:
        labels = [str(s) for s in f["labels"]]
        panels = [(f[f"panel{k}"], labels[k]) for k in range(len(labels))]
        extra = {
            k: f[k] for k in f.files
            if not k.startswith("panel") and k not in ("x", "z", "times", "labels")
        }
        return panels, f["x"], f["z"], f["times"], extra
