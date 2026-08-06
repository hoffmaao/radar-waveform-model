"""Helpers shared by the example scripts.

Caching the snapshot stacks means the look of a movie can be re-tuned for a
talk without paying for the simulation again -- run once, then iterate with
``--render-only``.
"""

from pathlib import Path

import numpy as np


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
