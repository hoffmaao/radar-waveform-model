"""Backwards-compatible shim over the :mod:`radarwave` package.

``crevasse.ipynb`` and the older scripts do ``from model import *`` and then
call ``finddx``, ``finddt``, ``blackharrispulse``, ``padgrid``, ``gridinterp``
and ``TM_model2d``.  Those names are kept here and forwarded to the package so
the notebook still runs.

The original implementation is preserved as ``legacy_model_reference.py`` for
comparison, but it no longer imports on current SciPy (it uses ``interp2d``,
removed in 1.14) and its absorbing boundary reflects about 22 dB more than the
current one.  New work should import from :mod:`radarwave` directly.
"""

import warnings

import numpy as np

from radarwave import FDTD2D, PropertyGrid
from radarwave.grid import max_spatial_step, max_time_step, pad_grid, regrid
from radarwave.sources import blackharrispulse

__all__ = [
    "finddx",
    "finddt",
    "blackharrispulse",
    "padgrid",
    "gridinterp",
    "TM_model2d",
    "TE_model2d",
]


def finddx(epmax, mumax, srcpulse, t, thres=0.02):
    """Deprecated alias for :func:`radarwave.grid.max_spatial_step`."""
    return max_spatial_step(epmax, mumax, srcpulse, t, thres)


def finddt(epmin, mumin, dx, dz):
    """Deprecated alias for :func:`radarwave.grid.max_time_step`."""
    return max_time_step(epmin, mumin, dx, dz)


def padgrid(A, x, z, n):
    """Deprecated alias for :func:`radarwave.grid.pad_grid`."""
    return pad_grid(A, x, z, n)


def gridinterp(A, x, z, x2, z2, method="nearest"):
    """Deprecated alias for :func:`radarwave.grid.regrid`.

    ``method='nearest'`` is passed through; the old SciPy ``interp2d`` spelling
    of ``'linear'``/``'cubic'`` maps onto the equivalent
    ``RegularGridInterpolator`` methods.
    """
    return regrid(A, x, z, x2, z2, method=method)


def _run(mode, ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t, npml,
         outstep=1, plotopt=None, **kwargs):
    if plotopt is not None and len(plotopt) and plotopt[0]:
        warnings.warn(
            "live plotting during the time loop is no longer supported; use "
            "snapshot_every with radarwave.viz.wavefield_movie instead",
            stacklevel=3,
        )
    ep = np.asarray(ep, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sig = np.asarray(sig, dtype=float)
    xprop = np.asarray(xprop, dtype=float)
    zprop = np.asarray(zprop, dtype=float)

    # The staggered scheme needs an odd number of property rows and columns.
    # np.arange on the half-spacing frequently lands on an even count, and the
    # original code responded by printing a message and returning None, which
    # then surfaced as a confusing unpacking error.  Trim instead, and say so.
    ni, nj = ep.shape
    if ni % 2 == 0 or nj % 2 == 0:
        ni -= ni % 2 == 0
        nj -= nj % 2 == 0
        warnings.warn(
            f"property grid was {ep.shape}; trimming to ({ni}, {nj}) so it has an "
            "odd number of rows and columns, as the staggered grid requires",
            stacklevel=3,
        )
        ep, mu, sig = ep[:ni, :nj], mu[:ni, :nj], sig[:ni, :nj]
        xprop, zprop = xprop[:ni], zprop[:nj]

    grid = PropertyGrid(
        x=xprop,
        z=zprop,
        eps={k: ep for k in ("xx", "yy", "zz")},
        mu=mu,
        sig={k: sig for k in ("xx", "yy", "zz")},
    )
    dt = t[1] - t[0]
    sim = FDTD2D(grid, dt, npml=npml, mode=mode)
    res = sim.run(
        np.atleast_2d(srcloc)[:, :2],
        srcpulse,
        np.atleast_2d(recloc)[:, :2],
        outstep=outstep,
        **kwargs,
    )
    return res.gather, res.t, res.src[:, 0], res.src[:, 1], res.rec[:, 0], res.rec[:, 1]


def TM_model2d(ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t, npml,
               outstep=1, plotopt=None, **kwargs):
    """Out-of-plane (``Ey``) FDTD, matching the original call signature.

    Returns ``(gather, tout, srcx, srcz, recx, recz)``.
    """
    return _run("TM", ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t,
                npml, outstep, plotopt, **kwargs)


def TE_model2d(ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t, npml,
               outstep=1, plotopt=None, **kwargs):
    """In-plane (``Ex``, ``Ez``) FDTD.

    Never ported to Python in the original repository; provided here for
    parity with ``matlab/TE_model2d.m``.  ``srcloc``/``recloc`` may carry a
    third column in the MATLAB convention (1 = Ex, 2 = Ez); only Ex sources are
    wired up, which is the horizontally polarised surface antenna case.
    """
    return _run("TE", ep, mu, sig, xprop, zprop, srcloc, recloc, srcpulse, t,
                npml, outstep, plotopt, source_component="Ex",
                receiver_component="Ex", **kwargs)
