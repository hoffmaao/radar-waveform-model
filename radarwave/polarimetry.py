"""Polarimetric processing: interferograms, delays and fabric inversion.

These are the operations applied to real quad-polarimetric Open Polar Radar
frames, written so a synthetic trace pair produced here can be pushed through
exactly the same steps as the data.
"""

import numpy as np
from scipy.signal import hilbert

from .constants import C0, DEPS_ICE

__all__ = [
    "analytic",
    "interferogram",
    "unwrap_phase",
    "delay_from_phase",
    "dlambda_from_delay",
    "synthetic_pair",
]


def analytic(trace, axis=0):
    """Complex analytic signal of a real trace."""
    return hilbert(np.asarray(trace, dtype=float), axis=axis)


def interferogram(hh, vv, axis=0):
    """Complex interferogram ``HH * conj(VV)``.

    Its phase is ``2 pi f_c * (t_VV - t_HH)``, i.e. proportional to the
    accumulated birefringent delay between the two eigenpolarisations, and its
    magnitude carries the coherence.
    """
    a = hh if np.iscomplexobj(hh) else analytic(hh, axis=axis)
    b = vv if np.iscomplexobj(vv) else analytic(vv, axis=axis)
    return a * np.conj(b)


def unwrap_phase(phase, axis=0):
    """Unwrap along the range axis."""
    return np.unwrap(np.asarray(phase), axis=axis)


def delay_from_phase(phase_unwrapped, fc):
    """Two-way traveltime difference (s) implied by an unwrapped phase."""
    return np.asarray(phase_unwrapped) / (2.0 * np.pi * fc)


def dlambda_from_delay(depth, dtau, column=None, smooth=None):
    """Invert a delay profile for the horizontal fabric contrast.

    The interval form of the forward model is

        d(dtau)/dz = 2 (S_x - S_y) ~= deps * dlam / (c * sqrt(eps_local)),

    so differentiating a measured ``dtau(z)`` recovers ``dlam = lam_x - lam_y``.
    Passing the :class:`~radarwave.ice.IceColumn` uses the local permittivity,
    which matters in firn where bubbles lower it well below the solid-ice value.

    Parameters
    ----------
    depth : (n,) array
        Depth (m), increasing.
    dtau : (n,) array
        Cumulative two-way delay ``t_x - t_y`` (s).
    column : IceColumn, optional
        Used for the local permittivity; solid ice is assumed otherwise.
    smooth : int, optional
        Length of a moving-average window applied to the gradient, in samples.
    """
    depth = np.asarray(depth, dtype=float)
    dtau = np.asarray(dtau, dtype=float)
    grad = np.gradient(dtau, depth)
    if smooth and smooth > 1:
        k = np.ones(int(smooth)) / float(int(smooth))
        grad = np.convolve(grad, k, mode="same")

    if column is None:
        eps_local = np.full_like(depth, 3.171)
    else:
        eps = column.permittivity(depth)
        eps_local = 0.5 * (eps[:, 0] + eps[:, 1])

    return grad * C0 * np.sqrt(eps_local) / DEPS_ICE


def delayed_copy(t, trace, dtau_of_t):
    """Resample ``trace`` so that each sample is delayed by ``dtau_of_t``.

    This is the compact way to build the second eigenpolarisation: the two
    traces see the same scatterers, and the only difference is that one has
    accumulated an extra traveltime that grows with range.  Warping the time
    axis reproduces that exactly and, unlike summing discrete reflectors,
    leaves no gaps where the interferogram has no signal and its phase is
    undefined.
    """
    t = np.asarray(t, dtype=float)
    return np.interp(t - np.asarray(dtau_of_t, dtype=float), t, np.asarray(trace, dtype=float))


def volume_scattering(t, rng, wavelet, decay_time=None):
    """A trace from continuously distributed volume scattering."""
    t = np.asarray(t, dtype=float)
    series = rng.normal(size=t.size)
    if decay_time:
        series = series * np.exp(-t / decay_time)
    return np.convolve(series, np.asarray(wavelet, dtype=float), mode="same")


def synthetic_pair(t, t0, reflectivity, dtau, wavelet):
    """Build an HH/VV trace pair separated by a birefringent delay.

    Each reflector arriving at ``t0[k]`` contributes the same wavelet to both
    polarisations, but the VV copy is delayed by ``dtau[k]``, which is what an
    eigenpolarised pair of traces looks like after coregistration.  ``t0`` is
    passed as traveltime rather than depth so that a depth-varying velocity
    (firn) is handled by the caller.

    Returns
    -------
    hh, vv : ndarray
        Real traces sampled on ``t``.
    """
    t = np.asarray(t, dtype=float)
    t0 = np.asarray(t0, dtype=float)
    reflectivity = np.asarray(reflectivity, dtype=float)
    dtau = np.asarray(dtau, dtype=float)

    dt = t[1] - t[0]
    hh = np.zeros_like(t)
    vv = np.zeros_like(t)

    for k in range(t0.size):
        for out, shift in ((hh, 0.0), (vv, dtau[k])):
            centre = t0[k] + shift
            idx = int(round(centre / dt))
            lo = idx - wavelet.size // 2
            hi = lo + wavelet.size
            if lo < 0 or hi > t.size:
                continue
            # Sub-sample placement matters: the delays being measured are a
            # small fraction of a sample.
            frac = centre / dt - idx
            w = np.interp(
                np.arange(wavelet.size) - frac,
                np.arange(wavelet.size),
                wavelet,
                left=0.0,
                right=0.0,
            )
            out[lo:hi] += reflectivity[k] * w

    return hh, vv
