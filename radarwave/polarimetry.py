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
    "subsample_lag",
    "isolate_arrival",
    "synthetic_pair",
]


def subsample_lag(a, b, dt):
    """Lag of ``b`` relative to ``a`` (s), from the cross-correlation peak.

    The delays this package measures are a fraction of a sample, so the peak is
    refined by fitting a parabola through it and its two neighbours.  A peak
    landing on an end sample has no such neighbours; the integer lag is
    returned there rather than wrapping around the correlation.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a - np.mean(a)
    b = b - np.mean(b)
    n = 1 << int(np.ceil(np.log2(max(a.size, 2) * 2)))
    cc = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    cc = np.concatenate([cc[-(n // 2):], cc[: n // 2]])
    k = int(np.argmax(cc))
    lag = float(k - n // 2)
    if 0 < k < cc.size - 1:
        denom = cc[k - 1] - 2.0 * cc[k] + cc[k + 1]
        if denom != 0.0:
            lag += 0.5 * (cc[k - 1] - cc[k + 1]) / denom
    return lag * dt


def isolate_arrival(t, traces, t0, halfwidth, taper=True):
    """Window ``traces`` onto a single arrival and strip the background wake.

    A monostatic 2-D line source leaves a slowly decaying wake behind the
    direct pulse.  A deep layer echo stands only about 10 dB above it, and
    across a window several cycles wide the wake still carries more energy than
    the echo -- so cross-correlating raw windows measures the wake, which has
    travelled nowhere and therefore has no delay, and the answer collapses
    towards zero.

    The wake varies slowly across a window a few cycles wide, so subtracting a
    least-squares line through each window removes it.  The Hann taper then
    weights the centre, where the arrival is, over the edges, where whatever is
    left of the background is, and keeps the window edges from ringing into the
    correlation.

    Parameters
    ----------
    t : (n,) array
        Time axis (s).
    traces : sequence of (n,) arrays
        Traces sharing that time axis.
    t0 : float
        Centre of the window (s).
    halfwidth : float
        Half-width of the window (s).
    taper : bool
        Apply the Hann taper.  Turn it off to read an amplitude off the window,
        since the taper weights two traces differently when their peaks sit at
        different places in it.

    Returns
    -------
    idx : ndarray of bool
        The window that was used.
    out : list of ndarray
        The windowed, detrended (and optionally tapered) traces.
    """
    t = np.asarray(t, dtype=float)
    idx = np.abs(t - float(t0)) < float(halfwidth)
    tw = t[idx]
    if tw.size < 8:
        return idx, [np.asarray(tr, dtype=float)[idx] for tr in traces]

    basis = np.vstack([np.ones_like(tw), tw - tw.mean()]).T
    window = np.hanning(tw.size) if taper else 1.0
    out = []
    for tr in traces:
        w = np.asarray(tr, dtype=float)[idx]
        coef, *_ = np.linalg.lstsq(basis, w, rcond=None)
        out.append((w - basis @ coef) * window)
    return idx, out


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
