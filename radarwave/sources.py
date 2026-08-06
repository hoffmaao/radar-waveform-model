"""Source wavelets for the FDTD solvers."""

import numpy as np

__all__ = ["blackharrispulse", "ricker", "gaussian_derivative", "dominant_frequency"]


def blackharrispulse(fr, t):
    """Blackman-Harris source pulse of Chen et al. (1997).

    This is the wavelet used by Irving and Knight (2006) and is the default
    throughout this package: it is compact in time and has a well behaved,
    band-limited spectrum with no DC component.

    Parameters
    ----------
    fr : float
        Nominal source frequency (Hz).  The spectral peak lands near
        ``1.4 * fr``; use :func:`dominant_frequency` for the actual peak.
    t : array_like
        Time vector (s), uniformly sampled and starting at zero.

    Returns
    -------
    ndarray
        Pulse amplitude, normalised to a peak absolute value of one.
    """
    t = np.asarray(t, dtype=float)
    a = np.array([0.35322222, -0.488, 0.145, -0.010222222])
    T = 1.14 / fr
    window = np.zeros_like(t)
    for n in range(4):
        window += a[n] * np.cos(2 * n * np.pi * t / T)
    window[t >= T] = 0.0

    # The pulse is the time derivative of the window.  ``np.diff`` divided by
    # dt keeps the amplitude independent of the sampling interval before the
    # final normalisation, which matters when comparing runs at different dt.
    dt = t[1] - t[0]
    p = np.concatenate(([0.0], np.diff(window))) / dt
    peak = np.max(np.abs(p))
    if peak > 0:
        p /= peak
    return p


def ricker(fc, t, t0=None):
    """Ricker (Mexican-hat) wavelet with centre frequency ``fc`` (Hz)."""
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = 1.0 / fc
    arg = (np.pi * fc * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def gaussian_derivative(fc, t, t0=None):
    """First derivative of a Gaussian with centre frequency ``fc`` (Hz)."""
    t = np.asarray(t, dtype=float)
    tau = 1.0 / (np.pi * fc)
    if t0 is None:
        t0 = 3.0 * tau
    x = (t - t0) / tau
    p = -x * np.exp(-(x**2) / 2.0)
    peak = np.max(np.abs(p))
    if peak > 0:
        p /= peak
    return p


def gabor(fc, t, bandwidth=0.2, t0=None):
    """Gaussian-modulated cosine: a band-limited pulse of known centre frequency.

    Unlike the Blackman-Harris pulse, whose fractional bandwidth is of order
    one, this has a controllable narrow band.  That matters for interferometry:
    the phase of ``HH * conj(VV)`` only equals ``2 pi fc * delay`` when the
    signal is narrowband, and a broadband trace has an instantaneous frequency
    that wanders far enough to make the unwrapped phase unusable.  Real
    ice-sounding products are band-limited (MCoRDS is 180-210 MHz, about 15
    percent), so this is also the more faithful wavelet for synthesising one.

    Parameters
    ----------
    fc : float
        Centre frequency (Hz).
    t : array_like
        Time vector (s).
    bandwidth : float
        Fractional full-width-at-half-maximum bandwidth of the amplitude
        spectrum.
    t0 : float, optional
        Centre time; defaults to the middle of ``t``.
    """
    t = np.asarray(t, dtype=float)
    if t0 is None:
        t0 = 0.5 * (t[0] + t[-1])
    sigma_f = bandwidth * fc / 2.3548
    sigma_t = 1.0 / (2.0 * np.pi * sigma_f)
    x = t - t0
    p = np.exp(-(x**2) / (2.0 * sigma_t**2)) * np.cos(2.0 * np.pi * fc * x)
    peak = np.max(np.abs(p))
    return p / peak if peak > 0 else p


def dominant_frequency(pulse, dt):
    """Frequency (Hz) of the peak of a pulse's amplitude spectrum."""
    pulse = np.asarray(pulse, dtype=float)
    n = int(2 ** np.ceil(np.log2(len(pulse))) * 4)
    spec = np.abs(np.fft.rfft(pulse, n))
    freqs = np.fft.rfftfreq(n, dt)
    return float(freqs[np.argmax(spec)])
