"""Chirped radar systems: the transmitted waveform and the range profile.

The other examples in this package sound the ice with an impulse, which is the
right wavelet when the question is "where are the reflectors".  It is the wrong
one when the question is "how far did they move".  A phase-sensitive
measurement lives on the phase of a range-compressed echo, and phase is only
meaningful relative to a known carrier - which means a *band-limited* waveform
with a centre frequency, not an impulse whose instantaneous frequency wanders
across an octave.  Every real ice sounder is band-limited for exactly this
reason, and the two modelled here bracket the ground-based range:

* **ApRES**, 200-400 MHz, FMCW: a 1 s chirp, deramped on receive and sampled at
  40 kHz.  The instrument the phase-sensitive method is named after.
* **accum3**, 600-900 MHz, pulsed LFM: the CReSIS/OPR accumulation radar, as
  configured for the 2022_Antarctica_Ground traverse
  (``default_radar_params_2022_Antarctica_Ground_accum.m``, 'Survey Mode
  600-900 MHz').  Compressed against the transmit chirp rather than deramped.

WHAT A BAND BUYS AND COSTS.  Two numbers follow from the band alone and they
pull against each other:

* range resolution ``c / (2 B sqrt(eps))`` - how finely two reflectors can be
  separated.  Set by the *bandwidth*: 0.42 m for ApRES in ice, 0.28 m for
  accum3.
* fringe spacing ``c / (2 f_c sqrt(eps))`` - how much a reflector has to move
  to advance the interferometric phase by one full turn.  Set by the *centre
  frequency*: 0.28 m for ApRES, 0.11 m for accum3.

So the higher band measures a displacement more finely and wraps two and a half
times as fast doing it.  That trade is the whole difference between the two
repeat-pass examples, which sound the same deforming column with the same chirp
chain and differ only in the band.

DERAMP VERSUS MATCHED FILTER.  A real ApRES multiplies the received chirp by
the transmit reference and Fourier-transforms the beat, because a 1 s chirp
cannot be digitised at RF and a beat at tens of kHz can.  A pulsed system
matched-filters instead.  On a linear medium the two produce the same
band-limited complex range profile - the deramp trades bandwidth for record
length and nothing else - and :func:`deramp` and :func:`range_compress` are
both here so that equivalence can be exercised rather than asserted (see
``tests/test_waveform_and_deform.py``).

The examples compress with :func:`range_compress`, because an FDTD record is
microseconds long and deramp-on-receive needs a chirp far longer than the
two-way time to the deepest target.  What the examples inject is still a real
chirp across the real band (:func:`lfm_chirp`), so the compressed wavelet, the
range resolution and the sidelobe structure are the instrument's own; only the
time-bandwidth product, and with it the SNR gain the long chirp exists to
provide, is not reproduced.  These traces carry no noise, so there is no SNR
for it to improve.
"""

from dataclasses import dataclass

import numpy as np

from .constants import C0, EPS_ICE_MEAN

__all__ = [
    "RadarSystem",
    "APRES",
    "ACCUM3",
    "lfm_chirp",
    "range_compress",
    "deramp",
    "beat_profile",
]


@dataclass(frozen=True)
class RadarSystem:
    """A chirped sounder, described by the band it actually transmits.

    Parameters
    ----------
    name : str
        Instrument name, used in figure labels.
    f_start, f_stop : float
        Band edges (Hz).
    chirp_duration : float
        The real instrument's pulse duration (s).  It does not enter the FDTD
        runs - see the module docstring - but it sets the deramp chain and is
        what makes ``time_bandwidth`` meaningful.
    sample_rate : float
        The real instrument's digitiser rate (Hz).  For an FMCW system this is
        the *deramped* rate, which is why ApRES samples at 40 kHz while
        transmitting 200 MHz of bandwidth.
    fmcw : bool
        Whether the instrument deramps on receive.
    """

    name: str
    f_start: float
    f_stop: float
    chirp_duration: float
    sample_rate: float
    fmcw: bool = False

    @property
    def bandwidth(self):
        return self.f_stop - self.f_start

    @property
    def fc(self):
        """Centre frequency (Hz)."""
        return 0.5 * (self.f_start + self.f_stop)

    @property
    def fractional_bandwidth(self):
        return self.bandwidth / self.fc

    @property
    def chirp_rate(self):
        """``K`` (Hz/s), the rate the instantaneous frequency sweeps."""
        return self.bandwidth / self.chirp_duration

    @property
    def time_bandwidth(self):
        return self.bandwidth * self.chirp_duration

    def range_resolution(self, eps=EPS_ICE_MEAN):
        """Two-way range resolution (m) in a medium of permittivity ``eps``.

        The Fourier limit, which only an unwindowed compression achieves.  A
        Blackman band window - what :func:`range_compress` applies by default,
        and what the ApRES chain uses - trades about 1.65x of this for far
        lower range sidelobes: 0.42 m becomes 0.69 m for ApRES and 0.28 m
        becomes 0.46 m for accum3.
        """
        return C0 / (2.0 * self.bandwidth * np.sqrt(eps))

    def fringe_spacing(self, eps=EPS_ICE_MEAN):
        """Displacement (m) that advances the interferometric phase by 2 pi."""
        return C0 / (2.0 * self.fc * np.sqrt(eps))

    def grid_step(self, eps_max=EPS_ICE_MEAN, nodes_per_wavelength=12.0):
        """Field spacing (m) resolving the band edge at the stated sampling.

        The *top* of the band sets this, not the centre: the shortest
        wavelength the source puts into the grid is the one that has to be
        resolved, and undersampling it does not just lose that frequency, it
        disperses it onto the rest of the pulse.
        """
        return C0 / (self.f_stop * np.sqrt(eps_max) * nodes_per_wavelength)

    def summary(self, eps=EPS_ICE_MEAN):
        return (
            f"{self.name}: {self.f_start / 1e6:.0f}-{self.f_stop / 1e6:.0f} MHz "
            f"(fc {self.fc / 1e6:.0f} MHz, {100 * self.fractional_bandwidth:.0f} percent), "
            f"range resolution {self.range_resolution(eps):.2f} m, "
            f"fringe {100 * self.fringe_spacing(eps):.1f} cm in ice"
        )


#: ApRES as normally deployed: 200-400 MHz swept in 1 s, deramped to 40 kHz.
#: The chirp gradient this implies, 200 MHz/s, is the value the McMurdo
#: processing chain in ``EAGER_ApRES`` carries as its default.
APRES = RadarSystem("ApRES", 200e6, 400e6, chirp_duration=1.0,
                    sample_rate=40e3, fmcw=True)

#: The CReSIS/OPR accumulation radar in its 600-900 MHz survey mode, as flown
#: on the 2022_Antarctica_Ground traverse whose repeat passes the EAGER
#: vertical-deformation product is built from.  2 us is the survey-mode pulse
#: duration and 1 GHz the digitiser rate in the mission parameter file.
ACCUM3 = RadarSystem("accum3", 600e6, 900e6, chirp_duration=2e-6,
                     sample_rate=1000e6, fmcw=False)


def _tukey(n, alpha):
    """Tukey window, written out rather than imported from scipy.signal.

    ``radarwave`` deliberately has no scipy dependency in its model path.
    """
    if n < 1:
        return np.ones(max(n, 0))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0:
        return np.ones(n)
    x = np.linspace(0.0, 1.0, n)
    w = np.ones(n)
    lo = x < alpha / 2.0
    hi = x > 1.0 - alpha / 2.0
    w[lo] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[lo] - alpha / 2.0)))
    w[hi] = 0.5 * (1.0 + np.cos(2.0 * np.pi / alpha * (x[hi] - 1.0 + alpha / 2.0)))
    return w


def lfm_chirp(system, t, tpd, t0=None, taper=0.25):
    """Linear-FM burst across ``system``'s band, for injection into the FDTD.

    Parameters
    ----------
    system : RadarSystem
        Supplies the band edges.  The sweep runs ``f_start`` to ``f_stop`` over
        ``tpd``, so the *band* is the instrument's even though the duration is
        not (see the module docstring).
    t : array_like
        Record time vector (s).
    tpd : float
        Duration of the burst (s).  ``tpd * bandwidth`` is the time-bandwidth
        product actually simulated; a few tens is enough for the compressed
        wavelet to take its asymptotic shape, and every extra microsecond is an
        extra microsecond of record that has to be simulated before the first
        echo can arrive.
    t0 : float, optional
        Launch time; defaults to zero, so the compressed record's time axis is
        two-way traveltime from transmit.
    taper : float
        Tukey taper fraction on the burst envelope.  A rectangular envelope
        rings: its spectrum is the band convolved with a sinc, which puts
        energy outside the band the grid was sized for and raises the
        compressed sidelobes.  0.25 costs about 12 percent of the duration and
        drops the first sidelobe well below the reflections these examples are
        about.

    Returns
    -------
    ndarray
        The burst, normalised to a peak absolute value of one, zero outside
        ``[t0, t0 + tpd]``.
    """
    t = np.asarray(t, dtype=float)
    t0 = 0.0 if t0 is None else float(t0)
    tau = t - t0
    inside = (tau >= 0.0) & (tau <= tpd)
    k = system.bandwidth / float(tpd)
    p = np.zeros_like(t)
    p[inside] = np.cos(2.0 * np.pi * (system.f_start * tau[inside]
                                      + 0.5 * k * tau[inside] ** 2))
    n_in = int(np.count_nonzero(inside))
    p[inside] *= _tukey(n_in, taper)
    peak = np.max(np.abs(p))
    return p / peak if peak > 0 else p


def _band_window(freqs, system, name="blackman"):
    """Amplitude taper across ``system``'s band, zero outside it.

    Range sidelobes are what a windowed compression buys, and they matter here
    for the same reason they matter in a real product: a reflector 40 dB down
    and half a metre from a bright one is invisible under an unwindowed sinc
    skirt.  Blackman is what the ApRES chain uses.
    """
    if name is None:
        return np.ones_like(freqs)
    lo, hi = system.f_start, system.f_stop
    u = (freqs - lo) / (hi - lo)
    w = np.zeros_like(freqs)
    inside = (u >= 0.0) & (u <= 1.0)
    v = u[inside]
    if name == "blackman":
        w[inside] = 0.42 - 0.5 * np.cos(2 * np.pi * v) + 0.08 * np.cos(4 * np.pi * v)
    elif name == "hann":
        w[inside] = 0.5 - 0.5 * np.cos(2 * np.pi * v)
    elif name == "rect":
        w[inside] = 1.0
    else:
        raise ValueError(f"unknown window {name!r}")
    return w


def _analytic_from_onesided(spec, nfft, n):
    """Inverse-transform a one-sided spectrum into an analytic time series.

    Doubling the positive-frequency bins and leaving the negative ones empty is
    the definition of the analytic signal, so the compression and the Hilbert
    transform happen in the same FFT.  Doing them separately would mean a
    second transform of a record that opens with a pulse 60-120 dB above every
    echo, which is the wrap-around problem
    :func:`radarwave.polarimetry.analytic` exists to work around.
    """
    full = np.zeros(nfft, dtype=complex)
    full[: spec.size] = 2.0 * spec
    full[0] = spec[0]
    if nfft % 2 == 0:
        full[nfft // 2] = spec[-1]
    return np.fft.ifft(full)[:n]


def range_compress(trace, wavelet, dt, system, *, window="blackman", pad=4):
    """Matched-filter range compression to a complex range profile.

    Correlating the record against the transmitted wavelet collapses each echo
    onto the wavelet's zero-phase autocorrelation, which is the shape a
    processed radar product shows and the shape whose phase means something.
    The result is returned as an *analytic* signal, so its magnitude is the
    range envelope and its argument is the interferometric phase.

    Because ``wavelet`` carries the same time origin as ``trace``, the launch
    time divides out and the profile's time axis is two-way traveltime.

    Parameters
    ----------
    trace : (n,) array
        Recorded real trace.
    wavelet : (n,) array
        The transmitted waveform on the same time axis.
    dt : float
        Sample interval (s) of both.
    system : RadarSystem
        Supplies the band the window is applied across.
    window : str or None
        ``"blackman"`` (default), ``"hann"``, ``"rect"``, or ``None`` for no
        band window at all.  ``None`` leaves the match unwindowed, which is
        sharper and rings.
    pad : int
        Zero-padding factor for the FFT.  At least 2 is needed for the
        correlation not to wrap; more refines where the envelope peak lands,
        which is where the phase gets read.

    Returns
    -------
    ndarray
        Complex profile, same length as ``trace``.
    """
    trace = np.asarray(trace, dtype=float)
    wavelet = np.asarray(wavelet, dtype=float)
    if trace.shape != wavelet.shape:
        raise ValueError(
            f"trace and wavelet must have the same shape: "
            f"{trace.shape} vs {wavelet.shape}"
        )
    n = trace.size
    nfft = int(2 ** np.ceil(np.log2(max(int(pad), 2) * n)))
    spec = np.fft.rfft(trace, nfft) * np.conj(np.fft.rfft(wavelet, nfft))
    if window is not None:
        spec = spec * _band_window(np.fft.rfftfreq(nfft, dt), system, window)
    return _analytic_from_onesided(spec, nfft, n)


def deramp(rx, t, system, tpd=None):
    """Mix a received FMCW chirp against the transmit reference.

    This is what an ApRES does in hardware: the returned sweep is multiplied by
    a copy of the transmitted one, and what survives the low-pass is a beat
    whose frequency is proportional to the target's delay,
    ``f_beat = K * tau``.  A 1 s / 200 MHz sweep against a target 1000 m into
    ice puts that beat at 2.4 kHz, which is why 40 kHz of digitiser is enough
    to record 200 MHz of bandwidth.

    The reference is the *analytic* chirp, so the product is complex and the
    sum-frequency term never appears - the hardware low-pass is implicit rather
    than approximated.  The residual video phase ``pi K tau^2``, which a real
    processor has to correct for and which is a full radian at 1000 m, is left
    in: :func:`beat_profile` removes it, and removing it here would leave a
    "beat signal" that no instrument produces.

    Parameters
    ----------
    rx : (n,) array
        Received real record covering the sweep.
    t : (n,) array
        Time vector (s), with the sweep starting at ``t = 0``.
    system : RadarSystem
        Supplies ``f_start`` and the chirp rate.
    tpd : float, optional
        Sweep duration; defaults to ``system.chirp_duration``.

    Returns
    -------
    ndarray
        Complex beat signal, zero outside the sweep.
    """
    t = np.asarray(t, dtype=float)
    rx = np.asarray(rx, dtype=float)
    tpd = system.chirp_duration if tpd is None else float(tpd)
    k = system.bandwidth / tpd
    ref = np.exp(-2j * np.pi * (system.f_start * t + 0.5 * k * t**2))
    return np.where((t >= 0.0) & (t <= tpd), rx * ref, 0.0)


def beat_profile(beat, t, system, tpd=None, *, window="blackman", pad=8):
    """Range profile from a deramped FMCW beat signal.

    Transforms the beat over the sweep, maps beat frequency to two-way delay
    through ``tau = f_beat / K``, and removes the residual video phase
    ``pi K tau^2`` so the returned phase is the ``-2 pi f_start tau`` a
    range measurement wants.

    Parameters
    ----------
    beat : (n,) array
        Complex beat signal from :func:`deramp`.
    t : (n,) array
        Its time vector (s).
    system, tpd
        As for :func:`deramp`.
    window : str or None
        Amplitude taper applied across the sweep before the transform.  Here it
        is a taper in *time*, because for a deramped system time is frequency;
        the effect on the range sidelobes is the same one
        :func:`range_compress` gets from its band window.
    pad : int
        Zero-padding factor, which sets how finely the delay axis is sampled.

    Returns
    -------
    tau, profile
        Two-way delay axis (s) and the complex range profile on it.
    """
    beat = np.asarray(beat)
    t = np.asarray(t, dtype=float)
    tpd = system.chirp_duration if tpd is None else float(tpd)
    dt = t[1] - t[0]
    k = system.bandwidth / tpd

    inside = (t >= 0.0) & (t <= tpd)
    seg = beat[inside]
    n = seg.size
    if window is not None:
        # Reuse the band-window shapes by mapping sweep time onto sweep
        # frequency, which for a linear FM chirp is the same axis.
        seg = seg * _band_window(system.f_start + k * t[inside], system, window)

    nfft = int(2 ** np.ceil(np.log2(max(int(pad), 1) * n)))
    spec = np.fft.fft(seg, nfft) / n
    freq = np.fft.fftfreq(nfft, dt)
    # Deramping against the conjugate reference puts a target at delay tau at
    # beat frequency -K tau, so only negative beat frequencies carry targets.
    # The positive ones hold the sum-frequency term - the half of the real
    # cosine the mixer folds up to about -2 f_start rather than down - which a
    # real receiver never digitises because its anti-alias filter stops at half
    # the deramped sample rate.  Applying that same bound here is what makes
    # the deramp chain end where the instrument's does, at an unambiguous range
    # of ``sample_rate / (2 K)`` in two-way time.
    keep = (freq <= 0.0) & (freq >= -0.5 * system.sample_rate)
    tau = -freq[keep] / k
    order = np.argsort(tau)
    tau = tau[order]
    prof = spec[keep][order]
    return tau, prof * np.exp(-1j * np.pi * k * tau**2)

