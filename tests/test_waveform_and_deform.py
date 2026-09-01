"""Chirped waveforms, range compression, and the repeat-pass deformation model.

These are the pieces examples 5 and 6 rest on, and they are tested here rather
than through the examples because the examples are hours of FDTD.  What matters
is the chain that turns two records into a displacement: a chirp goes in, a
compressed profile comes out, and the phase between two of those profiles is
``-2 pi f_c`` times the change in two-way traveltime.  Every step of that has a
sign, and a sign error anywhere in it reports thinning as thickening.
"""

import numpy as np
import pytest

from radarwave import IceColumn, PropertyGrid, VerticalStrain
from radarwave.constants import C0, EPS_ICE_MEAN
from radarwave.polarimetry import interferogram, subsample_lag, upsampled_lag
from radarwave.scenes import IceModelBuilder, Layer
from radarwave.waveform import (
    ACCUM3,
    APRES,
    RadarSystem,
    beat_profile,
    deramp,
    lfm_chirp,
    range_compress,
)


def _shift(trace, dt, tau):
    """Delay ``trace`` by ``tau`` exactly, in the frequency domain.

    ``np.interp`` would be the obvious way and is not good enough here: these
    tests move a target by picoseconds to check a phase, and linear
    interpolation of a 400 MHz signal introduces its own amplitude and phase
    error that is larger than the effect under test.
    """
    n = trace.size
    nfft = int(2 ** np.ceil(np.log2(2 * n)))
    spec = np.fft.rfft(trace, nfft)
    spec = spec * np.exp(-2j * np.pi * np.fft.rfftfreq(nfft, dt) * tau)
    return np.fft.irfft(spec, nfft)[:n]


# ---------------------------------------------------------------- systems


def test_system_numbers_follow_from_the_band():
    assert APRES.fc == pytest.approx(300e6)
    assert APRES.bandwidth == pytest.approx(200e6)
    assert ACCUM3.fc == pytest.approx(750e6)

    # Resolution is set by bandwidth, fringe spacing by centre frequency, and
    # the two examples turn on those being different comparisons.
    assert APRES.range_resolution() == pytest.approx(0.421, abs=0.005)
    assert ACCUM3.range_resolution() == pytest.approx(0.281, abs=0.005)
    assert APRES.fringe_spacing() == pytest.approx(0.281, abs=0.005)
    assert ACCUM3.fringe_spacing() == pytest.approx(0.112, abs=0.005)

    # accum3 reads a given displacement with 2.5x the phase, and wraps 2.5x as
    # often doing it.
    assert APRES.fringe_spacing() / ACCUM3.fringe_spacing() == pytest.approx(2.5, abs=0.01)

    # The grid has to resolve the TOP of the band, not the centre.
    dx = APRES.grid_step(EPS_ICE_MEAN, 12.0)
    assert dx == pytest.approx(C0 / (400e6 * np.sqrt(EPS_ICE_MEAN) * 12.0))


def test_chirp_puts_its_energy_in_the_band_and_nowhere_else():
    dt = 2e-11
    t = np.arange(0.0, 400e-9, dt)
    tpd = 100e-9
    p = lfm_chirp(APRES, t, tpd)

    assert np.max(np.abs(p)) == pytest.approx(1.0)
    assert np.all(p[t > tpd] == 0.0)

    spec = np.abs(np.fft.rfft(p, 1 << 16))
    freq = np.fft.rfftfreq(1 << 16, dt)
    inside = (freq >= APRES.f_start) & (freq <= APRES.f_stop)
    # Almost all of the energy is inside the band, and there is no DC.
    assert np.sum(spec[inside] ** 2) / np.sum(spec**2) > 0.95
    assert spec[0] / spec.max() < 1e-3


def test_a_rectangular_chirp_leaks_where_a_tapered_one_does_not():
    """The taper is load-bearing, not cosmetic: it keeps the source in band.

    Energy above ``f_stop`` is energy the grid was not sized to carry, and it
    disperses onto the rest of the pulse rather than simply being lost.
    """
    dt = 2e-11
    t = np.arange(0.0, 400e-9, dt)
    freq = np.fft.rfftfreq(1 << 16, dt)
    out = freq > APRES.f_stop * 1.1

    def leak(taper):
        spec = np.abs(np.fft.rfft(lfm_chirp(APRES, t, 100e-9, taper=taper), 1 << 16))
        return np.sum(spec[out] ** 2) / np.sum(spec**2)

    assert leak(0.25) < 0.2 * leak(0.0)


# ------------------------------------------------------------ compression


def test_compressed_echo_lands_at_its_own_two_way_time():
    """No wavelet offset survives compression, which every prediction assumes.

    The other examples have to add ``envelope_peak_time`` to every predicted
    arrival.  Correlating against a wavelet that starts at t = 0 removes that,
    and examples 5 and 6 depend on it: they invert the compressed time axis
    straight to depth.
    """
    dt = 2e-11
    t = np.arange(0.0, 1.2e-6, dt)
    tx = lfm_chirp(APRES, t, 100e-9)
    for tau in (300e-9, 620e-9, 910e-9):
        prof = range_compress(_shift(tx, dt, tau), tx, dt, APRES)
        peak = t[int(np.argmax(np.abs(prof)))]
        assert peak == pytest.approx(tau, abs=2 * dt)


def test_compressed_phase_is_minus_two_pi_fc_times_the_delay():
    """The whole measurement, in one assertion."""
    dt = 2e-11
    t = np.arange(0.0, 1.2e-6, dt)
    tx = lfm_chirp(APRES, t, 100e-9)
    tau = 600e-9
    prof_1 = range_compress(_shift(tx, dt, tau), tx, dt, APRES)
    k = int(np.argmax(np.abs(prof_1)))

    for centimetres in (1.0, 5.0, 12.0):
        dtau = 2.0 * (centimetres / 100.0) * np.sqrt(EPS_ICE_MEAN) / C0
        prof_2 = range_compress(_shift(tx, dt, tau + dtau), tx, dt, APRES)
        phase = np.angle(interferogram(prof_2, prof_1)[k])
        assert phase == pytest.approx(-2 * np.pi * APRES.fc * dtau, abs=1e-3)


def test_a_windowed_compression_buries_its_far_range_sidelobes():
    """What the band window is for, and what it costs.

    It kills the far skirt, which is what matters when a layer 40 dB down sits
    a few resolution cells from a bright one, as they do in examples 5 and 6:
    twenty cells out Blackman is more than 20 dB below a hard band truncation.
    What it costs is main-lobe width, and therefore resolution - the number
    :meth:`RadarSystem.range_resolution` reports is the Fourier limit, which
    only an unwindowed compression achieves.
    """
    dt = 2e-11
    t = np.arange(0.0, 1.2e-6, dt)
    tx = lfm_chirp(APRES, t, 100e-9)
    rx = _shift(tx, dt, 600e-9)

    def profile(window):
        env = np.abs(range_compress(rx, tx, dt, APRES, window=window))
        return env / env.max(), int(np.argmax(env))

    def skirt(window, cells):
        env, k = profile(window)
        away = np.abs(t - t[k]) > cells / APRES.bandwidth
        return 20 * np.log10(env[away].max())

    def width(window):
        env, _ = profile(window)
        return float(np.count_nonzero(env > 10 ** (-3.0 / 20.0)) * dt)

    assert skirt("blackman", 20) < skirt("rect", 20) - 20.0
    # The price, measured rather than assumed, so that changing the window
    # later is a decision and not a silent loss of resolution.
    assert width("blackman") > 1.4 * width("rect")


def test_deramp_and_matched_filter_measure_the_same_displacement():
    """A real ApRES deramps; accum3 matched-filters; the pair must agree.

    Examples 5 and 6 compress the same way and claim the difference between
    them is the band and nothing else.  That claim is only true if the two
    chains are equivalent, so it is checked rather than asserted.  The sweep
    here is 10 us rather than the instrument's 1 s, which is what lets the same
    record be sampled at RF and put through both chains.
    """
    system = RadarSystem("test", 200e6, 400e6, chirp_duration=10e-6,
                         sample_rate=100e6, fmcw=True)
    dt = 1.0 / 2.0e9
    t = np.arange(0.0, 12e-6, dt)
    tx = lfm_chirp(system, t, system.chirp_duration)

    tau = 0.6e-6
    dtau = 40e-12  # about 6 mm of ice: a fortieth of a fringe
    rx_1 = _shift(tx, dt, tau)
    rx_2 = _shift(tx, dt, tau + dtau)

    tau_ax, beat_1 = beat_profile(deramp(rx_1, t, system), t, system)
    _, beat_2 = beat_profile(deramp(rx_2, t, system), t, system)
    k = int(np.argmax(np.abs(beat_1)))
    assert tau_ax[k] == pytest.approx(tau, rel=1e-3)

    mf_1 = range_compress(rx_1, tx, dt, system)
    mf_2 = range_compress(rx_2, tx, dt, system)
    j = int(np.argmax(np.abs(mf_1)))

    expected = -2 * np.pi * system.fc * dtau
    from_deramp = np.angle(beat_2[k] * np.conj(beat_1[k]))
    from_matched = np.angle(interferogram(mf_2, mf_1)[j])
    assert from_matched == pytest.approx(expected, abs=0.01)
    assert from_deramp == pytest.approx(expected, abs=0.05)


# ------------------------------------------------------------- deformation


def _column():
    return IceColumn(thickness=1200.0, rho_sfc=0.35, rho_bco=0.81, depth_bco=100.0)


def test_burial_and_thinning_pull_opposite_ways_and_cross():
    column = _column()
    strain = VerticalStrain(column, accumulation=0.20, strain_rate=-4.0e-3)

    # At the surface the dynamic term vanishes and burial is b / rho_sfc.
    assert strain.burial_rate(0.0) == pytest.approx(0.20 / 0.35, rel=1e-6)

    depth = np.linspace(0.0, 200.0, 401)
    disp = strain.displacement(depth, 1.0)
    assert disp[0] > 0.0                       # shallow markers sink
    assert disp[-1] < 0.0                      # deep ones rise
    assert np.sum(np.diff(np.sign(disp)) != 0) == 1   # exactly one crossing

    # Markers never pass through one another; a deformation that overturned
    # the stratigraphy would be a modelling error, not a result.
    assert np.all(np.diff(strain.displaced_depth(depth, 1.0)) > 0.0)


def test_no_strain_leaves_pure_firn_burial():
    column = _column()
    strain = VerticalStrain(column, accumulation=0.20, strain_rate=0.0)
    depth = np.array([10.0, 50.0, 150.0])
    # dz/dt = b / rho, so a marker in solid ice sinks at b and one in 0.4
    # relative-density firn at 2.5 b.  Integrated, not Euler-stepped, so the
    # check is against the trajectory rather than the instantaneous rate.
    assert np.all(strain.displacement(depth, 1.0) > 0.0)
    assert strain.displacement(10.0, 1.0) > strain.displacement(150.0, 1.0)

    # Ten one-year steps must land where one ten-year step does.
    z = 20.0
    stepwise = z
    for _ in range(10):
        stepwise = float(strain.displaced_depth(stepwise, 1.0))
    assert stepwise == pytest.approx(float(strain.displaced_depth(z, 10.0)), rel=1e-4)


def test_delay_and_phase_carry_the_sign_the_interferogram_expects():
    """Thinning must not come back as thickening.

    ``interferogram(epoch2, epoch1)`` has phase ``2 pi fc (t_1 - t_2)``.  A
    marker that sank arrives later at epoch 2, so its delay is positive and its
    phase negative; one that rose is the other way round.  This ties
    :meth:`VerticalStrain.phase` to that convention so the examples cannot
    quietly flip it.
    """
    column = _column()
    strain = VerticalStrain(column, accumulation=0.20, strain_rate=-4.0e-3)

    shallow, deep = 10.0, 200.0
    assert strain.displacement(shallow, 1.0) > 0
    assert strain.delay(shallow, 1.0) > 0
    assert strain.phase(shallow, 1.0, APRES.fc, wrap=False) < 0

    assert strain.displacement(deep, 1.0) < 0
    assert strain.delay(deep, 1.0) < 0
    assert strain.phase(deep, 1.0, APRES.fc, wrap=False) > 0

    # And the delay is the geometric one: the medium does not change, so it is
    # the displacement divided by the local wave speed, two ways.
    eps = column.permittivity(shallow)
    n_local = np.sqrt(0.5 * (eps[0] + eps[1]))
    expected = 2.0 * strain.displacement(shallow, 1.0) * n_local / C0
    assert strain.delay(shallow, 1.0) == pytest.approx(expected, rel=0.02)


def test_accum3_winds_two_and_a_half_times_the_phase_of_apres():
    column = _column()
    strain = VerticalStrain(column, accumulation=0.20, strain_rate=-4.0e-3)
    depth = np.linspace(2.0, 60.0, 200)
    turns = {
        s.name: np.ptp(strain.phase(depth, 1.0, s.fc, wrap=False)) / (2 * np.pi)
        for s in (APRES, ACCUM3)
    }
    assert turns["accum3"] / turns["ApRES"] == pytest.approx(2.5, rel=0.01)


# ------------------------------------------------------------- sub-cell layers


def _layer_centroid(edge_width, depth, dx=0.035):
    """Density-weighted centroid of one layer's anomaly on a grid."""
    grid = PropertyGrid.uniform((-0.5, 0.5), (-1.0, 12.0), dx)
    column = _column()
    layer = Layer(depth=depth, thickness=0.7, d_rho=0.02, edge_width=edge_width)
    builder = IceModelBuilder(grid, column, surface=0.0, air=True)
    plain = builder.rho.copy()
    builder.add_layer(layer)
    anomaly = (builder.rho - plain)[grid.nx // 2]
    z = grid.z
    return float(np.sum(anomaly * z) / np.sum(anomaly))


def test_zero_edge_width_is_exactly_the_old_hard_mask():
    """The default must not perturb examples 1, 3 and 4 by a single node."""
    grid = PropertyGrid.uniform((-1.0, 1.0), (-1.0, 12.0), 0.05)
    column = _column()
    layer = Layer(depth=6.0, thickness=0.7, d_rho=0.02, sigma_factor=3.0)

    builder = IceModelBuilder(grid, column, surface=0.0, air=True)
    mask = builder.layer_mask(layer) & ~builder.is_air
    expected_rho = builder.rho.copy()
    expected_rho[mask] += 0.02
    expected_sig = np.ones_like(builder.rho)
    expected_sig[mask] = 3.0

    builder.add_layer(layer)
    assert np.array_equal(builder.rho, expected_rho)
    assert np.array_equal(builder.sigma_factor, expected_sig)


def test_a_ramped_layer_tracks_a_sub_cell_displacement():
    """The reason ``edge_width`` exists.

    A phase-sensitive example reads a displacement of a few centimetres off a
    3.5 cm grid.  With hard edges the layer can only sit on a node, so a
    sub-cell move quantises - to zero, or to a whole cell, which at 300 MHz is
    most of a radian either way.  Ramped edges let the sampled profile's
    centroid follow the request.
    """
    dx = 0.035
    base = 6.0
    offsets = np.array([0.2, 0.4, 0.6, 0.8]) * dx

    ramped = np.array([_layer_centroid(2.5 * dx, base + d, dx) for d in offsets])
    ramped -= _layer_centroid(2.5 * dx, base, dx)
    hard = np.array([_layer_centroid(0.0, base + d, dx) for d in offsets])
    hard -= _layer_centroid(0.0, base, dx)

    # The ramped centroid follows to a small fraction of a cell...
    assert np.max(np.abs(ramped - offsets)) < 0.15 * dx
    # ...where the hard-edged one is quantised, and wrong by a good part of a
    # cell somewhere in that sweep.
    assert np.max(np.abs(hard - offsets)) > 0.35 * dx


def test_a_ramped_edge_costs_reflectivity_but_not_much():
    """Stated because it is a real cost the examples pay knowingly.

    The ramp is a fraction of a wavelength, so the anomaly it builds is a
    little weaker and a little wider than the stepped one.  What must not
    happen is the ramp eating the layer: the integrated anomaly is conserved to
    within a few percent, because the raised cosine is symmetric about the
    nominal edge.
    """
    dx = 0.035
    grid = PropertyGrid.uniform((-0.5, 0.5), (-1.0, 12.0), dx)
    column = _column()

    def integrated(edge_width):
        builder = IceModelBuilder(grid, column, surface=0.0, air=True)
        plain = builder.rho.copy()
        builder.add_layer(Layer(depth=6.0, thickness=0.7, d_rho=0.02,
                                edge_width=edge_width))
        return float(np.sum((builder.rho - plain)[grid.nx // 2]) * dx)

    assert integrated(2.5 * dx) == pytest.approx(integrated(0.0), rel=0.06)


# ------------------------------------------------------- the recovery chain


def _examples_on_path():
    """Put the examples on ``sys.path``: they are a script directory, not a package."""
    import sys
    from pathlib import Path

    path = str(Path(__file__).resolve().parents[1] / "examples")
    if path not in sys.path:
        sys.path.insert(0, path)


def test_upsampled_lag_is_unbiased_on_an_envelope_where_parabolic_is_not():
    """Why example 5 does not use :func:`subsample_lag` for its coarse step.

    A range-compressed envelope is one-signed, so the mean removal inside
    ``subsample_lag`` digs a pedestal either side of the correlation peak and
    the fitted parabola's vertex is pulled towards zero lag.  Interpolating the
    correlation instead has no such bias.  Left in, it costs several
    centimetres of range - a third of an ApRES fringe and most of an accum3
    one - and would silently pick the wrong fringe.
    """
    dt = 2e-11
    t = np.arange(0.0, 1.2e-6, dt)
    tx = lfm_chirp(APRES, t, 100e-9)
    tau = 600e-9

    prof_1 = range_compress(_shift(tx, dt, tau), tx, dt, APRES)
    cell = int(round(1.0 / (APRES.bandwidth * dt)))
    k = int(np.argmax(np.abs(prof_1)))
    win = slice(k - 4 * cell, k + 4 * cell + 1)

    errors_up, errors_para = [], []
    for shift in (0.4e-9, 1.0e-9, 2.0e-9, 3.0e-9):
        prof_2 = range_compress(_shift(tx, dt, tau + shift), tx, dt, APRES)
        a, b = np.abs(prof_1[win]), np.abs(prof_2[win])
        errors_up.append(upsampled_lag(a, b, dt) - shift)
        errors_para.append(subsample_lag(a, b, dt) - shift)

    bias_up = abs(float(np.mean(errors_up)))
    bias_para = abs(float(np.mean(errors_para)))
    assert bias_up < 0.06e-9          # a millimetre of range
    assert bias_para > 4 * bias_up


def test_the_recovery_chain_returns_the_deformation_it_was_given():
    """End to end, without the FDTD: synthesise, process, and get the truth back.

    Each layer is a copy of the transmit chirp at its own two-way time, and the
    epoch-2 record is the same layers at their deformed depths.  That is the
    exact thing the FDTD approximates, so a failure here is a failure of the
    processing rather than of the wave solver - which is what makes it worth
    testing separately from a run that takes an hour.
    """
    _examples_on_path()
    import ex05_apres_repeat as ex05

    system = APRES
    column = ex05.ice_column()
    strain = ex05.strain_model(column)
    layers = ex05.layering(70.0)

    dt = 2e-11
    depths = np.array([layer.depth for layer in layers])
    moved = strain.displaced_depth(depths, ex05.EPOCH_YEARS)
    t_end = float(column.two_way_time(90.0, z0=ex05.Z_ANT)) + 2 * ex05.TPD
    t = np.arange(0.0, t_end, dt)
    tx = lfm_chirp(system, t, ex05.TPD)

    def record(zs):
        out = np.zeros_like(t)
        for z in zs:
            out += _shift(tx, dt, float(column.two_way_time(z, z0=ex05.Z_ANT)))
        return out

    prof_1 = ex05.compressed(record(depths), tx, dt, system)
    prof_2 = ex05.compressed(record(moved), tx, dt, system)

    ref = float(np.max(np.abs(ex05.compressed(tx, tx, dt, system))))
    env = 20 * np.log10(np.maximum(np.abs(prof_1), 1e-30) / ref)
    depth_axis = column.depth_from_two_way_time(t, z0=ex05.Z_ANT)
    usable = (depth_axis > depths[0] - 2.0) & (depth_axis < depths[-1] + 2.0)

    peaks = ex05.find_echoes(env, usable, dt, system)
    # One echo per layer: the prominence rule has to reject the compressed
    # wavelet's own sidelobes, or each of them arrives carrying its parent's
    # phase and the profile fills with layers that are not there.
    assert len(peaks) == len(depths)

    motion = ex05.apparent_motion(prof_1, prof_2, t, dt, column, system, peaks)
    truth = strain.displacement(motion["depth"], ex05.EPOCH_YEARS)

    # The motion here spans 20-50 cm, i.e. one to two whole ApRES fringes, so
    # this exercises the ambiguity resolution and not just the phase.
    assert np.ptp(truth) > system.fringe_spacing()
    assert np.max(np.abs(motion["displacement"] - truth)) < 0.02


def test_a_record_cache_is_refused_when_the_model_moved(tmp_path):
    """Examples 5 and 6 cache traces, not a wavefield, and want the same guard."""
    _examples_on_path()
    import _common

    cache = _common.save_arrays(tmp_path / "records.npz",
                                stamp={"years": 1.0, "strain": {"strain_rate": -4e-3}},
                                t_rec=np.arange(4.0), rec_1=np.ones(4))
    back = _common.load_arrays(cache, stamp={"years": 1.0,
                                             "strain": {"strain_rate": -4e-3}})
    assert sorted(back) == ["rec_1", "t_rec"]

    with pytest.raises(_common.StaleCache, match="strain.strain_rate"):
        _common.load_arrays(cache, stamp={"years": 1.0,
                                          "strain": {"strain_rate": -2e-3}})


def test_trajectory_agrees_with_integrating_straight_to_each_time():
    """A sampled path and a direct integration must be the same path.

    :meth:`VerticalStrain.trajectory` integrates once and samples as it goes,
    which is what makes a five-epoch series one trajectory rather than five.
    If it disagreed with :meth:`displaced_depth` the figures would compare a
    measurement against a slightly different truth than the one the scene was
    built from.
    """
    strain = VerticalStrain(_column(), accumulation=0.20, strain_rate=-4.0e-3)
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    for depth in (5.0, 40.0, 120.0):
        sampled = strain.trajectory(depth, times)
        direct = np.array([strain.displacement(depth, float(y)) for y in times])
        assert np.allclose(sampled, direct, atol=1e-4)

    # Order must not matter, and going backwards is a caller error rather than
    # a silently different answer.
    shuffled = np.array([3.0, 0.0, 4.0, 1.0, 2.0])
    assert np.allclose(strain.trajectory(40.0, shuffled),
                       strain.trajectory(40.0, times)[[3, 0, 4, 1, 2]], atol=1e-9)
    with pytest.raises(ValueError, match="non-negative"):
        strain.trajectory(40.0, np.array([0.0, -1.0]))


def test_markers_are_chained_through_the_series_and_dropped_when_lost():
    """Tracking has to follow reflectors, not range bins.

    Each step measures the echoes in its own first record, so the lists are not
    the same reflectors in the same order.  A marker that cannot be matched must
    be dropped: chaining onto the wrong reflector does not look like an error
    afterwards, it looks like a layer that moved several metres in a year.
    """
    _examples_on_path()
    import ex05_apres_repeat as ex05

    def step(depths, moves):
        return {"depth": np.asarray(depths, dtype=float),
                "displacement": np.asarray(moves, dtype=float),
                "dtau": np.asarray(moves, dtype=float) * 1e-9}

    # Three markers at 10, 20 and 30 m, each moving 1 m per step.  The middle
    # one has no echo in the second step's list, so it must not be chained onto
    # its neighbour.
    steps = [
        step([10.0, 20.0, 30.0], [1.0, 1.0, 1.0]),
        step([11.0, 31.0], [1.0, 1.0]),
    ]
    markers = ex05.track_markers(steps, tolerance=0.5)
    assert sorted(round(m["depth0"]) for m in markers) == [10, 30]
    followed = next(m for m in markers if m["depth0"] == 10.0)
    assert followed["displacement"] == pytest.approx([0.0, 1.0, 2.0])

    # And with every echo present, every marker survives with the right total.
    steps = [step([10.0, 20.0], [1.0, 2.0]), step([11.0, 22.0], [1.0, 2.0])]
    markers = ex05.track_markers(steps, tolerance=0.5)
    assert len(markers) == 2
    assert [m["displacement"][-1] for m in markers] == pytest.approx([2.0, 4.0])
