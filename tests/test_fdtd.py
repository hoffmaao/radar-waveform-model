"""Validation of the FDTD engine against analytic results."""

import numpy as np
import pytest

from radarwave import (
    C0,
    DEPS_ICE,
    EPS_ICE_MEAN,
    FDTD2D,
    PropertyGrid,
    blackharrispulse,
    max_time_step,
    run_common_offset,
)
from radarwave.polarimetry import subsample_lag

EPS_ICE = 3.17
V_ICE = C0 / np.sqrt(EPS_ICE)
FC = 100e6


def homogeneous(dx=0.25, eps=EPS_ICE, xlim=(-30, 30), zlim=(0, 30), npml=10, **kw):
    grid = PropertyGrid.uniform(xlim, zlim, dx, eps=eps, **kw).padded(npml=npml)
    dt = 0.8 * max_time_step(grid.eps_min, grid.mu_min, grid.dx, grid.dz)
    return grid, dt


def _vertical_speed(mode, dx=0.25, eps=EPS_ICE):
    """Group speed of a downgoing wave between two depths.

    Propagation is vertical because that is both the ice-sounding geometry and
    the direction in which an ``Ex``-polarised TE source radiates strongly; a
    horizontal profile would sit in the source dipole's null.
    """
    grid, dt = homogeneous(dx=dx, eps=eps, xlim=(-20, 20), zlim=(0, 40))
    t = np.arange(0, 500e-9, dt)
    sim = FDTD2D(grid, dt, npml=10, mode=mode)
    kw = {} if mode == "TM" else dict(source_component="Ex", receiver_component="Ex")
    res = sim.run(
        np.array([[0.0, 4.0]]),
        blackharrispulse(FC, t),
        np.array([[0.0, 12.0], [0.0, 34.0]]),
        **kw,
    )
    delay = subsample_lag(res.gather[:, 0, 0], res.gather[:, 1, 0], dt)
    return (res.rec[1, 1] - res.rec[0, 1]) / delay


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_propagation_speed(mode):
    """Group velocity between two depths matches c / sqrt(eps) to <1%."""
    assert abs(_vertical_speed(mode) / V_ICE - 1.0) < 0.01


def test_dispersion_improves_with_refinement():
    """Halving dx from a marginally resolved grid reduces the speed error."""
    coarse = abs(_vertical_speed("TM", dx=0.5) / V_ICE - 1.0)
    fine = abs(_vertical_speed("TM", dx=0.25) / V_ICE - 1.0)
    assert coarse > 0.02  # 3.4 points per wavelength is visibly dispersive
    assert fine < coarse / 5


def boundary_reflection(mode="TM", npml=10, tight=12.0, roomy=60.0, rec=(0.0, 10.0), **pml):
    """Boundary error, measured by shrinking the domain around a fixed geometry.

    The difference between a tight and a roomy domain is entirely the boundary's
    doing.  Measuring the late-time amplitude instead would mostly capture the
    physical wake of a two-dimensional line source, which decays only as
    t**-0.5 and would mask the boundary completely.
    """
    dt = 0.8 * max_time_step(EPS_ICE, 1.0, 0.25, 0.25)
    t = np.arange(0, 700e-9, dt)
    pulse = blackharrispulse(FC, t)
    src = np.array([[0.0, 2.0]])
    rec = np.array([list(rec)])
    kw = {} if mode == "TM" else dict(source_component="Ex", receiver_component="Ex")

    traces = []
    for zbot in (tight, roomy):
        grid = PropertyGrid.uniform((-20, 20), (0, zbot), 0.25, eps=EPS_ICE).padded(npml=npml)
        sim = FDTD2D(grid, dt, npml=npml, mode=mode, **pml)
        traces.append(sim.run(src, pulse, rec, **kw).gather[:, 0, 0])
    a, b = traces
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("mode", ["TM", "TE"])
def test_pml_absorption(mode):
    """The absorbing boundary is better than -40 dB at normal incidence."""
    err = boundary_reflection(mode=mode)
    assert err < 1e-2, f"boundary reflection {20 * np.log10(err):.1f} dB"


def test_pml_improves_with_thickness():
    """Doubling the layer must reduce the reflection.

    It did not in the inherited configuration: real coordinate stretching with
    kappa_max = 5 divided the damping conductivity by up to 5.9, pinning the
    reflection near -23 dB no matter how many cells were added.
    """
    thin = boundary_reflection(npml=10)
    thick = boundary_reflection(npml=20)
    assert thick < thin / 2


def test_pml_stretching_regression():
    """The shipped default beats the kappa_max = 5 configuration it replaced."""
    tuned = boundary_reflection(npml=10)
    stretched = boundary_reflection(npml=10, pml_kmax=5.0, pml_sigma_scale=1.0)
    assert 20 * np.log10(stretched / tuned) > 15  # at least 15 dB better


def _planar_reflection(eps_bot, eps_top=EPS_ICE, dx=0.1, z_interface=14.0, z_src=2.0):
    """Peak amplitude of the normal-incidence reflection off a planar contrast."""
    zlim = (0.0, 34.0)
    grid = PropertyGrid.uniform((-24, 24), zlim, dx, eps=eps_top)
    _, Z = grid.mesh()
    grid = grid.with_properties(eps=np.where(Z >= z_interface, eps_bot, eps_top)).padded(npml=10)

    dt = 0.8 * max_time_step(min(eps_top, eps_bot), 1.0, grid.dx, grid.dz)
    t = np.arange(0, 700e-9, dt)
    sim = FDTD2D(grid, dt, npml=10, mode="TM")
    res = sim.run(np.array([[0.0, z_src]]), blackharrispulse(FC, t), np.array([[0.0, z_src]]))

    # Isolate the reflection from the source-region field by arrival time.
    t_refl = 2 * (z_interface - z_src) / (C0 / np.sqrt(eps_top))
    win = np.abs(res.t - t_refl) < 25e-9
    return np.max(np.abs(res.gather[win, 0, 0])), dt, t


def _fresnel(eps_top, eps_bot):
    a, b = np.sqrt(eps_top), np.sqrt(eps_bot)
    return (a - b) / (a + b)


def test_reflection_amplitude_scales_as_fresnel():
    """Two contrasts reflect in the ratio of their Fresnel coefficients.

    Taking a ratio cancels the source normalisation and the geometric
    spreading that are common to both runs.
    """
    strong, _, _ = _planar_reflection(4.60)
    weak, _, _ = _planar_reflection(3.60)
    expected = abs(_fresnel(EPS_ICE, 4.60) / _fresnel(EPS_ICE, 3.60))
    assert abs(strong / weak / expected - 1.0) < 0.05


def test_reflection_amplitude_absolute():
    """Reflected amplitude equals |r| times the incident field at two-way range."""
    eps_bot, z_interface, z_src = 4.60, 14.0, 2.0
    amp_refl, dt, t = _planar_reflection(eps_bot, z_interface=z_interface, z_src=z_src)

    # Same source in a whole space, recorded at the two-way path length, so the
    # cylindrical spreading matches that of the reflection.
    two_way = 2 * (z_interface - z_src)
    ref_grid = PropertyGrid.uniform((-24, 24), (0.0, 34.0), 0.1, eps=EPS_ICE).padded(npml=10)
    ref = FDTD2D(ref_grid, dt, npml=10, mode="TM").run(
        np.array([[0.0, z_src]]),
        blackharrispulse(FC, t),
        np.array([[0.0, z_src + two_way]]),
    )
    amp_inc = np.max(np.abs(ref.gather[:, 0, 0]))
    assert abs(amp_refl / amp_inc / abs(_fresnel(EPS_ICE, eps_bot)) - 1.0) < 0.12


def test_birefringent_delay_matches_theory():
    """The TM/TE split through anisotropic ice matches 2 d (sx - sy).

    This is the physics behind the polarimetric interferograms: the two
    eigenpolarisations see eps_yy and eps_xx respectively, so a fabric with
    lam_x != lam_y delays one relative to the other.
    """
    dlam = 0.30  # exaggerated so the split is many samples at this scale
    eps_x = EPS_ICE_MEAN + DEPS_ICE * (0.5 * (1 - 1 / 3.0) + dlam / 2 - 1 / 3.0)
    eps_y = EPS_ICE_MEAN + DEPS_ICE * (0.5 * (1 - 1 / 3.0) - dlam / 2 - 1 / 3.0)

    dx = 0.15
    grid = PropertyGrid.uniform(
        (-40, 40), (0, 130), dx, eps={"xx": eps_x, "yy": eps_y, "zz": eps_x}
    ).padded(npml=10)
    dt = 0.8 * max_time_step(grid.eps_min, grid.mu_min, grid.dx, grid.dz)
    t = np.arange(0, 1600e-9, dt)
    pulse = blackharrispulse(FC, t)

    src = np.array([[0.0, 2.0]])
    rec = np.array([[0.0, 122.0]])
    tm = FDTD2D(grid, dt, npml=10, mode="TM").run(src, pulse, rec)
    te = FDTD2D(grid, dt, npml=10, mode="TE").run(
        src, pulse, rec, source_component="Ex", receiver_component="Ex"
    )

    # subsample_lag(a, b) is positive when b lags a; TE sees the larger
    # permittivity here, so it must arrive after TM.
    measured = subsample_lag(tm.gather[:, 0, 0], te.gather[:, 0, 0], dt)
    path = rec[0, 1] - src[0, 1]
    expected = path * (np.sqrt(eps_x) - np.sqrt(eps_y)) / C0
    assert expected > 0
    assert abs(measured - expected) < 0.08 * abs(expected)


def test_anisotropy_only_affects_the_matching_component():
    """TM responds to eps_yy and is blind to eps_xx (and vice versa)."""
    dx = 0.25

    def travel(eps_dict, mode):
        grid = PropertyGrid.uniform((-20, 20), (0, 40), dx, eps=eps_dict).padded(npml=10)
        dt = 0.8 * max_time_step(3.0, 1.0, grid.dx, grid.dz)
        t = np.arange(0, 500e-9, dt)
        sim = FDTD2D(grid, dt, npml=10, mode=mode)
        res = sim.run(
            np.array([[0.0, 4.0]]),
            blackharrispulse(FC, t),
            np.array([[0.0, 12.0], [0.0, 34.0]]),
            **({} if mode == "TM" else dict(source_component="Ex", receiver_component="Ex")),
        )
        return subsample_lag(res.gather[:, 0, 0], res.gather[:, 1, 0], dt)

    ref = {"xx": 3.17, "yy": 3.17, "zz": 3.17}
    bumped_y = {"xx": 3.17, "yy": 3.60, "zz": 3.17}
    bumped_x = {"xx": 3.60, "yy": 3.17, "zz": 3.17}

    t_ref_tm = travel(ref, "TM")
    assert travel(bumped_y, "TM") > 1.03 * t_ref_tm  # TM slows down
    assert abs(travel(bumped_x, "TM") - t_ref_tm) < 1e-3 * t_ref_tm  # TM unaffected

    t_ref_te = travel(ref, "TE")
    assert travel(bumped_x, "TE") > 1.03 * t_ref_te
    assert abs(travel(bumped_y, "TE") - t_ref_te) < 1e-3 * t_ref_te


def test_rejects_unstable_time_step():
    grid, dt = homogeneous()
    with pytest.raises(ValueError, match="stability limit"):
        FDTD2D(grid, dt * 2.0, npml=10)


def test_gather_and_time_indexing():
    """Output times and per-shot columns line up (the old port was off by one)."""
    grid, dt = homogeneous(xlim=(-10, 10), zlim=(0, 10))
    t = np.arange(0, 200e-9, dt)
    sim = FDTD2D(grid, dt, npml=10, mode="TM")

    # Ey nodes sit on the half-grid, so 0 is not generally a node.  Anchor the
    # receiver on a real node and place the two shots symmetrically about it,
    # otherwise the snapped geometry is asymmetric and the traces differ.
    x_rec = sim.x_half[np.argmin(np.abs(sim.x_half))]
    src = np.array([[x_rec - 4.0, 4.0], [x_rec + 4.0, 4.0]])
    rec = np.array([[x_rec, 4.0]])
    outstep = 3
    res = sim.run(src, blackharrispulse(FC, t), rec, outstep=outstep)
    np.testing.assert_allclose(res.src[:, 0] - x_rec, [-4.0, 4.0], atol=1e-9)

    assert res.gather.shape == ((len(t) - 1) // outstep + 1, 1, 2)
    assert res.t[0] == 0.0
    np.testing.assert_allclose(np.diff(res.t), outstep * dt)
    assert np.max(np.abs(res.gather)) > 0

    # Shots must not leak into each other: running the second one on its own
    # has to reproduce its column bit for bit.
    solo = sim.run(src[1:], blackharrispulse(FC, t), rec, outstep=outstep)
    np.testing.assert_array_equal(res.gather[:, 0, 1], solo.gather[:, 0, 0])

    # The two shots are mirror images about the receiver, so their traces agree
    # up to the small asymmetry of the absorbing boundary around that point.
    a, b = res.gather[:, 0, 0], res.gather[:, 0, 1]
    assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 0.02


def test_common_offset_returns_one_trace_per_shot():
    """Shot ``k`` is paired with receiver ``k``, serially and in parallel.

    The section is what every example plots, and a gather that quietly hands
    back receiver 0 for every shot is a common-*receiver* record instead: the
    same picture, with offsets growing to the length of the profile.
    """
    grid, dt = homogeneous(xlim=(-14, 14), zlim=(0, 10))
    t = np.arange(0, 200e-9, dt)
    pulse = blackharrispulse(FC, t)
    xs = np.array([-6.0, 0.0, 6.0])
    shots = np.column_stack([xs, np.full_like(xs, 2.0)])
    recs = np.column_stack([xs + 1.0, np.full_like(xs, 2.0)])

    co = run_common_offset(grid, dt, shots, recs, pulse, npml=10, mode="TM")
    assert co.gather.shape == (len(t), 1, len(xs))
    assert co.common_offset.shape == (len(t), len(xs))

    # Every column has to be that shot's own receiver, not shot 0's.
    sim = FDTD2D(grid, dt, npml=10, mode="TM")
    full = sim.run(shots, pulse, recs)
    for k in range(len(xs)):
        np.testing.assert_array_equal(co.common_offset[:, k], full.gather[:, k, k])

    par = run_common_offset(grid, dt, shots, recs, pulse, npml=10, mode="TM", processes=2)
    assert par.gather.shape == co.gather.shape
    np.testing.assert_allclose(par.common_offset, co.common_offset, rtol=0, atol=0)
    np.testing.assert_allclose(par.src, co.src)
    np.testing.assert_allclose(par.rec, co.rec)


def test_common_offset_needs_a_receiver_per_shot():
    grid, dt = homogeneous(xlim=(-14, 14), zlim=(0, 10))
    t = np.arange(0, 60e-9, dt)
    shots = np.array([[-6.0, 2.0], [6.0, 2.0]])
    with pytest.raises(ValueError, match="one receiver per shot"):
        run_common_offset(grid, dt, shots, shots[:1], blackharrispulse(FC, t))


def test_birefringent_delay_converges_with_refinement():
    """The measured split approaches the traveltime model as the grid refines.

    The delay is a small fraction of a time step, so it is sensitive to
    numerical dispersion -- which differs slightly between the two runs because
    their wave speeds differ.  Refining must shrink that difference.
    """
    from radarwave import IceColumn
    from radarwave.scenes import IceModelBuilder

    column = IceColumn(
        thickness=1850.0, depth_bco=100.0,
        lam_z_sfc=0.30, lam_z_bed=0.30,
        dlambda=lambda d: np.full_like(np.asarray(d, float), 0.60),
    )
    z_src, z_rec = 3.0, 120.0
    expected = 0.5 * (column.birefringent_delay(z_rec) - column.birefringent_delay(z_src))

    errors = []
    for dx in (0.5, 0.25):
        grid = PropertyGrid.uniform((-30, 30), (-6, 140), dx)
        model = IceModelBuilder(grid, column, surface=0.0, air=True).finalize(npml=12)
        dt = 0.9 * max_time_step(model.eps_min, model.mu_min, model.dx, model.dz)
        t = np.arange(0, 1.4e-6, dt)
        pulse = blackharrispulse(60e6, t)
        src = np.array([[0.0, z_src]])
        rec = np.array([[0.0, z_rec]])

        slow_axis = model.with_properties(
            eps={"xx": model.eps["xx"], "yy": model.eps["xx"], "zz": model.eps["zz"]}
        )
        a = FDTD2D(slow_axis, dt, npml=12, mode="TM").run(src, pulse, rec).gather[:, 0, 0]
        b = FDTD2D(model, dt, npml=12, mode="TM").run(src, pulse, rec).gather[:, 0, 0]
        errors.append(abs(subsample_lag(b, a, dt) / expected - 1.0))

    assert errors[0] > 0.10  # 5.6 nodes per wavelength is clearly under-resolved
    assert errors[1] < 0.02  # 11 nodes per wavelength is within a couple of percent
    assert errors[1] < errors[0] / 5
