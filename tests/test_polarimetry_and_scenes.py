"""Polarimetric processing and 2-D scene construction."""

import numpy as np
import pytest

from radarwave import C0, IceColumn, PropertyGrid, RIDGE_A, gabor, ridge_a_dlambda
from radarwave.polarimetry import (
    delay_from_phase,
    delayed_copy,
    dlambda_from_delay,
    interferogram,
    unwrap_phase,
    volume_scattering,
)
from radarwave.scenes import IceModelBuilder, Layer, dipping_depth, undulating_depth

FC = 195e6


def _synthetic(column, n_traces=40, seed=3):
    """A pair of eigenpolarised sections through a fabric column."""
    zz = np.arange(2.0, column.thickness - 20.0, 3.0)
    eps = column.permittivity(zz)
    slow = 0.5 * (np.sqrt(eps[:, 0]) + np.sqrt(eps[:, 1])) / C0
    t0 = 2.0 * np.concatenate(
        [[zz[0] * slow[0]], zz[0] * slow[0] + np.cumsum(np.diff(zz) * slow[:-1])]
    )
    dtau = column.birefringent_delay(zz)

    dt = 1.0 / (8.0 * FC)
    t = np.arange(0.0, t0[-1] + 60e-9, dt)
    wavelet = gabor(FC, np.arange(0, 14.0 / FC, dt), bandwidth=0.16)
    dtau_t = np.interp(t, t0, dtau, left=0.0, right=dtau[-1])

    rng = np.random.default_rng(seed)
    ig = np.zeros((t.size, n_traces), dtype=complex)
    for j in range(n_traces):
        hh = volume_scattering(t, rng, wavelet, decay_time=14e-6)
        ig[:, j] = interferogram(hh, delayed_copy(t, hh, dtau_t))
    return t, t0, zz, dtau, dtau_t, ig


def test_delayed_copy_is_a_pure_delay():
    t = np.linspace(0, 1e-6, 4001)
    x = np.sin(2 * np.pi * 50e6 * t)
    shift = 3e-9
    y = delayed_copy(t, x, np.full_like(t, shift))
    expected = np.sin(2 * np.pi * 50e6 * (t - shift))
    inner = slice(200, -200)
    np.testing.assert_allclose(y[inner], expected[inner], atol=2e-3)


def test_interferogram_phase_equals_two_pi_f_delay():
    """The core relation the whole polarimetric method rests on."""
    dt = 1.0 / (16 * FC)
    t = np.arange(0, 4e-6, dt)
    rng = np.random.default_rng(0)
    wavelet = gabor(FC, np.arange(0, 14.0 / FC, dt), bandwidth=0.15)
    hh = volume_scattering(t, rng, wavelet)
    for delay in (0.5e-9, 2e-9):
        vv = delayed_copy(t, hh, np.full_like(t, delay))
        ig = interferogram(hh, vv)
        inner = slice(500, -500)
        phase = np.angle(ig[inner]).mean()
        assert phase == pytest.approx(2 * np.pi * FC * delay, rel=0.1)


def test_delay_from_phase_roundtrip():
    delay = np.array([0.0, 1e-9, 4e-9])
    phase = 2 * np.pi * FC * delay
    np.testing.assert_allclose(delay_from_phase(phase, FC), delay)


def test_full_chain_recovers_ridge_a_dlambda():
    """Synthesise fringes from a fabric profile, process them, get it back."""
    from scipy.ndimage import uniform_filter1d

    column = IceColumn(**RIDGE_A)
    t, t0, zz, dtau, dtau_t, ig = _synthetic(column, n_traces=60)

    depth = np.interp(t, t0, zz, left=0.0, right=zz[-1])
    keep = (depth > 200.0) & (depth < column.thickness - 100.0)

    st = ig[keep].mean(axis=1)
    st = uniform_filter1d(st.real, 101) + 1j * uniform_filter1d(st.imag, 101)
    phi = unwrap_phase(np.angle(st))
    z_valid = depth[keep]
    tau = delay_from_phase(phi - phi[0], FC)
    dlam = dlambda_from_delay(z_valid, tau, column=column, smooth=401)

    truth = ridge_a_dlambda(z_valid)
    inner = slice(1000, -1000)  # away from the smoothing edge effects
    assert np.median(np.abs(dlam[inner] - truth[inner])) < 0.008
    assert np.corrcoef(dlam[inner], truth[inner])[0, 1] > 0.8


def test_fringe_count_matches_the_traveltime_model():
    column = IceColumn(**RIDGE_A)
    t, t0, zz, dtau, dtau_t, ig = _synthetic(column, n_traces=60)
    from scipy.ndimage import uniform_filter1d

    st = ig.mean(axis=1)
    st = uniform_filter1d(st.real, 101) + 1j * uniform_filter1d(st.imag, 101)
    phi = unwrap_phase(np.angle(st))
    span = phi[-1] - phi[0]
    assert span / (2 * np.pi * FC) == pytest.approx(dtau[-1], rel=0.1)


# -- scenes -------------------------------------------------------------


def test_dipping_depth_geometry():
    x = np.array([-100.0, 0.0, 100.0])
    z = dipping_depth(x, 175.0, 35.0)
    assert z[1] == pytest.approx(175.0)
    assert z[2] - z[1] == pytest.approx(100 * np.tan(np.deg2rad(35.0)))
    assert z[0] < z[1] < z[2]


def test_undulating_depth_sums_components():
    x = np.linspace(-200, 200, 401)
    z = undulating_depth(x, 100.0, [(5.0, 200.0, 0.0)], dip_deg=0.0)
    assert np.max(z) == pytest.approx(105.0, abs=0.1)
    assert np.min(z) == pytest.approx(95.0, abs=0.1)


def test_builder_air_and_ice():
    column = IceColumn()
    grid = PropertyGrid.uniform((-10, 10), (-5, 40), 0.5)
    model = IceModelBuilder(grid, column, surface=0.0, air=True).finalize()
    _, Z = grid.mesh()
    assert np.allclose(model.eps["yy"][Z < 0], 1.0)
    assert np.all(model.sig["yy"][Z < 0] == 0.0)
    assert np.all(model.eps["yy"][Z > 1] > 1.3)
    model.validate()


def test_layer_changes_only_its_own_depth_range():
    column = IceColumn()
    grid = PropertyGrid.uniform((-10, 10), (0, 60), 0.5)
    plain = IceModelBuilder(grid, column, air=False).finalize()
    layered = (
        IceModelBuilder(grid, column, air=False)
        .add_layer(Layer(depth=30.0, thickness=2.0, d_rho=0.05))
        .finalize()
    )
    diff = np.abs(layered.eps["yy"] - plain.eps["yy"])
    _, Z = grid.mesh()
    inside = np.abs(Z - 30.0) <= 1.0
    assert np.max(diff[inside]) > 1e-3
    assert np.max(diff[~inside]) == 0.0


def test_conductivity_layer_leaves_permittivity_alone():
    column = IceColumn()
    grid = PropertyGrid.uniform((-10, 10), (0, 60), 0.5)
    plain = IceModelBuilder(grid, column, air=False).finalize()
    acid = (
        IceModelBuilder(grid, column, air=False)
        .add_layer(Layer(depth=30.0, thickness=1.0, sigma_factor=4.0))
        .finalize()
    )
    np.testing.assert_allclose(acid.eps["yy"], plain.eps["yy"])
    # Compare at the layer, not globally: conductivity also rises with density,
    # so the deepest cell is the maximum in both models.
    j = int(np.argmin(np.abs(grid.z - 30.0)))
    assert acid.sig["yy"][5, j] == pytest.approx(4.0 * plain.sig["yy"][5, j], rel=1e-9)


def test_set_fabric_keeps_eigenvalues_normalised():
    column = IceColumn()
    grid = PropertyGrid.uniform((-10, 10), (0, 60), 0.5)
    b = IceModelBuilder(grid, column, air=False)
    b.set_fabric(np.ones(b.rho.shape, dtype=bool), dlambda=0.3, lam_z=0.5)
    assert np.allclose(b.lam.sum(axis=-1), 1.0)
    np.testing.assert_allclose(b.dlambda, 0.3)
    # And the permittivity split follows the fabric contrast.
    model = b.finalize()
    deep = model.eps["xx"][:, -1] - model.eps["yy"][:, -1]
    assert np.all(deep > 0)


def test_fabric_transition_is_polarisation_dependent():
    """A fabric change can be bright to one polarisation and dark to the other.

    This is what separates a fabric reflector from a density or acidity layer,
    and it is the mechanism behind example 3.
    """
    # Solid ice throughout, so the only thing that changes across the boundary
    # is the fabric -- otherwise the firn density gradient dominates.
    column = IceColumn(rho_sfc=0.999, rho_bco=0.9999, depth_bco=1.0, e0=0.0)
    grid = PropertyGrid.uniform((-10, 10), (0, 120), 0.5)
    b = IceModelBuilder(grid, column, air=False)
    below = b.below_mask(lambda x: np.full_like(x, 60.0))
    b.set_fabric(~below, dlambda=0.03, lam_z=0.34)
    b.set_fabric(below, dlambda=0.34, lam_z=0.58)
    model = b.finalize()

    j_above = np.argmin(np.abs(grid.z - 50.0))
    j_below = np.argmin(np.abs(grid.z - 70.0))
    jump_x = abs(model.eps["xx"][10, j_below] - model.eps["xx"][10, j_above])
    jump_y = abs(model.eps["yy"][10, j_below] - model.eps["yy"][10, j_above])
    assert jump_y > 5 * jump_x
