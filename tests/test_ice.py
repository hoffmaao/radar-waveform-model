"""The firn/fabric dielectric model."""

import numpy as np
import pytest

from radarwave import C0, DEPS_ICE, EPS_ICE_MEAN, IceColumn, RIDGE_A, ridge_a_dlambda
from radarwave.ice import (
    depolarization,
    fabric_permittivity,
    herron_langway_density,
    maxwell_garnett,
)


def test_herron_langway_endpoints():
    """The profile hits the surface and close-off densities it was given."""
    rho_sfc, rho_bco, d_bco = 0.35, 0.81, 100.0
    assert herron_langway_density(0.0, rho_sfc, rho_bco, d_bco) == pytest.approx(rho_sfc)
    assert herron_langway_density(d_bco, rho_sfc, rho_bco, d_bco) == pytest.approx(rho_bco)
    deep = herron_langway_density(2000.0, rho_sfc, rho_bco, d_bco)
    assert 0.99 < deep <= 1.0


def test_density_increases_monotonically():
    z = np.linspace(0, 500, 200)
    rho = herron_langway_density(z)
    assert np.all(np.diff(rho) > 0)


def test_depolarization_limits():
    """Spherical bubbles give 1/3 on every axis; elongated ones give less."""
    assert depolarization(np.array([0.0]))[0] == pytest.approx(1 / 3)
    assert depolarization(np.array([1e-9]))[0] == pytest.approx(1 / 3, abs=1e-9)
    assert depolarization(np.array([0.8]))[0] < 1 / 3
    # Monotonic in eccentricity, and the trace stays one by construction.
    e = np.linspace(0, 0.95, 40)
    Nz = depolarization(e)
    assert np.all(np.diff(Nz) < 0)
    assert np.allclose(Nz + 2 * (1 - Nz) / 2, 1.0)


def test_isotropic_fabric_gives_mean_permittivity():
    eps = fabric_permittivity(np.array([1 / 3, 1 / 3, 1 / 3]))
    assert np.allclose(eps, EPS_ICE_MEAN)


def test_fabric_permittivity_difference_scales_with_dlambda():
    """eps_x - eps_y is deps * (lam_x - lam_y), which is the whole basis of
    the polarimetric method."""
    for dlam in (0.0, 0.05, 0.3):
        lam = np.array([(0.6 + dlam) / 2, (0.6 - dlam) / 2, 0.4])
        eps = fabric_permittivity(lam)
        assert eps[0] - eps[1] == pytest.approx(DEPS_ICE * dlam)


def test_maxwell_garnett_limits():
    """Solid ice returns the solid values; pure air returns 1."""
    eps_ice = fabric_permittivity(np.array([[1 / 3, 1 / 3, 1 / 3]]))
    solid = maxwell_garnett(eps_ice, np.array([1.0]), np.array([0.0]))
    assert np.allclose(solid, EPS_ICE_MEAN)

    air = maxwell_garnett(eps_ice, np.array([1e-9]), np.array([0.0]))
    assert np.allclose(air, 1.0, atol=1e-6)


def test_firn_permittivity_between_air_and_ice():
    col = IceColumn()
    eps = col.permittivity(np.array([0.0, 50.0, 500.0]))
    assert np.all(eps > 1.0)
    assert np.all(eps < EPS_ICE_MEAN + 0.05)
    # Denser with depth means larger permittivity.
    assert eps[0, 0] < eps[1, 0] < eps[2, 0]


def test_ridge_a_dlambda_profile():
    """The digitised profile is bounded, positive and clamps outside its range."""
    z = np.linspace(-100, 3000, 400)
    dl = ridge_a_dlambda(z)
    assert np.all(dl >= 0)
    assert np.all(dl < 0.1)
    assert ridge_a_dlambda(-50.0) == ridge_a_dlambda(0.0)
    assert ridge_a_dlambda(5000.0) == ridge_a_dlambda(1850.0)


def test_eigenvalues_sum_to_one():
    col = IceColumn(**RIDGE_A)
    lam = col.eigenvalues(np.linspace(0, 1800, 50))
    assert np.allclose(lam.sum(axis=-1), 1.0)
    assert np.all(lam >= 0)


def test_birefringent_delay_matches_closed_form():
    """For a uniform solid-ice column the integral has an analytic value."""
    dlam = 0.2
    col = IceColumn(
        thickness=2000.0,
        rho_sfc=0.999999,  # effectively solid ice everywhere
        rho_bco=0.9999995,
        depth_bco=1.0,
        e0=0.0,
        lam_z_sfc=0.4,
        lam_z_bed=0.4,
        dlambda=lambda d: np.full_like(np.asarray(d, float), dlam),
    )
    depth = 1000.0
    lam = np.array([(0.6 + dlam) / 2, (0.6 - dlam) / 2, 0.4])
    eps = fabric_permittivity(lam)
    expected = 2.0 * depth * (np.sqrt(eps[0]) - np.sqrt(eps[1])) / C0
    assert col.birefringent_delay(depth) == pytest.approx(expected, rel=2e-3)


def test_traveltime_consistency():
    """Two-way traveltime difference between axes equals the delay integral."""
    col = IceColumn(**RIDGE_A)
    d = 900.0
    delta = 2 * (col.traveltime(0, d, axis=0) - col.traveltime(0, d, axis=1))
    assert delta == pytest.approx(col.birefringent_delay(d), rel=1e-3)


def test_delay_grows_with_depth_and_dlambda():
    col = IceColumn(**RIDGE_A)
    d = col.birefringent_delay(np.array([200.0, 800.0, 1600.0]))
    assert np.all(np.diff(d) > 0)
    assert d[-1] > 0
