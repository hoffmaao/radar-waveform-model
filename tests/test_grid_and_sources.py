"""Grids, discretisation criteria and source wavelets."""

import numpy as np
import pytest

from radarwave import (
    C0,
    PropertyGrid,
    blackharrispulse,
    dominant_frequency,
    gabor,
    max_spatial_step,
    max_time_step,
    pad_grid,
    regrid,
    ricker,
)


# -- sources ------------------------------------------------------------


@pytest.mark.parametrize("fc", [25e6, 100e6, 400e6])
def test_blackharris_peak_frequency(fc):
    dt = 1.0 / (60.0 * fc)
    t = np.arange(0, 6.0 / fc, dt)
    p = blackharrispulse(fc, t)
    assert dominant_frequency(p, dt) == pytest.approx(fc, rel=0.1)
    assert np.max(np.abs(p)) == pytest.approx(1.0)


def test_blackharris_has_no_dc():
    """A pulse with a DC component would leave a static field behind."""
    dt = 1e-10
    t = np.arange(0, 60e-9, dt)
    p = blackharrispulse(100e6, t)
    assert abs(np.sum(p) * dt) < 1e-3 * np.max(np.abs(p)) * (t[-1] - t[0])


def test_blackharris_is_compact():
    """The pulse ends after 1.14/fr, give or take the one sample the finite
    difference straddles the truncation on."""
    dt = 1e-10
    t = np.arange(0, 100e-9, dt)
    p = blackharrispulse(100e6, t)
    tail = p[t > 1.14 / 100e6 + 2 * dt]
    assert np.all(tail == 0)
    assert np.max(np.abs(p[t > 1.14 / 100e6])) < 1e-4 * np.max(np.abs(p))


@pytest.mark.parametrize("bandwidth", [0.1, 0.2, 0.4])
def test_gabor_bandwidth(bandwidth):
    """The wavelet delivers the fractional bandwidth it was asked for."""
    fc = 195e6
    dt = 1.0 / (40.0 * fc)
    t = np.arange(0, 40.0 / fc, dt)
    p = gabor(fc, t, bandwidth=bandwidth)

    n = 1 << 16
    spec = np.abs(np.fft.rfft(p, n))
    f = np.fft.rfftfreq(n, dt)
    assert f[np.argmax(spec)] == pytest.approx(fc, rel=0.02)

    half = spec >= 0.5 * spec.max()
    fwhm = f[half].max() - f[half].min()
    assert fwhm / fc == pytest.approx(bandwidth, rel=0.15)


def test_ricker_is_zero_mean():
    dt = 1e-10
    t = np.arange(0, 80e-9, dt)
    p = ricker(100e6, t)
    assert abs(np.mean(p)) < 0.02 * np.max(np.abs(p))


# -- discretisation -----------------------------------------------------


def test_max_time_step_scales_correctly():
    """dt_max is proportional to spacing and to sqrt(eps)."""
    a = max_time_step(1.0, 1.0, 0.2, 0.2)
    assert max_time_step(1.0, 1.0, 0.4, 0.4) == pytest.approx(2 * a)
    assert max_time_step(4.0, 1.0, 0.2, 0.2) == pytest.approx(2 * a)


def test_max_spatial_step_gives_five_per_wavelength():
    dt = 1e-10
    t = np.arange(0, 100e-9, dt)
    p = blackharrispulse(100e6, t)
    dx, wlmin, fmax = max_spatial_step(3.17, 1.0, p, t)
    assert dx == pytest.approx(wlmin / 5)
    assert fmax > 100e6  # the significant band runs above the spectral peak
    assert wlmin == pytest.approx(C0 / np.sqrt(3.17) / fmax, rel=1e-6)


# -- property grids -----------------------------------------------------


def test_uniform_grid_shape_and_spacing():
    g = PropertyGrid.uniform((-10, 10), (0, 5), 0.25, eps=3.0)
    nxp, nzp = g.shape
    assert nxp % 2 == 1 and nzp % 2 == 1
    assert g.dx == pytest.approx(0.25)
    assert g.dz == pytest.approx(0.25)
    assert g.nx == (nxp + 1) // 2
    assert g.x[0] == pytest.approx(-10.0)
    g.validate()


def test_anisotropic_construction():
    g = PropertyGrid.uniform((0, 4), (0, 4), 0.5, eps={"xx": 3.2, "yy": 3.1, "zz": 3.3})
    assert np.allclose(g.eps["xx"], 3.2)
    assert np.allclose(g.eps["yy"], 3.1)
    assert g.eps_min == pytest.approx(3.1)
    assert g.eps_max == pytest.approx(3.3)
    g.validate()


def test_missing_component_is_rejected():
    with pytest.raises(ValueError, match="missing principal components"):
        PropertyGrid.uniform((0, 2), (0, 2), 0.5, eps={"xx": 3.0, "yy": 3.0})


def test_permittivity_below_one_is_rejected():
    g = PropertyGrid.uniform((0, 2), (0, 2), 0.5, eps=1.0)
    bad = g.with_properties(eps=0.5)
    with pytest.raises(ValueError, match="must be >= 1"):
        bad.validate()


def test_padding_preserves_parity_and_edge_values():
    g = PropertyGrid.uniform((0, 4), (0, 4), 0.5, eps=3.0)
    X, _ = g.mesh()
    g = g.with_properties(eps=3.0 + 0.1 * X)
    p = g.padded(npml=4)
    n = 2 * 4 + 2
    assert p.shape[0] == g.shape[0] + 2 * n
    assert p.shape[0] % 2 == 1 and p.shape[1] % 2 == 1
    assert p.dx == pytest.approx(g.dx)
    # Edge replication, so the PML never straddles a material contrast.
    assert np.allclose(p.eps["xx"][0, :], p.eps["xx"][n, :])
    assert np.allclose(p.eps["xx"][-1, :], p.eps["xx"][-n - 1, :])
    p.validate()


def test_pad_grid_coordinates():
    a = np.arange(9.0).reshape(3, 3)
    x = np.array([0.0, 1.0, 2.0])
    z = np.array([0.0, 1.0, 2.0])
    a2, x2, z2 = pad_grid(a, x, z, 2)
    assert a2.shape == (7, 7)
    assert x2[0] == pytest.approx(-2.0)
    assert x2[-1] == pytest.approx(4.0)
    np.testing.assert_allclose(np.diff(x2), 1.0)


def test_regrid_reproduces_a_linear_field():
    x = np.linspace(0, 10, 11)
    z = np.linspace(0, 5, 6)
    X, Z = np.meshgrid(x, z, indexing="ij")
    a = 2.0 * X + 3.0 * Z
    x2 = np.linspace(0, 10, 21)
    z2 = np.linspace(0, 5, 11)
    out = regrid(a, x, z, x2, z2, method="linear")
    X2, Z2 = np.meshgrid(x2, z2, indexing="ij")
    np.testing.assert_allclose(out, 2.0 * X2 + 3.0 * Z2, atol=1e-10)


def test_regrid_clamps_outside_the_input_range():
    """Padding a grid queries outside the original extent; NaN there would
    poison the model."""
    x = np.linspace(0, 4, 5)
    z = np.linspace(0, 4, 5)
    a = np.ones((5, 5)) * 7.0
    out = regrid(a, x, z, np.array([-3.0, 7.0]), np.array([-3.0, 7.0]))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 7.0)
