"""Polarimetric processing and 2-D scene construction."""

import numpy as np
import pytest

from radarwave import C0, IceColumn, PropertyGrid, RIDGE_A, gabor, ridge_a_dlambda
from radarwave.polarimetry import (
    analytic,
    delay_from_phase,
    delayed_copy,
    dlambda_from_delay,
    interferogram,
    isolate_arrival,
    subsample_lag,
    unwrap_phase,
    volume_scattering,
)
from radarwave.scenes import (
    IceModelBuilder,
    Layer,
    conformal_layering,
    dipping_depth,
    undulating_depth,
)

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


def test_subsample_lag_survives_a_peak_on_an_end_sample():
    """Parabolic refinement must not index off the end of the correlation."""
    a = np.zeros(64)
    a[32] = 1.0
    assert subsample_lag(a, a, 1.0) == pytest.approx(0.0, abs=1e-9)
    # A lag of exactly half the correlation length lands the peak on sample 0.
    b = np.roll(a, 64)
    assert np.isfinite(subsample_lag(a, b, 1.0))


def test_isolate_arrival_recovers_a_delay_buried_in_a_wake():
    """A common, delay-free background must not drag the measured lag to zero."""
    dt = 0.5e-9
    t = np.arange(0, 4e-6, dt)
    fc, t0, lag = 60e6, 3.0e-6, 20e-9

    def echo(centre):
        x = t - centre
        return np.exp(-((x / 25e-9) ** 2)) * np.cos(2 * np.pi * fc * x)

    # Across one window the transmit wake reads as a strong sloping background.
    # It is identical in both traces, so it carries no delay at all, and being
    # twenty times the echo it dominates the raw correlation and drags the
    # answer to zero -- which is how a 40 ns split gets reported as 3 ns.
    wake = 20.0 * (t - t0) / 1e-6
    par, perp = wake + echo(t0), wake + echo(t0 + lag)

    idx = np.abs(t - t0) < 0.12e-6
    raw = subsample_lag(par[idx], perp[idx], dt)
    _, (a, b) = isolate_arrival(t, [par, perp], t0, 0.12e-6)
    assert abs(raw - lag) > 5e-9  # the background wins outright
    assert subsample_lag(a, b, dt) == pytest.approx(lag, rel=0.05)

    # Without the taper the window is still detrended, which is what makes an
    # amplitude readable off it.
    _, (c, d) = isolate_arrival(t, [par, perp], t0, 0.12e-6, taper=False)
    assert np.max(np.abs(c)) == pytest.approx(np.max(np.abs(d)), rel=0.02)
    assert abs(np.mean(c)) < 1e-9


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


def test_analytic_does_not_wrap_the_transmit_pulse():
    """The end of a record must not inherit the transmit pulse.

    ``scipy.signal.hilbert`` convolves circularly and the Hilbert kernel decays
    only as 1/t, so an impulse at t = 0 bleeds onto the last samples of the
    record -- about 107 dB of spurious "reflection" on these traces, which is
    what ``analytic`` zero-pads to avoid.
    """
    from scipy.signal import hilbert

    n = 4096
    t = np.arange(n) * 0.5e-9
    trace = 0.6 * np.exp(-(((t - 2e-9) / 3e-9) ** 2)) * np.cos(2 * np.pi * 60e6 * (t - 2e-9))
    trace += 5e-7 * np.exp(-(((t - 5e-7) / 12e-9) ** 2)) * np.cos(2 * np.pi * 60e6 * (t - 5e-7))

    padded = np.abs(analytic(trace))
    tail_db = 20 * np.log10(padded[-60:].max() / padded.max())
    assert tail_db < -80, f"end of record sits at {tail_db:.1f} dB"

    # The unpadded transform is what this guards against: on this trace it puts
    # the record's end within a few dB of the transmit pulse.
    raw = np.abs(hilbert(trace))
    raw_tail_db = 20 * np.log10(raw[-60:].max() / raw.max())
    assert raw_tail_db > -10
    assert raw_tail_db - tail_db > 60


def test_analytic_preserves_the_signal():
    """Padding must not change what the transform is.

    Checked against the definition rather than against ``hilbert`` itself: the
    circular wrap contaminates the raw transform across the whole record, not
    only at its ends, so it is not a valid reference here.
    """
    n = 2048
    t = np.arange(n) * 0.5e-9
    envelope = np.exp(-(((t - 5e-7) / 1.5e-7) ** 2))
    trace = envelope * np.cos(2 * np.pi * 60e6 * t)

    z = analytic(trace)
    assert z.shape == trace.shape
    # The analytic signal carries the original as its real part.
    np.testing.assert_allclose(np.real(z), trace, rtol=1e-9, atol=1e-12)
    # ... and its modulus is the envelope it was built from.
    inner = slice(200, -200)
    np.testing.assert_allclose(np.abs(z)[inner], envelope[inner], rtol=0.02, atol=1e-3)


def test_analytic_handles_a_2d_stack():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(512, 4))
    z = analytic(data, axis=0)
    assert z.shape == data.shape
    np.testing.assert_allclose(np.real(z), data, rtol=1e-9, atol=1e-12)


def test_conformal_layering_clears_an_excluded_band():
    """A band asked for is left empty, and only that band changes.

    The exclusion must not shift the random stream: two models built from the
    same seed have to share their stratigraphy everywhere outside the band, or
    a run with the gap is not comparable with one without it.
    """
    band = (125.0, 155.0)
    full = conformal_layering(300.0, seed=7)
    gapped = conformal_layering(300.0, seed=7, exclude=band)

    assert [lay.depth for lay in full if band[0] <= lay.depth <= band[1]]
    assert not [lay.depth for lay in gapped if band[0] <= lay.depth <= band[1]]

    outside = [lay for lay in full if not band[0] <= lay.depth <= band[1]]
    assert len(outside) == len(gapped)
    for a, b in zip(outside, gapped):
        assert a.depth == b.depth
        assert a.thickness == b.thickness
        assert a.d_rho == b.d_rho
        assert a.sigma_factor == b.sigma_factor


def test_conformal_layering_accepts_several_excluded_bands():
    bands = [(60.0, 80.0), (200.0, 230.0)]
    layers = conformal_layering(300.0, seed=3, exclude=bands)
    for lo, hi in bands:
        assert not [lay.depth for lay in layers if lo <= lay.depth <= hi]
    assert len(layers) > 3


@pytest.mark.parametrize("empty", [[], (), np.empty((0, 2))])
def test_conformal_layering_excludes_nothing_for_an_empty_band_list(empty):
    """No bands is not an error: it is the same as not asking for any.

    A caller that builds its bands programmatically naturally ends up with an
    empty list, and a single band given as an array has to keep working.
    """
    reference = conformal_layering(300.0, seed=5)
    assert [lay.depth for lay in conformal_layering(300.0, seed=5, exclude=empty)] == [
        lay.depth for lay in reference
    ]
    one = conformal_layering(300.0, seed=5, exclude=np.array([125.0, 155.0]))
    assert not [lay.depth for lay in one if 125.0 <= lay.depth <= 155.0]


def test_strip_text_clears_titles_annotations_and_insets():
    """--bare has to reach inset axes, which are not in ``fig.axes``.

    Axis labels, tick labels and colourbar text are deliberately kept: they are
    what lets a slide still be read quantitatively.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from radarwave.viz import strip_text

    fig, ax = plt.subplots()
    im = ax.imshow(np.arange(9).reshape(3, 3))
    cb = fig.colorbar(im)
    cb.set_label("colourbar label")
    fig.suptitle("figure title")
    fig.text(0.1, 0.9, "figure note")
    ax.set_title("axes title", loc="left")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("depth (m)")
    ax.annotate("in-plot label", (1, 1))
    ax.plot([0, 1], [0, 1], label="a line")
    ax.legend()
    inset = ax.inset_axes([0.5, 0.5, 0.4, 0.4])
    inset.set_title("inset title")
    # A figure-level legend is not on any axes, so the per-axes sweep never sees
    # it and it would otherwise survive --bare with its labels intact.
    fig.legend(loc="lower right")

    strip_text(fig)

    assert list(fig.legends) == []

    # The suptitle stays attached but blank: removing it leaves ``fig._suptitle``
    # dangling and the next tight_layout dies measuring it.  Every other
    # figure-level text goes.
    assert fig._suptitle.get_text() == ""
    assert [t.get_text() for t in fig.texts] == [""]
    assert ax.get_title(loc="left") == ""
    assert list(ax.texts) == []
    assert ax.get_legend() is None
    assert inset.get_title() == ""
    # kept: the axes stay readable
    assert ax.get_xlabel() == "distance (m)"
    assert ax.get_ylabel() == "depth (m)"
    assert cb.ax.get_ylabel() == "colourbar label"
    plt.close(fig)


def _examples_on_path():
    """Put the examples on ``sys.path``: they are a script directory, not a package."""
    import sys
    from pathlib import Path

    path = str(Path(__file__).resolve().parents[1] / "examples")
    if path not in sys.path:
        sys.path.insert(0, path)


def _common():
    """The examples' shared helpers."""
    _examples_on_path()
    import _common

    return _common


def test_snapshot_cache_roundtrips_with_a_matching_stamp(tmp_path):
    mod = _common()
    stamp = {"dx": 0.32, "xlim": (-110.0, 110.0), "fabric": {"lam_z": 0.8}}
    cache = mod.save_snapshots(
        tmp_path / "snapshots.npz", [(np.ones((2, 3, 4)), "E along 89 deg")],
        np.arange(3.0), np.arange(4.0), np.arange(2.0), stamp=stamp, trace=np.zeros(5),
    )
    panels, x, z, times, extra = mod.load_snapshots(cache, stamp=stamp)
    assert [lbl for _, lbl in panels] == ["E along 89 deg"]
    assert panels[0][0].shape == (2, 3, 4)
    # The stamp must not leak into the payload the caller unpacks.
    assert sorted(extra) == ["trace"]
    assert x.size == 3 and z.size == 4 and times.size == 2


def test_snapshot_cache_refuses_a_stamp_that_does_not_match(tmp_path):
    """A model change has to fail the reload, not be rendered under new numbers.

    Re-rendering a stale cache draws the old wavefield and the old trace beside
    predictions computed from the current constants, and nothing in the figure
    says so -- which is exactly the failure this guards.
    """
    mod = _common()
    args = ([(np.ones((2, 3, 4)), "panel")], np.arange(3.0), np.arange(4.0),
            np.arange(2.0))
    cache = mod.save_snapshots(tmp_path / "snapshots.npz", *args,
                               stamp={"transition_width": 0.6, "dip_deg": 35.0})

    with pytest.raises(mod.StaleCache) as exc:
        mod.load_snapshots(cache, stamp={"transition_width": 1.2, "dip_deg": 35.0})
    # The message has to name the parameter that moved, and only that one.
    assert "transition_width" in str(exc.value)
    assert "dip_deg" not in str(exc.value)
    assert "--render-only" in str(exc.value)

    # A parameter that did not exist when the cache was written also counts.
    with pytest.raises(mod.StaleCache, match="layer_exclusion_band"):
        mod.load_snapshots(cache, stamp={"transition_width": 0.6, "dip_deg": 35.0,
                                         "layer_exclusion_band": (100.0, 130.0)})

    # A parameter nested inside a mapping of constructor arguments is named as
    # itself, not as the whole mapping having moved.
    nested = mod.save_snapshots(
        tmp_path / "nested.npz", *args,
        stamp={"column": {"thickness": 1850.0, "sigma_ice": 1.2e-5}},
    )
    with pytest.raises(mod.StaleCache) as exc:
        mod.load_snapshots(nested, stamp={"column": {"thickness": 1850.0,
                                                     "sigma_ice": 9.9e-5}})
    assert "column.sigma_ice: cache has 1.2e-05, model now has 9.9e-05" in str(exc.value)
    assert "column.thickness" not in str(exc.value)

    # A cache from before stamping existed cannot be trusted either.
    unstamped = mod.save_snapshots(tmp_path / "old.npz", *args)
    with pytest.raises(mod.StaleCache, match="no model stamp"):
        mod.load_snapshots(unstamped, stamp={"transition_width": 1.2})
    # ...but an unstamped cache still loads when no stamp is asked for.
    assert mod.load_snapshots(unstamped)[0][0][1] == "panel"


def test_render_only_refuses_a_cache_that_is_not_there(tmp_path):
    """A missing cache is a failed request, not a licence to simulate.

    Falling through to the simulation costs tens of minutes and gigabytes on
    exactly the run ``--render-only`` was chosen to avoid, so it has to fail the
    same way a stale cache does.
    """
    mod = _common()
    missing = tmp_path / "snapshots.npz"

    with pytest.raises(mod.StaleCache) as exc:
        mod.render_from_cache(missing, render_only=True)
    assert str(missing) in str(exc.value)
    assert "--render-only" in str(exc.value)

    # Without the flag a missing cache is simply the run that has to happen.
    assert mod.render_from_cache(missing, render_only=False) is False

    mod.save_snapshots(missing, [(np.ones((2, 3, 4)), "panel")],
                       np.arange(3.0), np.arange(4.0), np.arange(2.0))
    assert mod.render_from_cache(missing, render_only=True) is True


def test_importing_ex04_leaves_ex03s_geometry_alone():
    """Importing one example must not re-aim another one's geometry.

    ex04 sounds the same shape at its own dip and depth.  Assigning those to
    ex03's module constants would be shorter than passing them, and would move
    ex03's specular point from -82.7 m to -71.4 m for anything else sharing the
    interpreter -- silently, since ex03 would still print a self-consistent
    number.  Both modules do no work at import time, so pinning it is cheap.
    """
    _examples_on_path()
    import ex03_dipping_fabric_transition as ex03
    import ex04_banded_fabric as ex04

    assert ex03.DIP_DEG == 35.0
    assert ex03.DEPTH_AT_X0 == 175.0
    _, px, pz = ex03.specular_geometry(0.0)
    assert px == pytest.approx(-82.69, abs=0.01)
    assert pz == pytest.approx(117.09, abs=0.01)

    # ...while ex04 still gets its own geometry out of the shared helper.
    _, px4, pz4 = ex03.specular_geometry(ex04.SRC_X, ex04.DIP_DEG, ex04.DEPTH_AT_X0)
    assert px4 == pytest.approx(-18.76, abs=0.01)
    assert pz4 == pytest.approx(346.01, abs=0.01)


def test_ex04_lays_horizons_through_its_column_and_across_its_own_event():
    """ex04's nadir stratigraphy is unbroken, and that is the deliberate part.

    ex03 clears a band of layers around its event because there the fabric
    transition and the layering come back within a few dB of each other, and a
    coincident horizon would bury it.  ex04's package returns clear of the
    brightest horizon, so the gap buys nothing -- while a gap cut in the
    stratigraphy exactly where the answer is would be the first thing a viewer
    distrusted.  The layering also has to reach the floor of a domain more than twice as
    deep as ex03's, or the record carries no nadir reference over most of its
    length and the package has nothing to be read against.
    """
    _examples_on_path()
    import ex03_dipping_fabric_transition as ex03
    import ex04_banded_fabric as ex04
    from radarwave import IceColumn

    zlim = (-8.0, 560.0)
    depths = np.array([lay.depth for lay in
                       conformal_layering(zlim[1], **ex04.LAYERING)])

    assert depths.max() > zlim[1] - 40.0
    ice = np.sort(depths[depths > 95.0])
    assert ice.size >= 18
    assert np.diff(ice).max() <= ex04.LAYERING["ice_spacing"][1]

    # ex03's helper is what knows where the package images at nadir; there has
    # to be a horizon inside the band ex03 would have cleared.
    column = IceColumn(**ex03.COLUMN)
    d_event, _ = ex03.layer_exclusion(column, zlim, ex04.SRC_X,
                                      ex04.DIP_DEG, ex04.DEPTH_AT_X0)
    assert np.min(np.abs(depths - d_event)) < ex03.LAYER_GAP_HALFWIDTH
