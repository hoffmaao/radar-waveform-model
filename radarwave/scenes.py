"""Construction of two-dimensional ice-sheet models for the FDTD solvers.

An :class:`IceModelBuilder` carries the *physical* fields -- density, bubble
eccentricity and the three fabric eigenvalues -- on the property grid, lets you
modify them geometrically, and only converts to permittivity and conductivity
at the end.  Doing it in that order matters: a density anomaly and a fabric
change enter the Maxwell-Garnett mixing differently, so editing permittivity
directly would not be self-consistent.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .grid import PropertyGrid
from .ice import (
    IceColumn,
    conductivity_from_density,
    fabric_permittivity,
    maxwell_garnett,
)

__all__ = [
    "Layer",
    "IceModelBuilder",
    "conformal_layering",
    "undulating_depth",
    "dipping_depth",
]


def conformal_layering(
    max_depth,
    *,
    seed=7,
    top=20.0,
    firn_base=95.0,
    d_rho=(0.010, 0.020),
    sigma_factor=(2.5, 3.5),
    firn_spacing=(14.0, 20.0),
    ice_spacing=(24.0, 36.0),
    thickness_firn=(1.2, 2.0),
    thickness_ice=(0.8, 1.4),
    amplitude=(1.4, 4.0),
    wavelengths=(300.0, 95.0),
    exclude=None,
):
    """Meteoric internal layering that is conformable with the surface.

    Every layer is given the *same* undulation shape, with the amplitude
    growing downwards.  That is what conformable means: the layers stay
    parallel to one another and to the surface.  Drawing independent random
    phases per layer would look superficially similar but would not be
    conformable, and it removes the contrast that makes a discordant feature
    (a folded or dipping fabric transition) read as discordant.

    Two kinds of contrast, matching what a real column carries: density
    (permittivity) banding down to ``firn_base``, and acidity (conductivity)
    horizons in the solid ice below, more widely spaced.

    Parameters
    ----------
    max_depth : float
        Layers are generated down to roughly ``max_depth - 18`` m.
    seed : int
        Seeds the layer depths, thicknesses and contrast amplitudes.  The same
        seed gives the same stratigraphy, so two models can share it.
    amplitude : (float, float)
        Undulation amplitude at the surface, and its increase per 250 m.
    exclude : (float, float) or sequence of them, optional
        Depth bands to leave clear of layers.  A layer whose two-way time
        coincides with the event an example is about puts its own wavelet on
        top of that event, and no amount of processing separates two arrivals
        at the same time.  Clearing a band a few wavelengths wide leaves the
        event alone in the record.  The band is applied to the layer's nominal
        depth; the undulations then carry it a few metres either side, which is
        why the band wants to be wider than the wavelet.

    Returns
    -------
    list of Layer
    """
    if exclude is None:
        bands = []
    else:
        bands = [exclude] if np.ndim(exclude[0]) == 0 else list(exclude)
    bands = [(float(min(a, b)), float(max(a, b))) for a, b in bands]
    rng = np.random.default_rng(seed)
    phase_long, phase_short = rng.uniform(0, 2 * np.pi, size=2)
    a0, a_grad = amplitude
    wl_long, wl_short = wavelengths

    layers, depth = [], float(top)
    while depth < max_depth - 18.0:
        amp = a0 + a_grad * depth / 250.0
        undul = [(amp, wl_long, phase_long), (0.3 * amp, wl_short, phase_short)]
        # Draw before deciding whether to keep the layer, so excluding a band
        # does not shift the random stream and change the stratigraphy either
        # side of it.  Two models with the same seed then differ only in the
        # band, which is what makes them comparable.
        if depth < firn_base:
            layer = Layer(
                depth=depth, thickness=rng.uniform(*thickness_firn),
                d_rho=rng.choice([-1.0, 1.0]) * rng.uniform(*d_rho),
                undulations=undul)
            step = rng.uniform(*firn_spacing)
        else:
            layer = Layer(
                depth=depth, thickness=rng.uniform(*thickness_ice),
                sigma_factor=rng.uniform(*sigma_factor), undulations=undul)
            step = rng.uniform(*ice_spacing)
        if not any(lo <= depth <= hi for lo, hi in bands):
            layers.append(layer)
        depth += step
    return layers


def undulating_depth(x, depth, undulations=(), dip_deg=0.0, x0=0.0):
    """Depth of a surface at horizontal positions ``x``.

    ``undulations`` is a sequence of ``(amplitude_m, wavelength_m, phase_rad)``
    triples that are summed, giving the gentle folding that real internal
    stratigraphy shows.  ``dip_deg`` adds a constant slope, positive meaning
    the surface deepens towards +x.
    """
    x = np.asarray(x, dtype=float)
    z = depth + np.tan(np.deg2rad(dip_deg)) * (x - x0)
    for amp, wavelength, phase in undulations:
        z = z + amp * np.sin(2 * np.pi * (x - x0) / wavelength + phase)
    return z


def dipping_depth(x, depth_at_x0, dip_deg, x0=0.0):
    """A planar surface through ``(x0, depth_at_x0)`` dipping at ``dip_deg``."""
    return undulating_depth(x, depth_at_x0, dip_deg=dip_deg, x0=x0)


@dataclass
class Layer:
    """A thin sub-horizontal layer carrying a density and/or acidity anomaly.

    Parameters
    ----------
    depth : float
        Depth of the layer centre below the surface at ``x = x0`` (m).
    thickness : float
        Full thickness (m).
    d_rho : float
        Relative-density anomaly added inside the layer (fraction of solid ice).
        This is what makes firn stratigraphy visible to radar.
    sigma_factor : float
        Multiplier on conductivity, used for volcanic acid horizons, which are
        what produce most deep internal reflections in cold ice.
    dip_deg : float
        Constant dip in degrees, positive deepening towards +x.
    undulations : sequence
        ``(amplitude_m, wavelength_m, phase_rad)`` triples summed into the
        layer geometry.
    """

    depth: float
    thickness: float = 1.0
    d_rho: float = 0.0
    sigma_factor: float = 1.0
    dip_deg: float = 0.0
    undulations: Sequence[Tuple[float, float, float]] = ()

    def depth_at(self, x, x0=0.0):
        return undulating_depth(x, self.depth, self.undulations, self.dip_deg, x0)


class IceModelBuilder:
    """Assemble a 2-D ice model on a property grid.

    Parameters
    ----------
    grid : PropertyGrid
        An empty grid of the right extent (properties are overwritten).
    column : IceColumn
        The background firn/fabric column, evaluated as a function of depth
        below ``surface``.
    surface : float or callable
        Depth of the snow surface.  A callable receives ``x`` and returns the
        surface height, which lets the surface undulate.
    air : bool
        Fill everything above the surface with air.  Including a few metres of
        air costs a smaller time step but produces the surface return that
        dominates any real radargram.
    """

    def __init__(self, grid: PropertyGrid, column: IceColumn, surface=0.0, air=True):
        self.grid = grid
        self.column = column
        self.air = air

        X, Z = grid.mesh()
        self.X, self.Z = X, Z
        self.surface = surface(grid.x) if callable(surface) else np.full(grid.x.size, surface)
        self.depth = Z - self.surface[:, None]  # depth below the local surface
        self.is_air = (self.depth < 0.0) if air else np.zeros(Z.shape, dtype=bool)

        d = np.maximum(self.depth, 0.0)
        self.rho = column.density(d)
        self.ecc = column.eccentricity(d)
        self.lam = column.eigenvalues(d)  # (nx, nz, 3)
        self.sigma_factor = np.ones_like(self.rho)

    # -- geometry helpers -------------------------------------------------

    def layer_mask(self, layer: Layer) -> np.ndarray:
        centre = layer.depth_at(self.grid.x)[:, None]
        return np.abs(self.depth - centre) <= layer.thickness / 2.0

    def below_mask(self, depth_fn) -> np.ndarray:
        """Mask of everything below a surface ``depth_fn(x) -> depth``."""
        boundary = np.asarray(depth_fn(self.grid.x), dtype=float)[:, None]
        return self.depth >= boundary

    # -- editing ----------------------------------------------------------

    def add_layer(self, layer: Layer) -> "IceModelBuilder":
        mask = self.layer_mask(layer) & ~self.is_air
        if layer.d_rho:
            self.rho[mask] = np.clip(self.rho[mask] + layer.d_rho, 0.05, 1.0)
        if layer.sigma_factor != 1.0:
            self.sigma_factor[mask] *= layer.sigma_factor
        return self

    def add_layers(self, layers: Sequence[Layer]) -> "IceModelBuilder":
        for layer in layers:
            self.add_layer(layer)
        return self

    def set_fabric(self, mask, *, dlambda=None, lam_z=None) -> "IceModelBuilder":
        """Override the fabric eigenvalues where ``mask`` is true.

        ``dlambda`` and ``lam_z`` may be scalars or arrays broadcastable to the
        grid; ``lam_x`` and ``lam_y`` are rebuilt from the horizontal budget so
        the eigenvalues still sum to one.
        """
        mask = mask & ~self.is_air
        lz = self.lam[..., 2] if lam_z is None else np.broadcast_to(
            np.asarray(lam_z, dtype=float), self.rho.shape
        )
        current_dl = self.lam[..., 0] - self.lam[..., 1]
        dl = current_dl if dlambda is None else np.broadcast_to(
            np.asarray(dlambda, dtype=float), self.rho.shape
        )
        h = 1.0 - lz
        dl = np.clip(dl, -h, h)
        self.lam[..., 0] = np.where(mask, (h + dl) / 2.0, self.lam[..., 0])
        self.lam[..., 1] = np.where(mask, (h - dl) / 2.0, self.lam[..., 1])
        self.lam[..., 2] = np.where(mask, lz, self.lam[..., 2])
        return self

    def set_density(self, mask, rho) -> "IceModelBuilder":
        self.rho = np.where(mask & ~self.is_air, rho, self.rho)
        return self

    def scale_conductivity(self, mask, factor) -> "IceModelBuilder":
        self.sigma_factor = np.where(mask & ~self.is_air, self.sigma_factor * factor,
                                     self.sigma_factor)
        return self

    def add_bed(self, depth_fn, eps=None, sigma=None) -> "IceModelBuilder":
        """Record a bed half-space, applied during :meth:`finalize`."""
        self._bed = (depth_fn, eps, sigma)
        return self

    # -- output -----------------------------------------------------------

    @property
    def dlambda(self) -> np.ndarray:
        """Current horizontal fabric contrast ``lam_x - lam_y`` on the grid."""
        return self.lam[..., 0] - self.lam[..., 1]

    def permittivity(self) -> np.ndarray:
        """Effective permittivity tensor diagonal, shape ``(nx, nz, 3)``."""
        eps = maxwell_garnett(fabric_permittivity(self.lam), self.rho, self.ecc)
        if self.air:
            eps = np.where(self.is_air[..., None], 1.0, eps)
        return eps

    def conductivity(self) -> np.ndarray:
        sig = conductivity_from_density(self.rho, self.column.sigma_ice) * self.sigma_factor
        if self.air:
            sig = np.where(self.is_air, 0.0, sig)
        return sig

    def finalize(self, npml: Optional[int] = None) -> PropertyGrid:
        """Return a :class:`PropertyGrid`, optionally padded for the PML."""
        eps = self.permittivity()
        sig = self.conductivity()

        bed = getattr(self, "_bed", None)
        if bed is not None:
            depth_fn, bed_eps, bed_sig = bed
            mask = self.below_mask(depth_fn)
            if bed_eps is not None:
                eps = np.where(mask[..., None], bed_eps, eps)
            if bed_sig is not None:
                sig = np.where(mask, bed_sig, sig)

        out = self.grid.with_properties(
            eps={"xx": eps[..., 0], "yy": eps[..., 1], "zz": eps[..., 2]},
            sig={"xx": sig, "yy": sig, "zz": sig},
        )
        out.metadata.update(
            {
                "surface": self.surface,
                "column": self.column,
            }
        )
        return out.padded(npml) if npml is not None else out
