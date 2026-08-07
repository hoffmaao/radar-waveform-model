"""Dielectric model of a firn/ice column with crystal-orientation fabric.

This is a depth-parameterised Python port of the polarimetric traveltime
toolbox used to invert Open Polar Radar quad-polarimetric data
(``fabric_anisotropy/+ptt``), which in turn follows Rathmann (2026) and
Fujita et al. (2000).  Keeping one dielectric model on both sides means a
synthetic radargram made here can be pushed through the same inversion that is
applied to the real frames.

The ingredients are

1. a Herron-Langway density profile for the firn column,
2. vertically elongated air bubbles whose depolarisation tensor makes firn
   itself birefringent,
3. the second-order orientation tensor eigenvalues ``lam = (lam_x, lam_y,
   lam_z)`` of the ice crystal c-axes, which set the solid-ice
   eigenpermittivities through ``eps_i = eps_bar + deps * (lam_i - 1/3)``,
4. Maxwell-Garnett mixing of (1)-(3) into the effective permittivity tensor.

The x axis is the horizontal fabric principal axis in the survey plane.  For
Ridge A that is the 89 deg E of N direction, so ``lam_x - lam_y`` is exactly
the quantity plotted as ``dlam`` in the SCAR figures.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constants import C0, DEPS_ICE, EPS_AIR, EPS_ICE_MEAN

#: ``np.trapz`` is deprecated in NumPy 2 and slated for removal; the successor
#: is spelled ``np.trapezoid`` and does not exist before 2.0.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _cumtrapz(y, x):
    """Cumulative trapezoidal integral of ``y`` over ``x``, starting at zero."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.diff(x) * 0.5 * (y[:-1] + y[1:]))])

__all__ = [
    "IceColumn",
    "herron_langway_density",
    "depolarization",
    "fabric_permittivity",
    "maxwell_garnett",
    "conductivity_from_density",
    "RIDGE_A",
    "ridge_a_dlambda",
]


def herron_langway_density(depth, rho_sfc=0.35, rho_bco=0.81, depth_bco=100.0):
    """Single-stage Herron-Langway relative density (fraction of solid ice).

    Parameters
    ----------
    depth : array_like
        Depth below the surface (m).
    rho_sfc, rho_bco : float
        Relative density at the surface and at bubble close-off.
    depth_bco : float
        Depth of bubble close-off (m).
    """
    depth = np.asarray(depth, dtype=float)
    beta = np.log(1.0 / rho_sfc - 1.0)
    alpha = (np.log(1.0 / rho_bco - 1.0) - beta) / depth_bco
    return 1.0 / (1.0 + np.exp(alpha * depth + beta))


def depolarization(e):
    """Vertical depolarisation factor of a prolate (standing) spheroid.

    ``e`` is the eccentricity in [0, 1).  Horizontal factors follow from the
    unit trace: ``Nx = Ny = (1 - Nz) / 2``.
    """
    e = np.asarray(e, dtype=float)
    Nz = np.empty_like(e)
    small = e < 1e-3
    # Series expansion avoids the 0/0 at e = 0.
    Nz[small] = 1.0 / 3.0 - (2.0 / 15.0) * e[small] ** 2
    es = e[~small]
    Nz[~small] = (1 - es**2) / (2 * es**3) * (np.log((1 + es) / (1 - es)) - 2 * es)
    return Nz


def bubble_eccentricity(depth, e0=0.8, p=0.5, depth_e0=None, depth_bco=100.0):
    """Power-law bubble eccentricity, vanishing just below close-off."""
    depth = np.asarray(depth, dtype=float)
    if depth_e0 is None:
        depth_e0 = depth_bco
    zeta = np.clip((depth_e0 - depth) / depth_e0, 0.0, None)
    return e0 * zeta**p


def fabric_permittivity(lam):
    """Solid-ice eigenpermittivities from orientation-tensor eigenvalues.

    ``lam`` has shape ``(..., 3)`` holding ``(lam_x, lam_y, lam_z)``.
    """
    return EPS_ICE_MEAN + DEPS_ICE * (np.asarray(lam, dtype=float) - 1.0 / 3.0)


def maxwell_garnett(eps_ice, rho, e):
    """Mix air bubbles into solid ice (Rathmann 2026, eqs. 3.1-3.2).

    Parameters
    ----------
    eps_ice : (..., 3) array
        Solid-ice eigenpermittivities.
    rho : (...,) array
        Relative density (1 = solid ice).
    e : (...,) array
        Bubble eccentricity.
    """
    eps_ice = np.asarray(eps_ice, dtype=float)
    rho = np.asarray(rho, dtype=float)[..., None]
    Nz = depolarization(np.asarray(e, dtype=float))[..., None]
    N = np.concatenate([(1 - Nz) / 2, (1 - Nz) / 2, Nz], axis=-1)
    phi = (1.0 - rho) / (1.0 + N * rho * (EPS_AIR / eps_ice - 1.0))
    return eps_ice + phi * (EPS_AIR - eps_ice)


def conductivity_from_density(rho, sigma_ice=1.5e-5):
    """Effective conductivity (S/m) of firn, scaled by the ice volume fraction.

    ``sigma_ice = 1.5e-5`` S/m corresponds to about 14 dB/km one-way loss at
    radar frequencies, typical of cold East Antarctic ice.
    """
    return np.asarray(rho, dtype=float) * sigma_ice


# ---------------------------------------------------------------------------
# Ridge A: the site analysed in the SCAR figures
# ---------------------------------------------------------------------------

#: Horizontal fabric contrast ``dlam = lam_perp - lam_par`` at Ridge A,
#: digitised from the median depth profile of the Open Polar Radar frame
#: 20250108_02_009 inversion (25 intervals of ~74 m x 35 blocks of ~124 m).
#: The two horizontal principal directions are 89 deg (perp, the model x axis)
#: and 179 deg (par, the model y axis) east of north.  Above ~150 m the firn
#: carries too few fringes to constrain dlam, so that interval is tapered to
#: zero rather than taken from the inversion.
RIDGE_A_DLAMBDA_DEPTH = np.array(
    [0.0, 150.0, 275.0, 400.0, 600.0, 800.0, 1000.0, 1200.0, 1400.0, 1500.0, 1700.0, 1850.0]
)
RIDGE_A_DLAMBDA = np.array(
    [0.000, 0.015, 0.019, 0.032, 0.048, 0.053, 0.057, 0.061, 0.066, 0.068, 0.064, 0.060]
)


def ridge_a_dlambda(depth):
    """Interpolate the Ridge A horizontal fabric contrast onto ``depth`` (m)."""
    return np.interp(
        np.asarray(depth, dtype=float),
        RIDGE_A_DLAMBDA_DEPTH,
        RIDGE_A_DLAMBDA,
        left=RIDGE_A_DLAMBDA[0],
        right=RIDGE_A_DLAMBDA[-1],
    )


@dataclass
class IceColumn:
    """A one-dimensional firn/ice column with a depth-varying fabric.

    Parameters
    ----------
    thickness : float
        Ice thickness (m).
    rho_sfc, rho_bco, depth_bco : float
        Herron-Langway firn parameters.
    e0, p : float
        Bubble eccentricity amplitude and power-law exponent.
    lam_z_sfc, lam_z_bed : float
        Assumed vertical eigenvalue at the surface and bed.  Common-offset
        polarimetry cannot constrain ``lam_z``; 1/3 (isotropic) near the
        surface increasing downwards represents a divide where a vertical
        single maximum develops with depth.
    dlambda : callable, optional
        ``dlambda(depth) -> lam_x - lam_y``.  Defaults to zero (no horizontal
        anisotropy).
    sigma_ice : float
        Conductivity of solid ice (S/m).
    """

    thickness: float = 1850.0
    rho_sfc: float = 0.35
    rho_bco: float = 0.81
    depth_bco: float = 100.0
    e0: float = 0.8
    p: float = 0.5
    lam_z_sfc: float = 1.0 / 3.0
    lam_z_bed: float = 0.55
    dlambda: Optional[callable] = None
    sigma_ice: float = 1.5e-5

    def density(self, depth):
        return herron_langway_density(depth, self.rho_sfc, self.rho_bco, self.depth_bco)

    def eccentricity(self, depth):
        return bubble_eccentricity(depth, self.e0, self.p, depth_bco=self.depth_bco)

    def lam_z(self, depth):
        frac = np.clip(np.asarray(depth, dtype=float) / self.thickness, 0.0, 1.0)
        return self.lam_z_sfc + (self.lam_z_bed - self.lam_z_sfc) * frac

    def eigenvalues(self, depth):
        """Orientation-tensor eigenvalues ``(lam_x, lam_y, lam_z)`` vs depth."""
        depth = np.asarray(depth, dtype=float)
        lz = self.lam_z(depth)
        h = 1.0 - lz  # horizontal budget
        dl = np.zeros_like(depth) if self.dlambda is None else np.asarray(self.dlambda(depth))
        dl = np.clip(dl, -h, h)
        return np.stack([(h + dl) / 2.0, (h - dl) / 2.0, lz], axis=-1)

    def permittivity(self, depth):
        """Effective relative permittivity tensor diagonal ``(xx, yy, zz)``."""
        depth = np.asarray(depth, dtype=float)
        eps_solid = fabric_permittivity(self.eigenvalues(depth))
        return maxwell_garnett(eps_solid, self.density(depth), self.eccentricity(depth))

    def conductivity(self, depth):
        return conductivity_from_density(self.density(depth), self.sigma_ice)

    def velocity(self, depth):
        """Phase velocity (m/s) along each principal axis."""
        return C0 / np.sqrt(self.permittivity(depth))

    def traveltime(self, z0, z1, axis=0, n=4001):
        """One-way traveltime (s) from ``z0`` to ``z1`` along a principal axis.

        ``axis`` is 0, 1 or 2 for x, y, z.  Use ``axis=None`` for the mean of
        the two horizontal axes, which is the ray a nadir sounder follows.
        """
        zz = np.linspace(float(z0), float(z1), n)
        eps = self.permittivity(zz)
        if axis is None:
            slow = 0.5 * (np.sqrt(eps[:, 0]) + np.sqrt(eps[:, 1])) / C0
        else:
            slow = np.sqrt(eps[:, axis]) / C0
        return float(_trapezoid(slow, zz))

    def _two_way_profile(self, zmax, n=4001):
        """``(zz, tt)``: two-way time from the surface to each depth in ``zz``."""
        zz = np.linspace(0.0, max(float(zmax), 1e-6), n)
        eps = self.permittivity(zz)
        slow = 0.5 * (np.sqrt(eps[:, 0]) + np.sqrt(eps[:, 1])) / C0
        return zz, 2.0 * _cumtrapz(slow, zz)

    def _antenna_time(self, z0, zz, tt):
        """Two-way time of an antenna at depth ``z0`` on the profile ``(zz, tt)``.

        A negative ``z0`` is an antenna above the snow surface, so that leg is
        in air and is timed at ``c``.  Interpolating the ice profile instead
        would clamp to the surface and silently drop the air path, which is
        6.7 ns for a metre of it -- comparable with the width of a reflection.
        """
        z0 = float(z0)
        if z0 < 0.0:
            return 2.0 * z0 / C0
        return float(np.interp(z0, zz, tt))

    def two_way_time(self, depth, z0=0.0, n=4001):
        """Two-way traveltime (s) from ``z0`` down to each depth (m).

        Integrates the actual slowness profile, so firn is handled properly.
        ``z0`` is the depth of the antenna: a sounder buried a few metres into
        the firn sees every return that much earlier, which is tens of ns and
        well inside the width of a reflection.  A negative ``z0`` is an antenna
        flown above the surface and adds the two-way air path.  Vectorised over
        ``depth``.
        """
        depth = np.asarray(depth, dtype=float)
        zmax = max(float(np.max(depth)) if depth.size else 1.0, float(z0))
        zz, tt = self._two_way_profile(zmax, n=n)
        return np.interp(depth, zz, tt) - self._antenna_time(z0, zz, tt)

    def depth_from_two_way_time(self, twt, z0=0.0, n=4001):
        """Depth (m) an echo at two-way time ``twt`` (s) came from.

        The inverse of :meth:`two_way_time`, and the only correct way to put a
        depth axis on a recorded section: firn is roughly 25 percent faster
        than solid ice, so converting at a single solid-ice velocity puts every
        event that travelled through it too shallow.
        """
        twt = np.asarray(twt, dtype=float)
        zz, tt = self._two_way_profile(self.thickness, n=n)
        tt = tt - self._antenna_time(z0, zz, tt)
        depth = np.interp(twt, tt, zz)
        # Past the bottom of the column, continue at the bed velocity rather
        # than flattening the axis onto the last sample.
        v_bed = (zz[-1] - zz[-2]) / (tt[-1] - tt[-2])
        return np.where(twt > tt[-1], zz[-1] + (twt - tt[-1]) * v_bed, depth)

    def birefringent_delay(self, depth, n=4001):
        """Two-way traveltime difference ``t_x - t_y`` (s) down to ``depth``.

        Normal incidence; both polarisations follow the same vertical ray,
        which is accurate because solid ice is only ~1 percent birefringent.
        The delay is one cumulative integral, so it is built once over the
        deepest requested depth and read off it.  Vectorised over ``depth``;
        the return has the shape of the argument.
        """
        depth = np.asarray(depth, dtype=float)
        zmax = float(np.max(depth)) if depth.size else 1.0
        zz = np.linspace(0.0, max(zmax, 1e-6), n)
        eps = self.permittivity(zz)
        slow = np.sqrt(eps[:, :2]) / C0
        tt = 2.0 * _cumtrapz(slow[:, 0] - slow[:, 1], zz)
        out = np.interp(depth, zz, tt)
        return out if depth.ndim else float(out)


#: Ridge A configuration used by the examples.
RIDGE_A = dict(
    thickness=1850.0,
    depth_bco=100.0,
    lam_z_sfc=1.0 / 3.0,
    lam_z_bed=0.55,
    dlambda=ridge_a_dlambda,
    sigma_ice=1.5e-5,
)
