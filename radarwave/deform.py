"""Lagrangian deformation of a firn/ice column between repeat soundings.

A phase-sensitive sounder revisiting the same site does not measure the ice; it
measures how far the ice has *moved* since the last visit.  This module holds
the forward model for that motion: where a material marker - a density band, an
acidity horizon, the bed - sits after a given elapsed time, given where it
started.

THE FRAME.  Everything here is expressed in depth below the snow surface,
because that is the coordinate a surface-referenced radar works in.  Two things
move a marker in that coordinate, and they pull in opposite directions:

* **burial.**  Snow accumulates on top and the firn beneath compacts, so a
  marker sinks.  In a firn column in steady state the mass flux through every
  horizon is the same, ``rho(z) w(z) = rho_ice * b``, so a marker's
  ice-equivalent depth grows at exactly the accumulation rate ``b`` and its
  *actual* depth grows at ``b / rho_rel(z)``.  That is 0.57 m/yr at a 0.35
  relative-density surface and 0.2 m/yr in solid ice: strongly depth-dependent,
  and entirely a firn effect.
* **dynamic thinning.**  A vertical strain rate ``eps_zz`` shortens the whole
  column above the marker by ``eps_zz * z`` per year, lifting it back towards
  the surface.  This one grows linearly with depth and is what glaciology wants
  out of the measurement.

    dz/dt = b / rho_rel(z) + eps_zz * z

Below firn close-off the first term is a depth-independent constant and the
second is linear in depth, so the *shape* of the displacement profile with
depth is the strain signal and the constant is not.  Near the surface the two
have opposite sign and comparable size, and the profile can cross zero: shallow
markers sink while deep ones rise.  That crossing is real, not a modelling
artefact, and it is the most legible thing a repeat sounding shows.

WHAT IS HELD FIXED.  The firn column is treated as being in steady state, so
the *mean* density field - and with it the permittivity, the wave speed and the
snow surface itself - is the same at both epochs.  Only the material markers
move through it.  That is what makes the epochs comparable: the antenna sits at
an identical position over an identical velocity structure at every visit, and
every difference between the records comes from the deformation.  Density
*banding* is a material property riding on top of that mean profile, so it
advects with the ice like everything else.

The steady-state assumption is what buys this.  A transient firn column - a
warm summer, a change in accumulation - would change the velocity structure
between epochs too, and a real repeat sounding cannot separate that from
deformation without an independent constraint.  Nothing here models it.
"""

from dataclasses import dataclass

import numpy as np

from .ice import IceColumn

__all__ = ["VerticalStrain"]


@dataclass
class VerticalStrain:
    """Steady firn burial plus a uniform dynamic vertical strain rate.

    Parameters
    ----------
    column : IceColumn
        Supplies the density profile, which sets the burial rate, and the
        slowness profile, which converts a displacement into a delay.
    accumulation : float
        Surface mass balance in metres of ice equivalent per year.
    strain_rate : float
        Vertical strain rate ``eps_zz`` per year, uniform with depth.  Negative
        is thinning (vertical compression), which is the usual sign.
    substeps : int
        Runge-Kutta substeps per year used to integrate the trajectory.  The
        equation is mildly nonlinear through ``rho(z)``, so a single Euler step
        is only good to a percent or so over a year and worse over a decade;
        integrating properly makes ``years`` an honest parameter rather than
        one that quietly has to stay small.
    """

    column: IceColumn
    accumulation: float = 0.20
    strain_rate: float = -4.0e-3
    substeps: int = 64

    def burial_rate(self, depth):
        """``dz/dt`` of a material marker at depth ``z`` (m/yr, down positive)."""
        depth = np.asarray(depth, dtype=float)
        rho = self.column.density(np.maximum(depth, 0.0))
        return self.accumulation / rho + self.strain_rate * depth

    def displaced_depth(self, depth, years):
        """Depth of the same marker ``years`` later (m).

        Integrated with fixed-step RK4 rather than evaluated as
        ``depth + rate * years``: the rate depends on the depth the marker is
        currently at, and in firn that dependence is steep.
        """
        z = np.array(depth, dtype=float, copy=True)
        # A floor of four substeps, not one: :meth:`trajectory` calls this with
        # short segments, and a single RK4 step per segment would make the path
        # depend on how finely it happened to be sampled.
        n = max(4, int(round(self.substeps * abs(float(years)))))
        h = float(years) / n
        for _ in range(n):
            k1 = self.burial_rate(z)
            k2 = self.burial_rate(z + 0.5 * h * k1)
            k3 = self.burial_rate(z + 0.5 * h * k2)
            k4 = self.burial_rate(z + h * k3)
            z = z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return z

    def displacement(self, depth, years):
        """How far the marker moved (m); positive is deeper."""
        return self.displaced_depth(depth, years) - np.asarray(depth, dtype=float)

    def trajectory(self, depth, years):
        """Displacement of one marker at each of several times (m).

        :meth:`displaced_depth` integrates to a single time.  A repeat-pass
        *series* wants the whole path, and calling that method once per sample
        would re-integrate the same trajectory from scratch each time - the
        same work done ``n`` times over, and ``n`` independent truncation
        errors where there should be one path.  This integrates once and
        samples as it goes.

        ``years`` must be non-negative; the samples may be in any order.
        """
        years = np.atleast_1d(np.asarray(years, dtype=float))
        if np.any(years < 0.0):
            raise ValueError("trajectory runs forwards from the first visit; "
                             "years must be non-negative")
        order = np.argsort(years)
        out = np.empty(years.shape, dtype=float)
        z, elapsed = float(depth), 0.0
        for i in order:
            z = float(self.displaced_depth(z, float(years[i]) - elapsed))
            elapsed = float(years[i])
            out[i] = z - float(depth)
        return out

    def stretch(self, depth, years, step=0.25):
        """``d(displaced)/d(depth)``: how a layer's thickness scales.

        A layer is a finite interval, not a point, so the two ends move by
        different amounts and the layer thins or thickens.  Over a year in the
        upper firn that is most of a percent - far below what any of these
        traces resolves, but it costs one extra evaluation to carry it and it
        keeps a decade-long baseline from silently accumulating an error.

        Central-differenced with a step wide enough that the difference is not
        dominated by the RK4 truncation error at either end.
        """
        depth = np.asarray(depth, dtype=float)
        lo = np.maximum(depth - step, 0.0)
        hi = depth + step
        return (self.displaced_depth(hi, years) - self.displaced_depth(lo, years)) / (hi - lo)

    def delay(self, depth, years, z0=0.0):
        """Change in two-way traveltime to the marker (s), later minus earlier.

        Both traveltimes are integrated through the *same* slowness profile,
        which is the steady-state assumption in the module docstring doing its
        work: the marker moves, the medium does not, so the delay is purely
        geometric.  Negative means the echo arrives earlier at the later visit,
        i.e. the marker came up.
        """
        depth = np.asarray(depth, dtype=float)
        moved = self.displaced_depth(depth, years)
        return self.column.two_way_time(moved, z0=z0) - self.column.two_way_time(depth, z0=z0)

    def phase(self, depth, years, fc, z0=0.0, wrap=True):
        """Interferometric phase (rad) the delay produces at ``fc``.

        Signed to match :func:`radarwave.polarimetry.interferogram` called as
        ``interferogram(later, earlier)``, whose phase is
        ``2 pi fc (t_earlier - t_later) = -2 pi fc * delay``.  Getting that sign from one
        place is the point of the method: a repeat-pass product that flips it
        reports thickening as thinning and the profile still looks plausible.
        """
        phi = -2.0 * np.pi * fc * self.delay(depth, years, z0=z0)
        return np.angle(np.exp(1j * phi)) if wrap else phi
