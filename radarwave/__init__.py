"""Two-dimensional FDTD modelling of radar waves in polar ice.

The package started as a Python port of the ground-penetrating-radar FDTD code
of Irving and Knight (2006) and has been extended to handle the anisotropic,
depth-varying dielectric structure of an ice sheet, so that the same solver can
reproduce

* reflections from meteoric internal layering,
* birefringent splitting between the two eigenpolarisations of ice with a
  horizontal crystal-orientation fabric, and
* off-nadir returns from a dipping fabric transition.

See ``examples/`` for the three scenarios.
"""

from .constants import C0, DEPS_ICE, EPS0, EPS_ICE_MEAN, MU0, V_ICE
from .fdtd import FDTD2D, Result, run_common_offset
from .grid import PropertyGrid, max_spatial_step, max_time_step, pad_grid, regrid
from .ice import IceColumn, RIDGE_A, ridge_a_dlambda
from .sources import (
    blackharrispulse,
    dominant_frequency,
    envelope_peak_time,
    gabor,
    gaussian_derivative,
    ricker,
)

__version__ = "0.2.0"

__all__ = [
    "C0",
    "DEPS_ICE",
    "EPS0",
    "EPS_ICE_MEAN",
    "FDTD2D",
    "IceColumn",
    "MU0",
    "PropertyGrid",
    "RIDGE_A",
    "Result",
    "V_ICE",
    "blackharrispulse",
    "dominant_frequency",
    "envelope_peak_time",
    "gabor",
    "gaussian_derivative",
    "max_spatial_step",
    "max_time_step",
    "pad_grid",
    "regrid",
    "ricker",
    "ridge_a_dlambda",
    "run_common_offset",
]
