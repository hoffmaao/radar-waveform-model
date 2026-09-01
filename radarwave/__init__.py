"""Two-dimensional FDTD modelling of radar waves in polar ice.

The package started as a Python port of the ground-penetrating-radar FDTD code
of Irving and Knight (2006) and has been extended to handle the anisotropic,
depth-varying dielectric structure of an ice sheet, so that the same solver can
reproduce

* reflections from meteoric internal layering,
* birefringent splitting between the two eigenpolarisations of ice with a
  horizontal crystal-orientation fabric,
* off-nadir returns from a dipping fabric transition, and
* the phase change a chirped sounder sees when it revisits the same ice year
  after year and the ice has deformed underneath it.

See ``examples/`` for the scenarios.
"""

from .constants import C0, DEPS_ICE, EPS0, EPS_ICE_MEAN, MU0, V_ICE
from .deform import VerticalStrain
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
from .waveform import ACCUM3, APRES, RadarSystem, lfm_chirp, range_compress

__version__ = "0.2.0"

__all__ = [
    "ACCUM3",
    "APRES",
    "C0",
    "DEPS_ICE",
    "EPS0",
    "EPS_ICE_MEAN",
    "FDTD2D",
    "IceColumn",
    "MU0",
    "PropertyGrid",
    "RIDGE_A",
    "RadarSystem",
    "Result",
    "V_ICE",
    "VerticalStrain",
    "blackharrispulse",
    "dominant_frequency",
    "envelope_peak_time",
    "gabor",
    "gaussian_derivative",
    "lfm_chirp",
    "max_spatial_step",
    "max_time_step",
    "pad_grid",
    "range_compress",
    "regrid",
    "ricker",
    "ridge_a_dlambda",
    "run_common_offset",
]
