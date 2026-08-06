"""Physical constants used throughout the radar waveform model.

Ice dielectric values follow Fujita et al. (2000) as parameterised by
Rathmann (2026, eq. 2.3).  They are deliberately identical to the constants
used by the polarimetric traveltime toolbox (``+ptt/constants.m``) so that a
forward FDTD simulation and the inversion applied to real Open Polar Radar
data share one dielectric model.
"""

import numpy as np

#: Permittivity of free space (F/m).
EPS0 = 8.8541878176e-12

#: Permeability of free space (H/m).
MU0 = 1.2566370614e-6

#: Speed of light in vacuum (m/s).
C0 = 1.0 / np.sqrt(EPS0 * MU0)

#: Mean relative permittivity of an ice grain at radar frequencies.
EPS_ICE_MEAN = 3.171

#: Dielectric anisotropy of an ice grain, eps_c - eps_a.
DEPS_ICE = 0.034

#: Relative permittivity of air.
EPS_AIR = 1.0

#: Density of solid ice (kg/m^3).
RHO_ICE = 917.0

#: Isotropic-ice relative permittivity, i.e. eigenvalues all equal to 1/3.
EPS_ICE_ISOTROPIC = EPS_ICE_MEAN

#: Radio-wave speed in isotropic solid ice (m/s).
V_ICE = C0 / np.sqrt(EPS_ICE_MEAN)
