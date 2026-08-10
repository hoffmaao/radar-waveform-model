% Shim for the PRISM-era physicalConstants script that specularNadir calls.
%
% The original lived on a Kansas network share
% (/projects/prism/radar/radarSimulator/physicalConstants.m) that the OPR
% clone does not carry; specularNadir path()-appends that location and then
% invokes the script, so without this file it errors on any machine outside
% that filesystem.  Values are CODATA; only u0/e0/c are consumed.
u0 = 4e-7 * pi;              % vacuum permeability (H/m)
e0 = 8.8541878128e-12;       % vacuum permittivity (F/m)
c  = 299792458;              % speed of light (m/s)
