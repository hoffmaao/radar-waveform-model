"""Export candidate interface models for the CReSIS/OPR ``em_model`` library.

The question this serves: what IS the deep Ridge A feature?  Each candidate
here is a physical hypothesis for it, expressed as a thin-layered permittivity
profile that ``specularNadir(depth, er, freq)`` (in ``opr/matlab/em_model``)
can evaluate across the MCoRDS band.  Their code is isotropic by design --
"birefringence not supported" -- but at nadir each eigenpolarisation
propagates through an isotropic profile of its own tensor component, the
substitution proved exact against gprMax and the analytic split in this
repository's validation suite.  So every candidate is exported twice, as
``er_x(z)`` and ``er_y(z)``, and the polarimetric contrast is the difference
of two isotropic runs.

The profiles embed the structure between uniform ice half-spaces, so the
MATLAB side returns the *local* reflection response of the candidate, directly
comparable with the transfer-matrix fingerprints computed here.  Full-column
responses (firn, attenuation, temperature) are a later step and need
``iceAcid``'s data file, which its hardcoded PRISM path no longer locates.

Conductivity enters through the imaginary permittivity, ``er'' = sigma /
(2 pi f eps0)``, which is why ``er`` is exported per frequency even though the
permittivity candidates are frequency-flat.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from radarwave.constants import C0, EPS0                      # noqa: E402
from radarwave.ice import fabric_permittivity                  # noqa: E402

OUT = REPO / "figures" / "validation"

# Ridge A-like numbers: background weak fabric over a candidate structure.
LAM_ABOVE = np.array([0.115, 0.085, 0.80])
LAM_BELOW = np.array([0.860, 0.060, 0.08])
SIG_ICE = 1.2e-5          # S/m, background
DZ = 0.05                 # profile discretisation (m)
HALF = 30.0               # half-space thickness represented each side (m)
FREQ = np.arange(150e6, 251e6, 2e6)


def _profile(z, frac):
    """eps_x(z), eps_y(z) blending ABOVE -> BELOW by ``frac(z)`` in [0, 1]."""
    ea = fabric_permittivity(LAM_ABOVE)
    eb = fabric_permittivity(LAM_BELOW)
    fx = np.asarray(frac, dtype=float)[:, None]
    eps = ea[None, :] + (eb - ea)[None, :] * fx
    return eps[:, 0], eps[:, 1]


def candidates():
    """Yield (name, z, eps_x, eps_y, sigma) thin-layer stacks."""
    z = np.arange(-HALF, HALF, DZ)
    sig0 = np.full(z.size, SIG_ICE)

    yield ("sharp_fabric_flip", z, *_profile(z, z >= 0), sig0)
    for L in (0.6, 1.2, 2.4):
        frac = 0.5 * (1.0 + np.tanh(z / L))
        yield (f"gradational_L{L:g}", z, *_profile(z, frac), sig0)

    # Banded stack tuned inside the MCoRDS band: period = v / (2 f0).
    ea = fabric_permittivity(LAM_ABOVE)
    v = C0 / np.sqrt(0.5 * (ea[0] + fabric_permittivity(LAM_BELOW)[0]))
    period = v / (2 * 195e6)
    n_band = 20
    frac = ((z >= 0) & (z < n_band * period)
            & ((np.floor(z / (period / 2)).astype(int) % 2) == 0)).astype(float)
    yield (f"banded_N{n_band}_d{period:.2f}m", z, *_profile(z, frac), sig0)

    # Acidity spike: permittivity flat, conductivity boosted over 2 m.
    ex, ey = _profile(z, np.zeros(z.size))
    sig = np.where((z >= 0) & (z < 2.0), 20.0 * SIG_ICE, SIG_ICE)
    yield ("acidity_spike_20x", z, ex, ey, sig)


def main():
    from scipy.io import savemat

    OUT.mkdir(parents=True, exist_ok=True)
    out = {"freq": FREQ.reshape(1, -1)}
    names = []
    for name, z, ex, ey, sig in candidates():
        names.append(name)
        # er'' from conductivity, per frequency; er matrices are
        # (n_layers, n_freq) as specularNadir expects.
        loss = sig[:, None] / (2 * np.pi * FREQ[None, :] * EPS0)
        depth = np.concatenate([[0.0], z - z[0] + DZ]).reshape(-1, 1)
        out[f"{name}__depth"] = depth
        out[f"{name}__er_x"] = ex[:, None] - 1j * loss
        out[f"{name}__er_y"] = ey[:, None] - 1j * loss
    out["names"] = np.array(names, dtype=object)
    path = OUT / "emmodel_candidates.mat"
    savemat(path, out)
    print(f"wrote {path} ({len(names)} candidates, "
          f"{FREQ.size} frequencies {FREQ[0]/1e6:.0f}-{FREQ[-1]/1e6:.0f} MHz)")
    for n in names:
        print("  -", n)


if __name__ == "__main__":
    main()
