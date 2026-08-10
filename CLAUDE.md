# radar-waveform-model - agent guide

2-D FDTD modelling of radar waves in polar ice, including crystal-fabric
anisotropy. Built for Andrew's SCAR talk on measuring ice fabric with radar;
the examples ARE the talk figures. The dielectric model mirrors
`~/projects/fabric_anisotropy/+ptt` (eps_bar = 3.171, deps = 0.034,
Herron-Langway, Maxwell-Garnett) - keep them in sync or results stop being
comparable with the Ridge A inversion.

## Commands

```sh
.venv/bin/python -m pytest -q            # 80 tests, ~3 min - run before any commit
.venv/bin/python -m pyflakes radarwave examples validation tests
.venv/bin/python examples/ex0N_*.py [--quick] [--render-only] [--bare] [--out DIR]
```

- `--quick`: pipeline check only. NEVER ship quick outputs - a quick artifact
  left in `figures/` has twice been mistaken for the real thing.
- `--render-only`: re-render figures/movies from the stamped `.npz` cache
  without re-simulating. A `StaleCache` error means the cache came from
  different physics - that is the guard working; delete the cache or re-run,
  never weaken the stamp.
- `--bare`: strip text for slides (axes and colourbars stay). Send ANNOTATED
  versions when the owner is reviewing; bare figures have been misread.
- `--out DIR`: redirect all outputs+caches (sweeps, cluster array jobs).
- Movies render straight to MP4; matplotlib needs `MPLBACKEND=Agg` headless.

## Runtimes (14-core M-series, full scale)

ex01 ~10 min, +2.4 h with `--radargram`; ex02 ~45 min; ex03 ~15 min, +50 min
radargram; ex04 ~80 min (4 runs). Cluster kit in `cluster/`.

**Long runs**: this environment reaps harness background tasks. Launch via
`nohup script > log 2>&1 &` + `disown` in a FOREGROUND call (write a tiny
launcher script), verify the pid detached, and poll with expendable checks.
Never combine launch and wait in one call - two multi-hour sims died that way.

## Gating and repo conventions

- All commits go through `/no-mistakes` on a feature branch. Write the intent
  COMPLETE: every deliberate choice a diff-only reviewer would question
  (figures absent, narrowband pulses, unrun MATLAB, etc.) or it gets flagged.
  The owner's standing pattern is "fix all findings"; ask-user findings that
  change slide-facing numbers still get relayed first.
- `figures/**` outputs are gitignored: regenerate once from FINAL code after
  the gate lands, full resolution. Caches carry parameter stamps.
- Owner preferences: MP4 not PNG series; quantities side by side, not
  differences; no em dashes anywhere.

## What is validated (do not re-derive)

`validation/` holds the suite; numbers measured 2026-08:

- Image-source identity: radarwave +0.38 dB / 0.49 ns (dx 0.15, converging).
- Thin layer vs exact transfer matrix: radarwave +0.20 dB, gprMax -0.80 dB.
- Birefringent split: analytic 4.582 ns; radarwave 4.576/4.580 (TM-sub / TE,
  identical peaks - the eps_xx-into-TM substitution is EXACT at nadir);
  gprMax (native tensor) 4.750.
- Full ex03 vs gprMax: every event at the same time; trace-level dB across
  codes is NOT well-posed (each code's line-source wake rides on its own
  source signature) - score events or use the exact tests.
- gprMax export traps are documented in `validation/gprmax_export.py`
  docstrings; the worst is that PropertyGrid arrays sample at HALF-cell
  resolution including PML pad - export `arr[1::2, 1::2]` and place sources
  from `model.x/z`, never the nominal limits.
- **Open**: radarwave TE gives a 39 dB weaker oblique in-plane return in
  ex03's 35 deg geometry than scalar effective-index algebra predicts (9 dB);
  nadir tests exonerate the TE solver. Only 3-D gprMax or proper
  extraordinary-wave theory can settle it. ex03/ex04 bright-pol amplitudes
  off nadir carry this caveat.

## Physics results the examples rest on

- Single fabric interface ceiling: r = deps/(4 eps) ~ -51 dB; below firn
  close-off fabric is the STRONGEST permittivity reflector (acidity -66 to
  -70 dB, density ~nothing). Gradational width is brutal: ~10-15 dB per
  doubling (0.6 m costs -14.5 dB at 60 MHz; 1.2 m is invisible at 195 MHz).
- Banded fabric (ex04): N Bragg-tuned bands beat the ceiling by ~20 log(2N);
  resonance is ~1/N wide so an IMPULSE cannot see it - band-limited pulses
  (12 percent Gabor = one MCoRDS sub-band) are required. Periodicity, not
  sharpness, is the requirement (a sinusoidal oscillation loses 1.5 dB);
  harmonics encode band shape. Fingerprint survives full 31-percent MCoRDS
  bandwidth at ~+12 dB.
- The CReSIS/OPR EM code lives at `~/projects/opr/matlab/em_model` (+sim for
  the chirp chain). Isotropic only - run fabric per-eigenpolarisation (the
  validated nadir substitution). `validation/emmodel_driver.m` runs in the
  fabric_anisotropy MATLAB container; the `physicalConstants` shim next to it
  is required (the original lives on a dead Kansas share).

## Restart checklist

1. `git log --oneline -5` and `git status` - the auto-memory
   (`ex04-banded-fabric-state-and-decisions` and friends) records what was in
   flight last; verify against the tree rather than trusting either alone.
2. `no-mistakes axi` - reattach or respond to any parked run before
   committing anything.
3. `gh pr list` - green PRs may be waiting on the owner's merge.
4. Check `figures/*/` timestamps against `git log` before believing any
   figure reflects current code.
