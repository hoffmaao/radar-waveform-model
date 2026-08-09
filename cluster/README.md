# Running radarwave on a cluster (CReSIS-style Linux)

The package is pure Python (numpy/scipy/matplotlib), single-node by design.
The pattern is **compute remote, render local**: heavy runs write parameter-
stamped `.npz` caches, you copy those back, and `--render-only` regenerates
every figure and movie locally without re-simulating.  A cache produced with
different physics constants is refused with the mismatched parameter named,
so remote and local results cannot silently mix.

## Deploy

```sh
git clone https://github.com/hoffmaao/radar-waveform-model
cd radar-waveform-model
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest -q          # 75 tests; verifies the install
```

Requirements: Python >= 3.9.  Set `MPLBACKEND=Agg` on headless nodes (the
sbatch templates do).  Movies need ffmpeg via the bundled imageio-ffmpeg
wheel; if a node lacks it, skip movie rendering remotely and render locally
from the cache instead.

## Single full-scale run

```sh
sbatch cluster/run_example.sbatch ex03_dipping_fabric_transition "--bare --radargram --processes 10"
```

## Parameter sweeps

Every example takes `--out DIR`.  That flag is what makes array jobs safe:
without it each task writes the same `figures/exNN` paths and the sweep
destroys its own results.  `cluster/sweep_ex03.sbatch` shows the pattern -
one array task per parameter combination, each writing to its own directory.

## Bringing results home

```sh
rsync -av cluster-host:path/to/sweep_out/ figures/sweeps/
.venv/bin/python examples/ex03_dipping_fabric_transition.py --render-only --out figures/sweeps/<combo>
```

## The other codes

- The CReSIS `em_model` bridge (`validation/emmodel_driver.m`) runs wherever
  the OPR toolbox lives - the fabric_anisotropy MATLAB container locally, or
  CReSIS machines natively.  Only `emmodel_candidates.mat` travels out and
  `emmodel_results.mat` back.
- The gprMax environment is deliberately disposable; rebuild it rather than
  transporting it (micromamba python=3.11 + numpy scipy cython h5py, clone
  github.com/gprMax/gprMax, `setup.py build_ext --inplace` with an OpenMP
  gcc).  On GPU nodes gprMax's CUDA path is worth enabling for 3-D runs.
