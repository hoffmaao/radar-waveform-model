#!/bin/bash
#SBATCH --job-name=radarwave
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
# Usage: sbatch cluster/run_example.sbatch <example-stem> "<flags>"
# e.g.:  sbatch cluster/run_example.sbatch ex01_meteoric_layers "--bare --radargram --processes 10"
set -euo pipefail
EXAMPLE="${1:?example stem, e.g. ex03_dipping_fabric_transition}"
FLAGS="${2:-}"
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg
exec .venv/bin/python "examples/${EXAMPLE}.py" ${FLAGS}
