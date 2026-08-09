#!/bin/bash
#SBATCH --job-name=ex03-sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-8
# A dip x transition-width sweep: one array task per combination, each with
# its own --out directory so tasks cannot clobber one another's caches.
# Parameters are injected through the environment; the example reads its
# module constants, so the sweep patches them via a tiny driver instead of
# editing the file.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export MPLBACKEND=Agg

DIPS=(25 35 45)
WIDTHS=(0.3 1.2 2.4)
i=$((SLURM_ARRAY_TASK_ID / 3))
j=$((SLURM_ARRAY_TASK_ID % 3))
DIP="${DIPS[$i]}"
WIDTH="${WIDTHS[$j]}"
OUT="sweep_out/dip${DIP}_L${WIDTH}"

.venv/bin/python - "$DIP" "$WIDTH" "$OUT" <<'PYEOF'
import sys
sys.path.insert(0, "examples")
import ex03_dipping_fabric_transition as ex03
dip, width, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
ex03.DIP_DEG = dip
ex03.TRANSITION_WIDTH = width
ex03.main(bare=True, out=out)
PYEOF
