#!/bin/bash
#SBATCH --job-name=attack-smoke
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=logs/slurm/attack_other_%A_%a.out
#SBATCH --error=logs/slurm/attack_other_%A_%a.err

# Pythia-410M x (BadNets, MetaBackdoor) x (LoRA, BitFit, Full).
# Existing BITE/HiddenKiller smokes are not repeated.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo_work/try26_zuziak/bdsurvive-venv/bin/activate

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=/leonardo_work/try26_zuziak/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

attacks=('badnets' 'metabackdoor')
methods=('lora' 'bitfit' 'full')
id="${SLURM_ARRAY_TASK_ID:?Must run as an SBATCH array}"
if (( id < 0 || id >= 6 )); then
  echo "Invalid task ID $id" >&2
  exit 2
fi
attack="${attacks[id/3]}"
method="${methods[id%3]}"
export SMOKE_ATTACK="$attack" SMOKE_FINETUNE="$method"

python - <<'PY'
import os
from bdsurvive import plan
rows = plan.manifest_rows(plan.expand(plan.load_study('configs/study/section1_grid.yaml')))
match = [r for r in rows if r['model'] == 'EleutherAI/pythia-410m' and
         r['attack'] == os.environ['SMOKE_ATTACK'] and r['finetune'] == os.environ['SMOKE_FINETUNE']]
assert len(match) == 1, f'Expected one matching grid row, got {len(match)}'
assert match[0]['grid_status'] == 'valid', match[0]
print('SELECTED:', match[0]['index'], match[0]['attack'], match[0]['finetune'], flush=True)
PY

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

python -u -m tests.smoke_matrix \
  --study configs/study/section1_grid.yaml \
  --model EleutherAI/pythia-410m \
  --attack "$attack" \
  --finetune "$method" \
  --plant-steps 1 \
  --benign-steps 1 \
  --batch-size 1 \
  --timeout 7200 \
  --out "smoke_runs/attack_other_${SLURM_ARRAY_JOB_ID}_${id}" \
  --execute --all

echo "SUCCESS: EleutherAI/pythia-410m | $attack | $method"
