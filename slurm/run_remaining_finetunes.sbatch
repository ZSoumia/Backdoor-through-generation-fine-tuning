#!/bin/bash
#SBATCH --job-name=ft-model-smoke
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%2
#SBATCH --output=logs/slurm/ft_model_%A_%a.out
#SBATCH --error=logs/slurm/ft_model_%A_%a.err

# 5 remaining models x (BitFit, Full): 10 independent one-step G0->G1 tests.
# Create logs/slurm on the submission node before calling sbatch.
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

[[ -s outputs/syntactic/ag_news.jsonl ]] || { echo 'Missing HiddenKiller table' >&2; exit 2; }

models=(
  'EleutherAI/pythia-1.4b'
  'EleutherAI/pythia-2.8b'
  'Qwen/Qwen2.5-1.5B'
  'HuggingFaceTB/SmolLM2-1.7B'
  'bert-base-uncased'
)
methods=('bitfit' 'full')

id="${SLURM_ARRAY_TASK_ID:?Must run as an SBATCH array}"
if (( id < 0 || id >= 10 )); then
  echo "Invalid task ID $id" >&2
  exit 2
fi
model="${models[id/2]}"
method="${methods[id%2]}"
export SMOKE_MODEL="$model" SMOKE_FINETUNE="$method"

# Refuse to run if the selected cell has changed or is not applicable.
python - <<'PY'
import os
from bdsurvive import plan
rows = plan.manifest_rows(plan.expand(plan.load_study('configs/study/section1_grid.yaml')))
match = [r for r in rows if r['model'] == os.environ['SMOKE_MODEL'] and
         r['attack'] == 'hiddenkiller' and r['finetune'] == os.environ['SMOKE_FINETUNE']]
assert len(match) == 1, f'Expected one matching grid row, got {len(match)}'
assert match[0]['grid_status'] == 'valid', match[0]
print('SELECTED:', match[0]['index'], match[0]['model'], match[0]['finetune'], flush=True)
PY

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

# One test per array task. Independent reports and fail status.
python -u -m tests.smoke_matrix \
  --study configs/study/section1_grid.yaml \
  --model "$model" \
  --attack hiddenkiller \
  --finetune "$method" \
  --plant-steps 1 \
  --benign-steps 1 \
  --batch-size 1 \
  --timeout 7200 \
  --out "smoke_runs/ft_model_${SLURM_ARRAY_JOB_ID}_${id}" \
  --execute --all

echo "SUCCESS: $model | hiddenkiller | $method"
