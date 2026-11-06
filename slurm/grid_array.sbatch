#!/bin/bash
#SBATCH --job-name=sok-grid
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.err

set -euo pipefail

: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch --array=0-N}"

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
MANIFEST="${MANIFEST:-manifest.jsonl}"
CINECA_AI_MODULE="${CINECA_AI_MODULE:-cineca-ai/4.3.0}"

module load profile/deeplrn
module load "$CINECA_AI_MODULE"

cd "$CODE_DIR"

export PYTHONPATH="$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-${WORK:?WORK must be set}/hf_cache}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ ! -f "$MANIFEST" ]]; then
    echo "Missing manifest: $MANIFEST" >&2
    exit 2
fi

python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA device not available'
print('GPU:', torch.cuda.get_device_name(0))
PY

python -m bdsurvive.cli run \
    --manifest "$MANIFEST" \
    --index "$SLURM_ARRAY_TASK_ID"
