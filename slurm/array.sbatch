#!/bin/bash
#SBATCH --job-name=bdsurvive
#SBATCH --array=0-11%4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One array element = one grid cell = one generation chain.
# Do NOT split generations across array elements: generation N needs N-1's
# checkpoint. If a chain exceeds walltime, just resubmit -- state.json makes
# it resume from the last completed generation.

set -euo pipefail

MANIFEST="${MANIFEST:-manifest.jsonl}"
IMAGE="${IMAGE:-container/bdsurvive.sif}"
PROJECT="${PROJECT:-$PWD}"
CACHE="${CACHE:-$PWD/hf_cache}"

# Compute nodes usually have no internet: caches must be pre-populated and
# offline mode forced, or every job dies on a model download.
export HF_HOME="$CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

apptainer exec --nv \
    --bind "$PROJECT":/workspace \
    --bind "$CACHE":"$CACHE" \
    --env HF_HOME="$CACHE",HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1 \
    --pwd /workspace \
    "$IMAGE" \
    python -m bdsurvive.cli run \
        --manifest "$MANIFEST" \
        --index "$SLURM_ARRAY_TASK_ID"
