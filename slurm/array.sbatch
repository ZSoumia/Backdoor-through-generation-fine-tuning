#!/bin/bash
# Leonardo Booster (CINECA) submission for scripts/lineage_runner.py
#
# ONE FILE for every generational experiment. What runs is controlled by
# environment variables at submit time, never by editing this file:
#
#   GENERATIONS=3 SEEDS=0,1,2 MECHANISMS=rare,semantic TAG=g3 \
#       sbatch slurm/array.sbatch
#
# Submitted without --array it runs the whole mechanism x seed sweep in one
# job, which is right while a 3-generation chain takes ~15 min. Submitted WITH
# --array it runs one chain per array element, which is what Section 1's grid
# will need:
#
#   # 2 mechanisms x 3 seeds = 6 chains, 2 at a time
#   GENERATIONS=3 SEEDS=0,1,2 MECHANISMS=rare,semantic TAG=g3 \
#       sbatch --array=0-5%2 slurm/array.sbatch
#
# One array element is always one CHAIN, never one generation: generation N
# needs N-1's merged checkpoint, so generations cannot run in parallel.

#SBATCH --job-name=bdsurvive
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/%x_%j.out
#SBATCH --error=logs/slurm/%x_%j.err

# partition: boost_usr_prod is mandatory. Leonardo's DEFAULT partition is
#   lrd_all_serial, which has NO GPUs -- omit this line and the job silently
#   lands on CPU.
#
# qos: boost_qos_dbg = 30 min, max 8 nodes, max 2 jobs/user. Fine for an
#   environment check or a short lineage; NOT enough to plant (see PLANT
#   below). For real runs switch to:
#     normal           24h,    up to 64 nodes
#     boost_qos_lprod  4 days, max 8 nodes / 32 GPUs per user
#   and raise --time to match.
#
# cpus/mem: do not raise these without reason. Leonardo bills
#   B_H = T * N * R * C, where R is the LARGEST fraction of any reserved
#   resource on the node and C = 32 CPUs. With 1 of 4 GPUs (0.25), 8 of 32
#   CPUs (0.25) and 48 of ~512 GB (0.09), R = 0.25 and an hour costs 8 CPUh.
#   Asking for 16 CPUs would push R to 0.5 and DOUBLE the cost for no extra
#   GPU. The three fractions are deliberately balanced at a quarter node.
#
# output: %j (job id) not %A_%a. For a non-array job Slurm has no array index
#   to substitute and writes a sentinel, giving files like
#   bdsurvive_12345_4294967294.out. %j is correct for both plain and array
#   jobs.

set -euo pipefail

# cineca-ai 4.3.0 ships torch 2.2.0a0, transformers 4.38.0, peft 0.10.0,
# datasets 2.18.0 -- the stack this code is pinned against.
CINECA_AI_MODULE="${CINECA_AI_MODULE:-cineca-ai/4.3.0}"

# ------------------------------------------------------------------ settings
GENERATIONS="${GENERATIONS:-1}"
SEEDS="${SEEDS:-0}"
MECHANISMS="${MECHANISMS:-rare}"
TAG="${TAG:-smoke}"
FINETUNE="${FINETUNE:-lora}"     # lora | bitfit | full  -- the Section 1 axis
PLANT="${PLANT:-0}"              # 1 to plant a parent that is missing
CONTINUOUS="${CONTINUOUS:-0}"    # 1 to also run the budget-matched control

CODE_DIR="${CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
VENV="${VENV:-$WORK/bdsurvive-venv}"
export BDSURVIVE_ROOT="${BDSURVIVE_ROOT:-$WORK/bdsurvive_runs}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"

# $HOME is 50 GB and backed up -> code. $WORK is 1 TB per project and not
# backed up -> venv, HF cache, checkpoints, results. $SCRATCH deletes anything
# untouched for 40 days, so nothing that must survive a submission cycle goes
# there.


# ------------------------------------------------- array mode: one chain each
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if [[ "$MECHANISMS" == "all" ]]; then
        echo "ERROR: --array needs MECHANISMS named explicitly, not 'all'," >&2
        echo "       so an index maps to a known (mechanism, seed) pair." >&2
        exit 2
    fi
    IFS=',' read -ra _MECHS <<< "$MECHANISMS"
    IFS=',' read -ra _SEEDS <<< "$SEEDS"
    _n_seeds=${#_SEEDS[@]}
    _total=$(( ${#_MECHS[@]} * _n_seeds ))
    if (( SLURM_ARRAY_TASK_ID >= _total )); then
        echo "ERROR: array index $SLURM_ARRAY_TASK_ID >= $_total chains." >&2
        exit 2
    fi
    MECHANISMS="${_MECHS[$(( SLURM_ARRAY_TASK_ID / _n_seeds ))]}"
    SEEDS="${_SEEDS[$(( SLURM_ARRAY_TASK_ID % _n_seeds ))]}"
    echo "array element $SLURM_ARRAY_TASK_ID of $_total -> $MECHANISMS seed $SEEDS"
fi

# Planting is 1500 steps, five times one generation -- and any mechanism with
# kl_lambda != 0 trains a 1500-step clean reference first, so ~10x a
# generation before the lineage starts. That does not reliably fit 30 minutes.
if [[ "$PLANT" == "1" && "${SLURM_JOB_QOS:-}" == "boost_qos_dbg" ]]; then
    echo "WARNING: PLANT=1 under boost_qos_dbg (30 min cap)." >&2
    echo "         Planting may be killed mid-training. Use --qos=normal" >&2
    echo "         with --time=04:00:00 for planting runs." >&2
fi

# ----------------------------------------------------------------- environment
# No 'module purge': a base profile is part of the default environment on
# CINECA and purging it before loading another profile leaves the module
# system half-configured.
module load profile/deeplrn
module load "$CINECA_AI_MODULE"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: no venv at $VENV. Create it on a login node first:" >&2
    echo "  module load profile/deeplrn && module load $CINECA_AI_MODULE" >&2
    echo "  python -m venv $VENV --system-site-packages" >&2
    echo "  source $VENV/bin/activate && pip install -r requirements.txt" >&2
    exit 1
fi
source "$VENV/bin/activate"

# Compute nodes have no outbound internet. The cache must already be populated
# by scripts/prefetch.py on a login node, or every job dies on a download.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$CODE_DIR"
export PYTHONPATH="$CODE_DIR"

# ------------------------------------------------------------- what we're on
echo "=== job ${SLURM_JOB_ID:-?} on $(hostname) at $(date -Is)"
echo "account    : ${SLURM_JOB_ACCOUNT:-?}   qos: ${SLURM_JOB_QOS:-?}"
echo "finetune   : $FINETUNE"
echo "code       : $CODE_DIR"
echo "venv       : $VENV"
echo "runs root  : $BDSURVIVE_ROOT"
echo "hf cache   : $HF_HOME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PYCHECK'
import torch
print(f"torch       : {torch.__version__}")
print(f"cuda avail  : {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("ERROR: no GPU visible. Check --partition=boost_usr_prod "
                     "and --gres=gpu:1 -- the default partition has no GPUs.")
print(f"device      : {torch.cuda.get_device_name(0)}")
PYCHECK

# ------------------------------------------------------------------- the run
# No 'srun': this is a single-task, single-GPU, non-MPI job. srun adds nothing
# and can re-initialise the environment, which is how an activated venv or an
# exported HF_HOME gets silently lost between this shell and the process.
ARGS=(--generations "$GENERATIONS" --seeds "$SEEDS" --mechanisms "$MECHANISMS"
      --tag "$TAG" --finetune "$FINETUNE" --device cuda)
[[ "$PLANT" == "1" ]] && ARGS+=(--plant)
[[ "$CONTINUOUS" == "1" ]] && ARGS+=(--continuous)

echo "=== python scripts/lineage_runner.py ${ARGS[*]}"
python scripts/lineage_runner.py "${ARGS[@]}"

echo "=== finished at $(date -Is)"
