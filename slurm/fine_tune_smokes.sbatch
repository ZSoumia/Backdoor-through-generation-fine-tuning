#!/bin/bash
#SBATCH --job-name=finetune-smokes
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/finetune_smokes_%j.out
#SBATCH --error=logs/slurm/finetune_smokes_%j.err

# Run from the root of Backdoor-through-generation-fine-tuning.
# Create logs/slurm BEFORE sbatch submission (Slurm opens files first).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"

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

# Both poison tables have been generated; never regenerate or overwrite them here.
for table in outputs/syntactic/ag_news.jsonl outputs/bite/ag_news.jsonl; do
    if [[ ! -s "$table" ]]; then
        echo "ERROR: Missing/empty poisoning table: $table" >&2
        exit 2
    fi
done

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: no CUDA GPU detected')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

# Four isolated G0 -> G1 engineering smokes:
# pythia-410m x (hiddenkiller, bite) x (bitfit, full).
# Each uses 1 optimizer step for planting and 1 for benign adaptation.
# Each cell writes to an unused directory under smoke_runs/ and generates a
# PASS/FAIL report; nothing is written to production experiment outputs.
python -u -m tests.smoke_matrix \
    --study configs/study/section1_grid.yaml \
    --model EleutherAI/pythia-410m \
    --attack hiddenkiller \
    --attack bite \
    --finetune bitfit \
    --finetune full \
    --plant-steps 1 \
    --benign-steps 1 \
    --batch-size 2 \
    --timeout 3600 \
    --execute \
    --all

printf '\nSUCCESS: all four BitFit/full smoke experiments passed.\n'
