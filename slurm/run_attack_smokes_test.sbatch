#!/bin/bash
#SBATCH --job-name=table-smokes
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=table_smokes_%j.out
#SBATCH --error=table_smokes_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"

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

# These are the poisoning tables already generated and validated on Leonardo.
for file in outputs/syntactic/ag_news.jsonl outputs/bite/ag_news.jsonl; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Missing or empty poisoning table: $file" >&2
        exit 2
    fi
done

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA GPU is not available')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

# Existing repository harness: isolated G0 -> G1 runs using the real CLI.
# Only one model, two attacks, LoRA; one optimizer step for planting and G1.
# Results are engineering smoke tests, not estimates of backdoor survival.
python -u -m tests.smoke_matrix \
    --study configs/study/section1_grid.yaml \
    --model EleutherAI/pythia-410m \
    --finetune lora \
    --attack hiddenkiller \
    --attack bite \
    --plant-steps 1 \
    --benign-steps 1 \
    --batch-size 2 \
    --timeout 3600 \
    --execute \
    --all

printf '\nSUCCESS: both table-attack smoke experiments passed.\n'
