#!/bin/bash
#SBATCH --job-name=model-lora-smokes
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/model_lora_smokes_%j.out
#SBATCH --error=logs/slurm/model_lora_smokes_%j.err

# Five remaining Section 1 models, using one-step HiddenKiller + LoRA G0 -> G1
# engineering smokes. No production experiments or poisoning tables are changed.
# Submit from the repository root; mkdir -p logs/slurm BEFORE calling sbatch.
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

if [[ ! -s outputs/syntactic/ag_news.jsonl ]]; then
    echo 'ERROR: HiddenKiller table missing/empty: outputs/syntactic/ag_news.jsonl' >&2
    exit 2
fi

# Prevent accidentally running the wrong grid cells if the study file changes.
python - <<'PY'
from bdsurvive import plan
wanted = {
    'EleutherAI/pythia-1.4b',
    'EleutherAI/pythia-2.8b',
    'Qwen/Qwen2.5-1.5B',
    'HuggingFaceTB/SmolLM2-1.7B',
    'bert-base-uncased',
}
rows = plan.manifest_rows(plan.expand(plan.load_study('configs/study/section1_grid.yaml')))
selected = [r for r in rows if r['model'] in wanted and r['attack'] == 'hiddenkiller' and r['finetune'] == 'lora']
assert len(selected) == len(wanted), f'Expected 5 cells, found {len(selected)}'
assert {r['model'] for r in selected} == wanted, 'Model list changed'
for r in selected:
    assert r['grid_status'] == 'valid', (r['index'], r['model'], r['grid_status'], r['reason'])
    print(f"READY: section1_grid[{r['index']}] {r['model']} | hiddenkiller | lora", flush=True)
PY

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA GPU not visible')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

# Five sequential, isolated G0 -> G1 runs (one optimizer step per phase).
# Use batch size 1 to reduce peak GPU memory for the larger models.
# Per-cell stdout/stderr and PASS/FAIL report are written under --out.
python -u -m tests.smoke_matrix \
    --study configs/study/section1_grid.yaml \
    --model EleutherAI/pythia-1.4b \
    --model EleutherAI/pythia-2.8b \
    --model Qwen/Qwen2.5-1.5B \
    --model HuggingFaceTB/SmolLM2-1.7B \
    --model bert-base-uncased \
    --attack hiddenkiller \
    --finetune lora \
    --plant-steps 1 \
    --benign-steps 1 \
    --batch-size 1 \
    --timeout 5400 \
    --out "smoke_runs/model_lora_${SLURM_JOB_ID}" \
    --execute \
    --all

printf '\nSUCCESS: all five model-family LoRA smoke tests passed.\n'
