#!/bin/bash
#SBATCH --job-name=multigen-smoke
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm/multigen_%j.out
#SBATCH --error=logs/slurm/multigen_%j.err

# Real Pythia-410M + HiddenKiller + LoRA G0->G1->G2->G3,
# with a subsequent continuous benign-FT control on the same G0.
# One step per phase. These are engineering tests, NOT efficacy measurements.
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
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA not available'
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

root="smoke_runs/multigen_${SLURM_JOB_ID}"
export MULTIGEN_ROOT="$root"
mkdir -p "$root"

# Make a *new* manifest containing exactly one reduced-budget, 3-generation cell.
python - <<'PY'
import os
from pathlib import Path
from bdsurvive import plan
from bdsurvive.run import config_from_row, run_dir_for
rows = plan.manifest_rows(plan.expand(plan.load_study('configs/study/section1_grid.yaml')))
matching = [r for r in rows if r['model']=='EleutherAI/pythia-410m' and
            r['attack']=='hiddenkiller' and r['finetune']=='lora']
assert len(matching) == 1 and matching[0]['grid_status'] == 'valid'
cfg = config_from_row(matching[0])
cfg.output_root = str((Path(os.environ['MULTIGEN_ROOT']) / 'artifacts').resolve())
cfg.attack.steps = 1
cfg.attack.batch_size = 1
cfg.training.steps = 1
cfg.training.batch_size = 1
cfg.training.log_every = 1
cfg.data.shard_size = 1
cfg.propagation.generations = 3
cfg.propagation.retention = 'all'
new_row = plan.manifest_rows([cfg])[0]
assert new_row['grid_status'] == 'valid', new_row
root = Path(os.environ['MULTIGEN_ROOT'])
plan.write_manifest([new_row], str(root / 'manifest.jsonl'))
(root / 'run_dir.txt').write_text(run_dir_for(cfg), encoding='utf-8')
(root / 'SMOKE_ONLY.txt').write_text('One-step generational engineering test; not for science.\n')
print('MULTIGEN RUN:', run_dir_for(cfg), flush=True)
PY

python -u -m bdsurvive.cli run --manifest "$root/manifest.jsonl" --index 0
# Rerunning a successful cell must be a no-op, not repeat any training.
python -u -m bdsurvive.cli run --manifest "$root/manifest.jsonl" --index 0
# Test the separately implemented continuous-control path.
python -u -m bdsurvive.cli continuous --manifest "$root/manifest.jsonl" --index 0

python - <<'PY'
import json
import os
from pathlib import Path
from tests.smoke_matrix import validate_metrics
root = Path(os.environ['MULTIGEN_ROOT'])
run_dir = Path((root / 'run_dir.txt').read_text().strip())
assert (run_dir / 'SUCCESS').exists(), 'Missing generational SUCCESS marker'
assert (run_dir / 'state.json').exists(), 'Missing lineage state'
state = json.loads((run_dir / 'state.json').read_text())
assert state['last_completed_generation'] == 3, state
metrics = run_dir / 'generations.jsonl'
assert metrics.is_file(), f'Missing metrics: {metrics}'
issues, warnings = validate_metrics(metrics, expected_generations=4)
assert not issues, issues
for g in range(4):
    assert (run_dir / f'checkpoint_gen{g:03d}').is_dir(), f'Missing checkpoint G{g}'
control = run_dir / 'continuous'
assert (control / 'SUCCESS').exists(), 'Missing continuous SUCCESS marker'
boundaries = control / 'checkpoint' / 'continuous_boundaries.jsonl'
assert boundaries.is_file(), f'Missing continuous metrics: {boundaries}'
records = [json.loads(s) for s in boundaries.read_text().splitlines() if s.strip()]
assert [r['generation'] for r in records] == [1, 2, 3], records
assert all(r.get('arm') == 'continuous' for r in records)
print('PASS: G0-G3 metrics, checkpoints, resume no-op, continuous G1-G3', flush=True)
for warning in warnings:
    print('WARNING:', warning, flush=True)
PY

echo 'SUCCESS: multi-generation and continuous-control smoke passed'
