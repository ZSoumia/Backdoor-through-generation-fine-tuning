
#!/bin/bash
#SBATCH --job-name=g0-validate
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/g0_validate_%j.out
#SBATCH --error=logs/slurm/g0_validate_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

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

# Check resources and corrected Python files.
test -s outputs/syntactic/ag_news.jsonl
python -m compileall -q \
  bdsurvive/steps.py \
  bdsurvive/eval/drift.py \
  bdsurvive/checkpoint.py

export VALIDATION_ROOT="validation_runs/g0_validate_${SLURM_JOB_ID}"
mkdir -p "$VALIDATION_ROOT"

# Create an isolated experiment manifest.
python - <<'PY'
import os
from pathlib import Path
from bdsurvive import plan
from bdsurvive.run import config_from_row, run_dir_for

root = Path(os.environ["VALIDATION_ROOT"]).resolve()

rows = plan.manifest_rows(
    plan.expand(
        plan.load_study("configs/study/section1_grid.yaml")
    )
)

matches = [
    r for r in rows
    if r["model"] == "EleutherAI/pythia-410m"
    and r["attack"] == "hiddenkiller"
    and r["finetune"] == "lora"
]

assert len(matches) == 1
assert matches[0]["grid_status"] == "valid"

cfg = config_from_row(matches[0])
cfg.output_root = str(root / "artifacts")

# Realistic G0 planting.
cfg.attack.steps = 1500
cfg.attack.batch_size = 16

# Minimal downstream generations for drift testing.
cfg.training.steps = 1
cfg.training.batch_size = 1
cfg.training.log_every = 1
cfg.data.shard_size = 1

# G2 is required to test cumulative drift against G0.
cfg.propagation.generations = 2
cfg.propagation.retention = "all"

manifest = plan.manifest_rows([cfg])
assert manifest[0]["grid_status"] == "valid"

plan.write_manifest(
    manifest, str(root / "manifest.jsonl")
)

(root / "run_dir.txt").write_text(run_dir_for(cfg))

print("RUN:", run_dir_for(cfg), flush=True)
print("G0: 1500 steps", flush=True)
print("G1/G2: 1 step each", flush=True)
PY

# Execute G0 -> G1 -> G2.
python -u -m bdsurvive.cli run \
  --manifest "$VALIDATION_ROOT/manifest.jsonl" \
  --index 0

# Validate the output, including drift calculations.
python - <<'PY'
import json
import math
import os
from pathlib import Path

from bdsurvive import plan
from bdsurvive.run import config_from_row
from bdsurvive.steps import load_model
from bdsurvive.eval.drift import weight_drift

root = Path(os.environ["VALIDATION_ROOT"])
run = Path((root / "run_dir.txt").read_text().strip())

assert (run / "SUCCESS").is_file()

metrics = [
    json.loads(line)
    for line in (run / "generations.jsonl").read_text().splitlines()
    if line.strip()
]

assert [r["generation"] for r in metrics] == [0, 1, 2]

for g in range(3):
    assert (run / f"checkpoint_gen{g:03d}").is_dir()

# At G1, cumulative and step drift must be equal.
assert math.isclose(
    metrics[1]["weight_drift_step"],
    metrics[1]["weight_drift_cum"],
    rel_tol=1e-5,
    abs_tol=1e-6
)

# Independently recompute the G2 drift values.
cfg = config_from_row(
    plan.read_manifest(str(root / "manifest.jsonl"))[0]
)

models = [
    load_model(
        str(run / f"checkpoint_gen{g:03d}"),
        cfg,
        "cpu"
    )
    for g in range(3)
]

expected_step = weight_drift(
    models[2].state_dict(),
    models[1].state_dict()
)

expected_cum = weight_drift(
    models[2].state_dict(),
    models[0].state_dict()
)

assert math.isclose(
    metrics[2]["weight_drift_step"],
    expected_step,
    rel_tol=1e-4,
    abs_tol=1e-4
)

assert math.isclose(
    metrics[2]["weight_drift_cum"],
    expected_cum,
    rel_tol=1e-4,
    abs_tol=1e-4
)

print("PASS: G2 step and cumulative drift verified")

g0 = metrics[0]

print("\nG0 ESTABLISHMENT RESULTS")
print("Clean accuracy:", g0["clean_metric"])
print("Trigger ASR:", g0["asr"])
print("Control rate:", g0["control_rate"])
print("Excess ASR:", g0["asr_excess"])
print("95% CI:", g0["ci_lo"], g0["ci_hi"])

if g0["clean_metric"] < 0.5 or g0["control_rate"] > 0.5:
    print("REVIEW: G0 classification or specificity may be inadequate")

print("Results:", run / "generations.jsonl")
PY

echo "SUCCESS: G0 validation and cumulative drift verification completed"
