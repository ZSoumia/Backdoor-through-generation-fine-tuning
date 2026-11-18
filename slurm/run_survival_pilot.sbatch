#!/bin/bash
#SBATCH --job-name=survival-pilot
#SBATCH --account=try26_zuziak
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/slurm/survival_pilot_%A_%a.out
#SBATCH --error=logs/slurm/survival_pilot_%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module load profile/deeplrn
module load cineca-ai/4.3.0
source /leonardo_work/try26_zuziak/bdsurvive-venv/bin/activate

export HF_HOME=/leonardo_work/try26_zuziak/hf_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

export PILOT_ROOT="$PWD/validation_runs/survival_pilot_from_64768748"
export SOURCE_RUN="$PWD/validation_runs/g0_validate_64768748/artifacts/runs_v3/pythia-410m-classification-hiddenkiller-lora-s0-2b642c8f1b"
export METHOD_INDEX="$SLURM_ARRAY_TASK_ID"

# Initialize each fine-tuning arm from the SAME G0.
# This does not retrain or alter the source checkpoint.
python - <<'PY'
import copy
import json
import os
import shutil
from pathlib import Path

from bdsurvive import checkpoint as ckpt, plan
from bdsurvive.run import run_dir_for

methods = ("lora", "bitfit", "full")
method = methods[int(os.environ["METHOD_INDEX"])]

root = Path(os.environ["PILOT_ROOT"]).resolve()
source = Path(os.environ["SOURCE_RUN"]).resolve()
source_ckpt = source / "checkpoint_gen000"
source_records = source / "generations.jsonl"

assert source_ckpt.is_dir(), f"G0 weights missing: {source_ckpt}"
assert source_records.is_file(), f"G0 metrics missing: {source_records}"

g0 = json.loads(source_records.read_text().splitlines()[0])
assert g0["generation"] == 0
assert g0["clean_metric"] >= 0.70
assert g0["asr_excess"] >= 0.50

configs = plan.expand(
    plan.load_study("configs/study/section1_grid.yaml")
)

choices = [
    c for c in configs
    if c.model.name == "EleutherAI/pythia-410m"
    and c.attack.name == "hiddenkiller"
    and c.finetune.name == method
    and c.seed == 0
]
assert len(choices) == 1

cfg = copy.deepcopy(choices[0])
cfg.output_root = str(root / method)
cfg.attack.steps = 1500
cfg.training.steps = 300
cfg.training.batch_size = 16
cfg.training.lr = 1e-4
cfg.propagation.generations = 3
cfg.propagation.retention = "all"

rows = plan.manifest_rows([cfg])
assert rows[0]["grid_status"] == "valid", rows[0]["reason"]

manifest = root / method / "manifest.jsonl"
plan.write_manifest(rows, str(manifest))

run_dir = Path(run_dir_for(cfg))
checkpoint = Path(ckpt.checkpoint_dir(str(run_dir), 0))
state_path = Path(ckpt.state_path(str(run_dir)))

if not state_path.exists():
    if checkpoint.exists() or (run_dir / "generations.jsonl").exists():
        raise RuntimeError(
            "Partial G0 initialization: inspect before retrying"
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_ckpt, checkpoint)

    (run_dir / "generations.jsonl").write_text(
        json.dumps(g0) + "\n"
    )

    (run_dir / "g0_provenance.json").write_text(
        json.dumps({
            "source_checkpoint": str(source_ckpt),
            "method": method,
            "seed": cfg.seed,
            "g0_clean_accuracy": g0["clean_metric"],
            "g0_excess_asr": g0["asr_excess"],
        }, indent=2) + "\n"
    )

    ckpt.write_state(
        str(run_dir),
        ckpt.ChainState(
            run_id=cfg.cell_id,
            last_completed_generation=0,
            current_checkpoint=str(checkpoint),
            status="running",
        ),
    )
else:
    state = ckpt.load_state(str(run_dir), cfg.cell_id)
    assert state.run_id == cfg.cell_id
    assert (run_dir / "generations.jsonl").is_file()
    assert checkpoint.is_dir(), "Retained G0 checkpoint missing"

print("METHOD:", method, flush=True)
print("SHARED SOURCE G0:", source_ckpt, flush=True)
print("LINEAGE:", run_dir, flush=True)
print("MANIFEST:", manifest, flush=True)
PY

METHODS=(lora bitfit full)
METHOD="${METHODS[$SLURM_ARRAY_TASK_ID]}"
MANIFEST="$PILOT_ROOT/$METHOD/manifest.jsonl"

# Run G1, G2, G3 from the established G0.
python -u -m bdsurvive.cli run \
  --manifest "$MANIFEST" \
  --index 0

# Continuous comparator: 900 updates from the same G0.
python -u -m bdsurvive.cli continuous \
  --manifest "$MANIFEST" \
  --index 0

# Validate and summarize the results.
python - <<'PY'
import json
import os
from pathlib import Path

from bdsurvive import plan
from bdsurvive.run import config_from_row, run_dir_for

method = ("lora", "bitfit", "full")[
    int(os.environ["METHOD_INDEX"])
]

manifest = (
    Path(os.environ["PILOT_ROOT"])
    / method
    / "manifest.jsonl"
)

cfg = config_from_row(
    plan.read_manifest(str(manifest))[0]
)

run = Path(run_dir_for(cfg))
assert (run / "SUCCESS").is_file()

rows = [
    json.loads(s)
    for s in (run / "generations.jsonl").read_text().splitlines()
    if s.strip()
]

assert [r["generation"] for r in rows] == [0, 1, 2, 3]

for g in range(4):
    assert (run / f"checkpoint_gen{g:03d}").is_dir()

cont = (
    run / "continuous"
    / "checkpoint"
    / "continuous_boundaries.jsonl"
)

assert cont.is_file()

control = [
    json.loads(s)
    for s in cont.read_text().splitlines()
    if s.strip()
]

assert [r["generation"] for r in control] == [1, 2, 3]

print("\nGENERATIONAL RESULTS:", method)

for r in rows:
    print(
        f"G{r['generation']}: "
        f"clean={r['clean_metric']:.3f} "
        f"trigger={r['asr']:.3f} "
        f"control={r['control_rate']:.3f} "
        f"excess={r['asr_excess']:.3f} "
        f"CI=[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}] "
        f"drift_step={r['weight_drift_step']:.4f} "
        f"drift_cum={r['weight_drift_cum']:.4f}"
    )

print("CONTINUOUS CONTROL:")

for r in control:
    print(
        f"G{r['generation']}: "
        f"clean={r['clean_metric']:.3f} "
        f"excess={r['asr_excess']:.3f}"
    )

print(
    "PASS:", method,
    "three realistic generations plus continuous comparator"
)
PY
