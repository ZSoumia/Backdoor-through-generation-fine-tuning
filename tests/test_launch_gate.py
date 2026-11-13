"""Fast launch gates: no Hugging Face downloads and no GPU required.

Run from repository root:
    python -m pytest -q tests/test_launch_gate.py
"""

import math
from pathlib import Path

import pytest

from bdsurvive import plan
from bdsurvive.core.registry import ATTACKS, FINETUNE, TRIGGERS
from bdsurvive.eval.base import BehavioralResult, DriftResult, GenerationResult
from bdsurvive.eval.extinction import delta_excess_ci, excess_ci
from bdsurvive.generation import GenerationChain
from bdsurvive.run import config_from_row  # also registers implementations
from bdsurvive.shards import make_shards


RUNNABLE = {"valid", "mechanism_not_applicable"}
STATUSES = RUNNABLE | {"not_applicable", "untested"}


def test_all_study_cells_expand_and_round_trip():
    studies = sorted(Path("configs/study").glob("*.yaml"))
    assert studies, "No study YAMLs found; run pytest from the repository root"
    for study_path in studies:
        study = plan.load_study(str(study_path))
        rows = plan.manifest_rows(plan.expand(study))
        assert rows, f"{study_path}: empty grid"
        ids = set()
        for position, row in enumerate(rows):
            assert row["index"] == position, f"{study_path}: bad index {position}"
            assert row["grid_status"] in STATUSES, (study_path, row)
            cfg = config_from_row(row)
            assert row["cell_id"] == cfg.cell_id, (study_path, position)
            assert cfg.cell_id not in ids, (study_path, cfg.cell_id)
            ids.add(cfg.cell_id)
            if row["grid_status"] in RUNNABLE:
                assert cfg.attack.trigger.name in TRIGGERS, (study_path, position)
                assert cfg.attack.installation in ATTACKS, (study_path, position)
                assert cfg.finetune.name in FINETUNE, (study_path, position)
                assert cfg.attack.steps > 0 and cfg.training.steps > 0
                assert cfg.attack.batch_size > 0 and cfg.training.batch_size > 0
                assert cfg.model.num_labels > 1
                if cfg.attack.trigger.name == "semantic_bite":
                    assert cfg.attack.poison_rate == 1.0, (study_path, position)


def test_shards_are_disjoint_from_planting_and_each_other():
    rows = [(f"sample-{i}", i % 4) for i in range(700)]
    alloc = make_shards(rows, n_generations=4, shard_size=100,
                        planting_block=250, seed=13)
    plant = set(alloc["planting_idx"])
    shards = [set(items) for items in alloc["shards"]]
    assert len(plant) == 250
    assert all(len(x) == 100 for x in shards)
    assert len(plant.union(*shards)) == 650
    for i, shard in enumerate(shards):
        assert plant.isdisjoint(shard)
        assert all(shard.isdisjoint(other) for other in shards[i + 1:])
    repeated = make_shards(rows, n_generations=4, shard_size=100,
                           planting_block=250, seed=13)
    assert alloc["shards"] == repeated["shards"]


def test_excess_ci_matches_trigger_minus_control():
    # High control rate intentionally mirrors the BERT positional pilot:
    # high raw ASR does not imply high *trigger-specific* excess.
    trigger = [1] * 84 + [0] * 16
    control = [1] * 74 + [0] * 26
    for paired in (True, False):
        result = excess_ci(trigger, control, paired=paired,
                           n_boot=500, seed=7)
        assert result["asr_excess"] == pytest.approx(0.10)
        assert -1 <= result["ci_lo"] <= result["ci_hi"] <= 1
        assert math.isfinite(result["ci_lo"])
        assert math.isfinite(result["ci_hi"])
        again = excess_ci(trigger, control, paired=paired,
                          n_boot=500, seed=7)
        assert result["ci_lo"] == again["ci_lo"]
        assert result["ci_hi"] == again["ci_hi"]


def test_paired_ci_rejects_unaligned_eval_arms():
    with pytest.raises(ValueError):
        excess_ci([1, 0, 1], [1, 0], paired=True, n_boot=100)


def test_generation_delta_ci_reports_zero_when_no_change():
    trigger = [1] * 80 + [0] * 20
    control = [1] * 20 + [0] * 80
    result = delta_excess_ci(trigger, control, trigger, control,
                             n_boot=250, seed=3)
    assert result["delta_excess"] == pytest.approx(0)
    assert result["delta_ci_lo"] == pytest.approx(0)
    assert result["delta_ci_hi"] == pytest.approx(0)
    assert not result["delta_excludes_zero"]


def test_delta_ci_rejects_unaligned_generations():
    with pytest.raises(ValueError):
        delta_excess_ci([1, 0], [0, 1], [1], [0, 1], n_boot=50)

class _FakeGenerationStep:
    """Exercise real checkpoint/state transitions without loading a model."""

    def __init__(self):
        self.calls = []

    def __call__(self, cfg, generation, parent_checkpoint, staging_dir):
        self.calls.append((generation, parent_checkpoint))
        staging = Path(staging_dir)
        staging.mkdir(parents=True, exist_ok=True)
        (staging / "checkpoint-marker.txt").write_text(f"G{generation}")
        behavioral = BehavioralResult(
            asr=0.8, control_rate=0.2, asr_excess=0.6,
            ci_lo=0.4, ci_hi=0.7, clean_metric=0.9,
            extinct=False, n_trigger=10, n_control=10,
        )
        return GenerationResult(
            generation=generation, behavioral=behavioral, drift=DriftResult(),
            bd_alignment=0.0, n_examples=4, n_tokens=20,
            coverage=1.0, loss_curve=[],
        )


def test_generation_chain_checkpoints_and_resume(tmp_path):
    study = plan.load_study("configs/study/section1_grid.yaml")
    cfg = plan.expand(study)[0]
    cfg.propagation.generations = 3
    cfg.propagation.retention = "all"
    fn = _FakeGenerationStep()
    run_dir = str(tmp_path / "mock_lineage")
    chain = GenerationChain(cfg, run_dir, fn)
    results = chain.run()

    assert [r.generation for r in results] == [0, 1, 2]
    assert [x[0] for x in fn.calls] == [0, 1, 2]
    assert fn.calls[0][1] is None
    assert Path(fn.calls[1][1]).exists()
    assert Path(fn.calls[2][1]).exists()
    assert (Path(run_dir) / "SUCCESS").exists()
    assert chain.completed_generations() == [0, 1, 2]

    # A completed chain must never repeat training or duplicate metric rows.
    chain.run()
    assert len(fn.calls) == 3
    assert chain.completed_generations() == [0, 1, 2]
