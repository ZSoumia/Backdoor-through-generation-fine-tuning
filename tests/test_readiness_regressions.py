
import json
from pathlib import Path

import pytest

from bdsurvive import checkpoint as ckpt
from bdsurvive.generation import GenerationChain
from bdsurvive.core.config import (
    ExperimentConfig, ModelSpec, AttackSpec,
    TargetSpec, TriggerSpec,
)
from bdsurvive.core.validation import validate, GridStatus
from bdsurvive.eval.base import (
    GenerationResult, BehavioralResult, DriftResult,
)
from bdsurvive.train_loop import automatic_training_coverage


class FakeStep:
    def __init__(self):
        self.calls = []

    def __call__(self, cfg, generation, parent_checkpoint, staging_dir):
        self.calls.append(generation)
        stage = Path(staging_dir)
        stage.mkdir(parents=True, exist_ok=True)
        (stage / "checkpoint-marker.txt").write_text(str(generation))

        behavioral = BehavioralResult(
            .8, .2, .6, .4, .7, .9, False, 10, 10
        )

        return GenerationResult(
            generation, behavioral, DriftResult(),
            0.0, 4, 20, 1.0, []
        )


def config_two_generations():
    cfg = ExperimentConfig()
    cfg.propagation.generations = 2
    cfg.propagation.retention = "all"
    return cfg


def test_normal_resume(tmp_path):
    cfg = config_two_generations()
    step = FakeStep()
    chain = GenerationChain(cfg, str(tmp_path), step)

    chain.run()
    chain.run()

    assert step.calls == [0, 1, 2]
    assert chain.completed_generations() == [0, 1, 2]
    assert ckpt.is_successful(str(tmp_path))

    rows = [
        json.loads(line)
        for line in (tmp_path / "generations.jsonl").read_text().splitlines()
    ]
    assert rows[0]["coverage"] == 1.0


def test_recovery_before_promotion(tmp_path, monkeypatch):
    cfg = config_two_generations()
    step = FakeStep()
    original = ckpt.promote
    monkeypatch.setattr(
        ckpt, "promote",
        _FailOnce(original, "simulated promotion interruption"),
    )

    chain = GenerationChain(cfg, str(tmp_path), step)

    with pytest.raises(RuntimeError):
        chain.run()

    chain.run()

    assert step.calls == [0, 0, 1, 2]
    assert chain.completed_generations() == [0, 1, 2]


def test_recovery_after_promotion(tmp_path, monkeypatch):
    cfg = config_two_generations()
    step = FakeStep()
    original = ckpt.write_state
    monkeypatch.setattr(
        ckpt, "write_state",
        _FailOnce(original, "simulated state interruption"),
    )

    chain = GenerationChain(cfg, str(tmp_path), step)

    with pytest.raises(RuntimeError):
        chain.run()

    chain.run()

    assert step.calls == [0, 1, 2]
    assert chain.completed_generations() == [0, 1, 2]
    assert ckpt.is_successful(str(tmp_path))


def test_training_coverage():
    assert automatic_training_coverage(100, 10, 2) == 0.2
    assert automatic_training_coverage(100, 10, 20) == 1.0
    assert automatic_training_coverage(103, 10, 20) == 100 / 103

    with pytest.raises(ValueError):
        automatic_training_coverage(5, 10, 2)


def test_validation_precedence():
    cfg = ExperimentConfig(
        model=ModelSpec(
            position_encoding="learned",
            task_type="classification",
        ),
        attack=AttackSpec(
            trigger=TriggerSpec(name="positional"),
            target=TargetSpec(kind="string", value="X"),
        ),
    )

    assert validate(cfg).status is GridStatus.NOT_APPLICABLE

    cfg.attack.target = TargetSpec(kind="label", value=0)

    assert validate(cfg).status is GridStatus.VALID


class _FailOnce:
    """Callable replacement that fails once, without a nested function."""

    def __init__(self, original, message):
        self.original = original
        self.message = message
        self.fired = False

    def __call__(self, *args, **kwargs):
        if not self.fired:
            self.fired = True
            raise RuntimeError(self.message)
        return self.original(*args, **kwargs)
