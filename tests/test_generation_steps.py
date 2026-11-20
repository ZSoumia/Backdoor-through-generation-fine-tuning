"""Generation execution and continuous-control tests.

All mock implementations are module-level and capture per-test state explicitly.
"""

import os
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

from bdsurvive import steps, train_loop
from bdsurvive.core.config import (
    AttackSpec,
    DataSpec,
    ExperimentConfig,
    PropagationSpec,
    TrainingSpec,
)
from bdsurvive.eval.base import BehavioralResult


class _ScalarModel:
    def __init__(self, value):
        self.value = float(value)

    def state_dict(self):
        return {"weight": torch.tensor([self.value])}

    def save_pretrained(self, path):
        with open(os.path.join(path, "model.txt"), "w") as handle:
            handle.write(str(self.value))


class _ScalarTokenizer:
    def save_pretrained(self, path):
        with open(os.path.join(path, "tok.txt"), "w") as handle:
            handle.write("ok")


class _PlantAttack:
    def __init__(self, context):
        self.context = context

    def install(self, model, tok, planting_rows, device):
        self.context.events.append(("plant", len(planting_rows)))
        model.value += 10
        return model


class _ScalarMethod:
    def __init__(self, context):
        self.context = context

    def wrap(self, model):
        self.context.events.append(("wrap", model.value))
        return model

    def merge(self, model):
        return model


class _ContinuousModel:
    def __init__(self, value):
        self.value = int(value)

    def save_pretrained(self, path):
        (Path(path) / "model.txt").write_text(str(self.value))


class _ContinuousTokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.txt").write_text("ok")


class _ContinuousMethod:
    def __init__(self, context):
        self.context = context

    def wrap(self, model):
        self.context.events.append("wrap")
        return model

    def merge(self, model):
        self.context.events.append("merge")
        return model


def _scalar_load_model(context, path, cfg, device):
    if os.path.isdir(path):
        with open(os.path.join(path, "model.txt")) as handle:
            return _ScalarModel(float(handle.read()))
    return _ScalarModel(0)


def _scalar_train(context, model, tok, texts, labels, steps, **kwargs):
    context.events.append(("benign", tuple(texts), steps))
    model.value += 1
    return [{"step": 0, "loss": 0.0, "ema_loss": 0.0}]


def _continuous_load_model(context, path, cfg, device):
    context.events.append(("load", path))
    return _ContinuousModel((Path(path) / "model.txt").read_text())


def _continuous_train(context, **kwargs):
    context.events.append("train")
    context.captured.update(kwargs)
    kwargs["model"].value += 1
    for completed_steps in range(1, kwargs["steps"] + 1):
        kwargs["on_step_end"](completed_steps, kwargs["model"])
    return [{"step": 0, "loss": 0.1, "ema_loss": 0.1}]


def _continuous_boundary(completed_steps, model, *, cfg, boundary_results, **kwargs):
    if completed_steps % cfg.training.steps != 0:
        return
    boundary_results.append({
        "arm": "continuous",
        "generation": completed_steps // cfg.training.steps,
        "updates_benign_cumulative": completed_steps,
    })


def _fixture_rows(context, task, split):
    return context.rows


def _scalar_tokenizer(name):
    return _ScalarTokenizer()


def _continuous_tokenizer(name):
    return _ContinuousTokenizer()


def _identity_model(model, tok):
    return model


def _unconfigured_registry(*args, **kwargs):
    return object()


def _scalar_method(context, *args, **kwargs):
    return _ScalarMethod(context)


def _plant_attack(context, *args, **kwargs):
    return _PlantAttack(context)


def _continuous_method(context, *args, **kwargs):
    return _ContinuousMethod(context)


def _fake_eval_sets(*args):
    return ["trigger"], ["control"], ["clean"], [0]


def _fake_backdoor_gradient(*args):
    return {"weight": torch.tensor([1.0])}


def _zero_functional_drift(*args):
    return 0.0


def _fake_token_count(tok, texts, max_len):
    return len(texts) * 3


def _fake_behavioral(**kwargs):
    return BehavioralResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.8, False, 1, 1)


def test_c2_training_step_does_not_benign_train_g0(monkeypatch, tmp_path):
    cfg = ExperimentConfig(
        attack=AttackSpec(planting_pool_size=12),
        data=DataSpec(shard_size=4),
        training=TrainingSpec(steps=2, batch_size=2),
        propagation=PropagationSpec(generations=4),
    )
    context = SimpleNamespace(
        rows=[(f"article {i}", i % 4) for i in range(80)],
        events=[],
    )

    monkeypatch.setattr(steps, "load_tokenizer", _scalar_tokenizer)
    monkeypatch.setattr(steps, "load_model", partial(_scalar_load_model, context))
    monkeypatch.setattr(steps, "prepare_model", _identity_model)
    monkeypatch.setattr(steps, "load_rows", partial(_fixture_rows, context))
    monkeypatch.setattr(steps.TRIGGERS, "create", _unconfigured_registry)
    monkeypatch.setattr(steps.FINETUNE, "create", partial(_scalar_method, context))
    monkeypatch.setattr(steps.SCORERS, "create", _unconfigured_registry)
    monkeypatch.setattr(steps.ATTACKS, "create", partial(_plant_attack, context))
    monkeypatch.setattr(steps, "eval_sets", _fake_eval_sets)
    monkeypatch.setattr(steps, "backdoor_gradient", _fake_backdoor_gradient)
    monkeypatch.setattr(steps, "functional_drift", _zero_functional_drift)
    monkeypatch.setattr(steps, "token_count", _fake_token_count)
    monkeypatch.setattr(steps, "evaluate_behavioral", _fake_behavioral)
    monkeypatch.setattr(steps, "train_supervised", partial(_scalar_train, context))

    stage0 = str(tmp_path / "g0")
    g0 = steps.training_step(cfg, 0, None, stage0, device="cpu")

    assert context.events == [("plant", 12)]
    assert g0.n_examples == 0
    assert g0.n_tokens == 0
    assert g0.updates_benign_cumulative == 0
    assert g0.loss_curve == []
    assert (tmp_path / "g0" / "model.txt").read_text() == "10.0"

    stage1 = str(tmp_path / "g1")
    g1 = steps.training_step(cfg, 1, stage0, stage1, device="cpu")

    assert [event[0] for event in context.events] == ["plant", "wrap", "benign"]
    allocation = steps.lineage_allocation(cfg, context.rows)
    expected = tuple(context.rows[i][0] for i in allocation["shards"][0])
    assert context.events[-1] == ("benign", expected, 2)
    assert g1.n_examples == 4
    assert g1.updates_benign_cumulative == 2
    assert (tmp_path / "g1" / "model.txt").read_text() == "11.0"


def test_continuous_control_from_g0(monkeypatch, tmp_path):
    cfg = ExperimentConfig(
        attack=AttackSpec(planting_pool_size=12),
        data=DataSpec(shard_size=4),
        training=TrainingSpec(steps=2, batch_size=2),
        propagation=PropagationSpec(generations=3),
    )
    context = SimpleNamespace(
        rows=[(f"article {i}", i % 4) for i in range(80)],
        events=[],
        captured={},
    )
    g0_dir = tmp_path / "g0"
    g0_dir.mkdir()
    (g0_dir / "model.txt").write_text("10")

    monkeypatch.setattr(steps, "load_rows", partial(_fixture_rows, context))
    monkeypatch.setattr(steps, "load_tokenizer", _continuous_tokenizer)
    monkeypatch.setattr(steps, "load_model", partial(_continuous_load_model, context))
    monkeypatch.setattr(steps, "prepare_model", _identity_model)
    monkeypatch.setattr(steps.FINETUNE, "create", partial(_continuous_method, context))
    monkeypatch.setattr(train_loop, "train_supervised", partial(_continuous_train, context))
    monkeypatch.setattr(steps, "evaluate_continuous_boundary", _continuous_boundary)

    control_dir = tmp_path / "continuous"
    result = steps.train_continuous_control(
        cfg, g0_checkpoint=str(g0_dir), staging_dir=str(control_dir), device="cpu"
    )

    assert context.events == [
        ("load", str(g0_dir)), "wrap", "train", "merge"
    ]
    assert context.captured["steps"] == 6
    assert context.captured["batch_size"] == 2
    assert len(context.captured["batch_order"]) == 6

    allocation = steps.lineage_allocation(cfg, context.rows)
    expected_batches = []
    for g, shard in enumerate(allocation["shards"], start=1):
        batches = train_loop.make_batches(len(shard), 2, cfg.seed * 1000 + g)
        for step in range(2):
            batch = batches[step % len(batches)]
            expected_batches.append([context.rows[shard[i]][0] for i in batch])

    actual_batches = [
        [context.captured["texts"][i] for i in batch]
        for batch in context.captured["batch_order"]
    ]
    assert actual_batches == expected_batches
    assert result["arm"] == "continuous"
