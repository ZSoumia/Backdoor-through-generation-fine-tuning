"""Sweep expansion -> manifest.

A study config lists axis values; plan.py expands the Cartesian product,
validates every cell, and writes one manifest row per cell. Invalid cells are
kept in the manifest with grid_status=not_applicable and a reason, so paper
tables can distinguish N/A from a tested failure -- and so no GPU is ever
spent discovering an incompatibility.

The four paper sections are four study YAMLs over this one planner: they
differ only in which axes are swept and which are pinned.
"""
import itertools
import json
import os
from dataclasses import replace
from typing import Any, Dict, Iterable, List

import yaml

from .core.config import (AttackSpec, DataSpec, ExperimentConfig, FinetuneSpec,
                          ModelSpec, PropagationSpec, TargetSpec, TrainingSpec,
                          TriggerSpec)
from .core.validation import GridStatus, validate


def build_model(entry: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(**entry)


def build_attack(entry: Dict[str, Any]) -> AttackSpec:
    payload = dict(entry)
    trigger = payload.pop("trigger", {"name": "lexical"})
    target = payload.pop("target", {"kind": "label", "value": 0})
    if isinstance(trigger, str):
        trigger = {"name": trigger}
    layers = payload.pop("localized_layers", None)
    spec = AttackSpec(trigger=TriggerSpec(**trigger),
                      target=TargetSpec(**target), **payload)
    if layers is not None:
        spec.localized_layers = tuple(layers)
    return spec


def build_finetune(entry: Dict[str, Any]) -> FinetuneSpec:
    return FinetuneSpec(**entry)


AXIS_BUILDERS = {
    "model": build_model,
    "attack": build_attack,
    "finetune": build_finetune,
}


def load_study(path: str) -> Dict[str, Any]:
    with open(path) as handle:
        return yaml.safe_load(handle)


def expand(study: Dict[str, Any]) -> List[ExperimentConfig]:
    """Cartesian product over model x attack x finetune x seed."""
    models = [build_model(e) for e in study.get("model", [{}])]
    attacks = [build_attack(e) for e in study.get("attack", [{}])]
    finetunes = [build_finetune(e) for e in study.get("finetune", [{}])]
    seeds = study.get("seed", [0])

    shared = study.get("shared", {})
    training = TrainingSpec(**shared.get("training", {}))
    data = DataSpec(**shared.get("data", {}))
    propagation = PropagationSpec(**shared.get("propagation", {}))
    output_root = shared.get("output_root", "~/bdsurvive_runs")

    configs: List[ExperimentConfig] = []
    for model, attack, finetune, seed in itertools.product(
            models, attacks, finetunes, seeds):
        configs.append(ExperimentConfig(
            model=model, attack=replace(attack, seed=seed),
            finetune=finetune, training=training, data=data,
            propagation=propagation, seed=seed, output_root=output_root))
    return configs


def manifest_rows(configs: Iterable[ExperimentConfig]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, cfg in enumerate(configs):
        result = validate(cfg)
        row = cfg.flat_row()
        row.update({
            "index": index,
            "grid_status": result.status.value,
            "reason": result.reason,
            "config": cfg.to_dict(),
        })
        rows.append(row)
    return rows


def write_manifest(rows: List[Dict[str, Any]], path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    return path


def read_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in rows if r["grid_status"] == GridStatus.VALID.value]
    na = [r for r in rows if r["grid_status"] == GridStatus.NOT_APPLICABLE.value]
    reasons: Dict[str, int] = {}
    for row in na:
        reasons[row["reason"]] = reasons.get(row["reason"], 0) + 1
    return {"total": len(rows), "valid": len(valid), "not_applicable": len(na),
            "na_reasons": reasons}
