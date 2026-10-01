"""Execute one manifest row.

Everything above this is planning; everything below is machinery. This module
is what a SLURM array element calls: pick index i from the manifest, build the
objects from the registries, run the generation chain, write results.
"""
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from . import checkpoint as ckpt
from .core.config import ExperimentConfig
from .core.registry import ATTACKS, FINETUNE, SCORERS, TRIGGERS
from .core.validation import RunStatus, validate
from .eval.base import DriftResult, GenerationResult
from .generation import GenerationChain

# importing the implementation modules populates the registries
from .triggers import lexical, semantic, positional     # noqa: F401
from .attacks import data_poisoning                     # noqa: F401
from .finetune import full, lora, bitfit                # noqa: F401
from .eval import scorers as _scorers                   # noqa: F401


def config_from_row(row: Dict[str, Any]) -> ExperimentConfig:
    from dataclasses import fields
    from .core.config import (AttackSpec, DataSpec, FinetuneSpec, ModelSpec,
                              PropagationSpec, TargetSpec, TrainingSpec,
                              TriggerSpec)

    payload = row["config"]
    attack_payload = dict(payload["attack"])
    trigger_payload = attack_payload.pop("trigger")
    target_payload = attack_payload.pop("target")
    attack_payload["localized_layers"] = tuple(attack_payload["localized_layers"])

    return ExperimentConfig(
        model=ModelSpec(**payload["model"]),
        attack=AttackSpec(trigger=TriggerSpec(**trigger_payload),
                          target=TargetSpec(**target_payload),
                          **attack_payload),
        finetune=FinetuneSpec(**payload["finetune"]),
        training=TrainingSpec(**payload["training"]),
        data=DataSpec(**payload["data"]),
        propagation=PropagationSpec(**payload["propagation"]),
        seed=payload["seed"], output_root=payload["output_root"])


def default_scorer_name(cfg: ExperimentConfig) -> str:
    return "label_match" if cfg.attack.target.kind == "label" else "exact_string"


def build_components(cfg: ExperimentConfig) -> Tuple[Any, Any, Any]:
    trigger = TRIGGERS.create(cfg.attack.trigger.name, **cfg.attack.trigger.params)
    attack = ATTACKS.create(cfg.attack.installation, cfg.attack, trigger)
    method = FINETUNE.create(cfg.finetune.name, **cfg.finetune.params)
    return trigger, attack, method


def run_dir_for(cfg: ExperimentConfig) -> str:
    root = os.path.expanduser(cfg.output_root)
    return os.path.join(root, "runs", cfg.cell_id)


def write_result_row(cfg: ExperimentConfig, run_dir: str, status: RunStatus,
                     reason: Optional[str] = None,
                     extra: Optional[Dict[str, Any]] = None) -> str:
    """One row per cell with the full config embedded, so analysis never has
    to reconstruct which config produced which number."""
    row = cfg.flat_row()
    row["status"] = status.value
    row["reason"] = reason
    row["run_dir"] = run_dir
    if extra:
        row.update(extra)
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "result.json")
    with open(path, "w") as handle:
        json.dump(row, handle, indent=2, default=str)
    return path


def execute_row(row: Dict[str, Any], step_fn: Any,
                skip_completed: bool = True) -> Dict[str, Any]:
    """Run one manifest row end to end."""
    if row["grid_status"] != "valid":
        cfg = config_from_row(row)
        run_dir = run_dir_for(cfg)
        write_result_row(cfg, run_dir, RunStatus.NOT_APPLICABLE, row.get("reason"))
        return {"status": RunStatus.NOT_APPLICABLE.value, "reason": row.get("reason")}

    cfg = config_from_row(row)
    run_dir = run_dir_for(cfg)

    if skip_completed and ckpt.is_successful(run_dir):
        return {"status": "skipped", "reason": "already successful"}

    chain = GenerationChain(cfg, run_dir, step_fn)
    try:
        results = chain.run()
    except KeyboardInterrupt:
        write_result_row(cfg, run_dir, RunStatus.INTERRUPTED)
        raise
    except Exception as error:                      # noqa: BLE001
        write_result_row(cfg, run_dir, RunStatus.FAILED, str(error))
        raise

    final = results[-1] if results else None
    extra = final.row() if final else {}
    write_result_row(cfg, run_dir, RunStatus.SUCCESS, extra=extra)
    return {"status": RunStatus.SUCCESS.value, "generations": len(results)}
