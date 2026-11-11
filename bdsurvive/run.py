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
from .triggers import lexical, semantic, positional, syntactic  
from .attacks import data_poisoning                     
from .finetune import full, lora, bitfit                
from .eval import scorers as _scorers                   


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
    return os.path.join(root, "runs_v3", cfg.cell_id)

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
    if row.get("cell_id") != cfg.cell_id:
        raise ValueError(
            "manifest cell ID is outdated; regenerate the manifest"
        )

    run_dir = run_dir_for(cfg)

    state = ckpt.load_state(run_dir, cfg.cell_id)

    if state.run_id != cfg.cell_id:
        raise ValueError("resume state belongs to a different experiment")

    if ckpt.is_successful(run_dir):
        if state.last_completed_generation >= cfg.propagation.generations:
            if not skip_completed:
                raise ValueError("refusing to overwrite a completed lineage")
            return {"status": "skipped", "reason": "already successful"}

        if not state.current_checkpoint or not os.path.isdir(state.current_checkpoint):
            raise ValueError("cannot extend: last completed checkpoint is missing")

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

def execute_continuous_row(
    row: Dict[str, Any],
    step_fn: Any,
    skip_completed: bool = True,
) -> Dict[str, Any]:
    import shutil

    if row["grid_status"] != "valid":
        return {
            "status": RunStatus.NOT_APPLICABLE.value,
            "reason": row.get("reason"),
        }

    cfg = config_from_row(row)

    if row["cell_id"] != cfg.cell_id:
        raise ValueError("Manifest cell_id does not match config")

    run_dir = run_dir_for(cfg)
    control_dir = os.path.join(run_dir, "continuous")
    final_dir = os.path.join(control_dir, "checkpoint")
    staging_dir = os.path.join(control_dir, "staging")
    results_file = "continuous_boundaries.jsonl"

    if skip_completed and ckpt.is_successful(control_dir):
        if not os.path.isfile(
            os.path.join(final_dir, results_file)
        ):
            raise RuntimeError(
                "Continuous SUCCESS exists but results are missing"
            )
        return {
            "status": "skipped",
            "reason": "continuous control already successful",
        }

    state = ckpt.load_state(run_dir, cfg.cell_id)

    if (
        not ckpt.is_successful(run_dir)
        or state.last_completed_generation
        < cfg.propagation.generations
    ):
        raise RuntimeError(
            "Complete the generational run before "
            "running its continuous control"
        )

    g0_checkpoint = ckpt.checkpoint_dir(run_dir, 0)

    if not os.path.isdir(g0_checkpoint):
        raise FileNotFoundError(
            f"G0 checkpoint is missing: {g0_checkpoint}"
        )

    os.makedirs(control_dir, exist_ok=True)

    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)

    success_marker = os.path.join(control_dir, "SUCCESS")
    if os.path.exists(success_marker):
        os.remove(success_marker)

    try:
        result = step_fn(
            cfg=cfg,
            g0_checkpoint=g0_checkpoint,
            staging_dir=staging_dir,
        )

        boundaries = result["boundary_results"]

        if len(boundaries) != cfg.propagation.generations:
            raise RuntimeError(
                "Incomplete continuous boundary evaluations"
            )

        if not os.path.isfile(
            os.path.join(staging_dir, results_file)
        ):
            raise RuntimeError(
                "Continuous boundary results were not saved"
            )

        ckpt.promote(staging_dir, final_dir)

        write_result_row(
            cfg,
            control_dir,
            RunStatus.SUCCESS,
            extra={
                "arm": "continuous",
                "updates_benign_cumulative":
                    result["updates_benign_cumulative"],
                "generations_evaluated": len(boundaries),
                "checkpoint": final_dir,
                "boundary_results_path": os.path.join(
                    final_dir, results_file
                ),
            },
        )

        ckpt.mark_success(control_dir)

    except KeyboardInterrupt:
        write_result_row(
            cfg, control_dir, RunStatus.INTERRUPTED
        )
        raise

    except Exception as error:
        write_result_row(
            cfg, control_dir, RunStatus.FAILED, str(error)
        )
        raise

    return {
        "status": RunStatus.SUCCESS.value,
        "arm": "continuous",
        "generations": len(boundaries),
        "run_dir": control_dir,
    }
