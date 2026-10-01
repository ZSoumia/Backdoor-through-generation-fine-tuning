"""Grid validity.

N/A is a first-class grid outcome, distinct from "ran and didn't survive".
A cell marked NOT_APPLICABLE never reaches a GPU, and its reason travels with
the manifest row so paper tables can print N/A instead of 0.00.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import ExperimentConfig


class GridStatus(str, Enum):
    VALID = "valid"
    NOT_APPLICABLE = "not_applicable"


class RunStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class ValidationResult:
    status: GridStatus
    reason: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.status is GridStatus.VALID


VALID = ValidationResult(GridStatus.VALID)


def not_applicable(reason: str) -> ValidationResult:
    return ValidationResult(GridStatus.NOT_APPLICABLE, reason)


def validate_trigger_model(cfg: ExperimentConfig) -> ValidationResult:
    """Positional/length triggers are RoPE-mediated (MetaBackdoor Sec. V-E):
    masked padding does not fire them, and stride-scaling relative positions
    does. A model with learned absolute position embeddings cannot express
    the mechanism at all -- that is N/A, not a failed attack."""
    if cfg.attack.trigger_type == "positional" and cfg.model.position_encoding != "rope":
        return not_applicable(
            f"positional trigger requires RoPE, model uses "
            f"{cfg.model.position_encoding}")
    return VALID


def validate_finetune_model(cfg: ExperimentConfig) -> ValidationResult:
    """Prompt-family methods prepend virtual tokens to a decoder context;
    they have no meaning against a pooled classification head here."""
    if cfg.finetune.family == "prompt" and cfg.model.task_type == "classification":
        return not_applicable(
            "prompt-family finetuning requires a generative task type")
    return VALID


def validate_target_task(cfg: ExperimentConfig) -> ValidationResult:
    """A string target needs a model that can emit strings; a label target
    needs a finite label space."""
    kind = cfg.attack.target.kind
    if kind == "string" and cfg.model.task_type != "generative":
        return not_applicable("string target requires a generative model")
    if kind == "label" and cfg.model.task_type != "classification":
        return not_applicable("label target requires a classification model")
    return VALID


def validate_propagation(cfg: ExperimentConfig) -> ValidationResult:
    if cfg.propagation.generations < 1:
        return not_applicable("generations must be >= 1")
    return VALID


CHECKS = (
    validate_trigger_model,
    validate_finetune_model,
    validate_target_task,
    validate_propagation,
)


def validate(cfg: ExperimentConfig) -> ValidationResult:
    """Run every check; first N/A wins. Called by plan.py, before any GPU."""
    for check in CHECKS:
        result = check(cfg)
        if not result.is_valid:
            return result
    return VALID


def validate_all(configs: List[ExperimentConfig]):
    return [(cfg, validate(cfg)) for cfg in configs]
