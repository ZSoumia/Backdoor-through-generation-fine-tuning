"""Grid validity
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import ExperimentConfig


class GridStatus(str, Enum):
    VALID = "valid"
    NOT_APPLICABLE = "not_applicable"
    # A cell where the specific causally-demonstrated mechanism is
    # unavailable but the broader behavioural question is still measurable.
    # Distinct from NOT_APPLICABLE, which means nothing can be learned from
    # running it. A MECHANISM_NOT_APPLICABLE cell SHOULD run: its result is
    # evidence about length-conditioned behaviour, it just must not be
    # labelled as the RoPE mechanism.
    MECHANISM_NOT_APPLICABLE = "mechanism_not_applicable"
    # Plausible but never run. An honest third state: absent from the grid is
    # not the same as shown to be impossible, and reporting the two with one
    # label is what made the GPT-Neo and OPT cells look settled.
    UNTESTED = "untested"


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

    @property
    def should_run(self) -> bool:
        """Whether the cell is worth executing.

        True for MECHANISM_NOT_APPLICABLE as well as VALID: that cell yields a
        real measurement, it just answers the length-conditioned question
        rather than the RoPE-mechanism one. Only NOT_APPLICABLE means there is
        nothing to learn.
        """
        return self.status in (GridStatus.VALID,
                               GridStatus.MECHANISM_NOT_APPLICABLE)

    @property
    def reporting_label(self) -> str:
        """What belongs in the results table for this cell."""
        return {
            GridStatus.VALID: "measured",
            GridStatus.MECHANISM_NOT_APPLICABLE:
                "measured (RoPE mechanism N/A)",
            GridStatus.UNTESTED: "untested",
            GridStatus.NOT_APPLICABLE: "N/A",
        }[self.status]


VALID = ValidationResult(GridStatus.VALID)


def not_applicable(reason: str) -> ValidationResult:
    return ValidationResult(GridStatus.NOT_APPLICABLE, reason)


def validate_trigger_model(cfg: ExperimentConfig) -> ValidationResult:
    """Graded verdict for a length trigger against a position scheme.

    The MetaBackdoor mechanism as causally demonstrated is RoPE-mediated
    (Sec. V-E): masked padding does not fire it, stride-scaling relative
    positions does. On a learned-absolute model that mechanism cannot exist.

    It does NOT follow that nothing can be measured there. Every scheme in
    POSITIONAL_APPLICABILITY exposes position information, so a
    length-conditioned label flip is plantable on all of them; what differs is
    which parameters carry it. This returns MECHANISM_NOT_APPLICABLE for those
    cells -- run them, measure persistence, and report the result without
    claiming it is the RoPE mechanism.

    The learned-absolute case is the most informative of them, not the least:
    the position table is trainable, so the backdoor's support sits outside
    LoRA's and BitFit's reach and inside full fine-tuning's. That is a
    support-overlap prediction no other cell in the grid can make, and the
    previous blanket N/A is what hid it.
    """
    if cfg.attack.trigger_type != "positional":
        return VALID
    return positional_verdict(cfg.model.position_encoding)


def positional_verdict(position_encoding: str) -> ValidationResult:
    """Shared by the Stack-B path and the live PlantConfig path.

    Gap 3 was that only validate_trigger_model knew this rule and it takes an
    ExperimentConfig, while the executing path carries a PlantConfig -- so the
    verdict was enforced in practice only by a mechanism entry being absent
    from the YAML. Both paths now call this.
    """
    from ..models import positional_applicability
    info = positional_applicability(position_encoding)

    if info["rope_mechanism"] == "unknown":
        return ValidationResult(
            GridStatus.UNTESTED,
            f"unrecognised position scheme {position_encoding!r}; add it to "
            f"POSITIONAL_APPLICABILITY before reporting this cell")
    if info["rope_mechanism"] == "applicable":
        return VALID
    if info["length_conditioned"] == "measurable":
        return ValidationResult(
            GridStatus.MECHANISM_NOT_APPLICABLE,
            f"RoPE-mediated mechanism unavailable under "
            f"{position_encoding!r}, but length-conditioned behaviour is "
            f"measurable: support would live in {info['support_lives_in']}. "
            f"In LoRA support: {info['in_lora_support']}; BitFit: "
            f"{info['in_bitfit_support']}; full: {info['in_full_support']}.")
    return not_applicable(
        f"{position_encoding!r} supports neither the RoPE mechanism nor a "
        f"length-conditioned behaviour")


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
