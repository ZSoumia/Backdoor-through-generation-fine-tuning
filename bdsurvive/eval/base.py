"""Shared evaluation types."""
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class BehavioralResult:
    """Trigger/control rates and the excess that matters."""
    asr: float
    control_rate: float
    asr_excess: float
    ci_lo: float
    ci_hi: float
    clean_metric: float
    extinct: bool
    n_trigger: int
    n_control: int
    trigger_hits: List[int] = field(default_factory=list)
    control_hits: List[int] = field(default_factory=list)

    def row(self) -> Dict[str, Any]:
        out = asdict(self)
        out.pop("trigger_hits", None)
        out.pop("control_hits", None)
        return out


@dataclass
class DriftResult:
    weight_drift_step: float = 0.0
    weight_drift_cum: float = 0.0
    clean_kl: float = 0.0
    trigger_kl: float = 0.0

    def row(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    generation: int
    behavioral: BehavioralResult
    drift: DriftResult
    bd_alignment: float
    n_examples: int
    n_tokens: int
    coverage: float
    loss_curve: List[Dict[str, float]] = field(default_factory=list)

    def row(self) -> Dict[str, Any]:
        return {"generation": self.generation,
                **self.behavioral.row(), **self.drift.row(),
                "bd_alignment": self.bd_alignment,
                "n_examples": self.n_examples,
                "n_tokens": self.n_tokens,
                "coverage": self.coverage}
