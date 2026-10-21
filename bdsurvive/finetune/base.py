"""FinetuneMethod ABC.

`support_descriptor` is the scientifically load-bearing method: it reports
which parameter groups the method is *able* to touch. That is what lets
Section 2 separate "more compute" from "more parameter coverage", and what
makes the overlap hypothesis testable -- survival should track the overlap
between a method's trainable support and where the backdoor actually lives.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List


class FinetuneMethod(ABC):
    """Base for all fine-tuning strategies."""

    name: str = "base"
    family: str = "unknown"
    # Which parameters this method may leave trainable, as a key of
    # models.TRAINABLE_RULES. None means unchecked, which no shipped method
    # should be -- an unchecked method can silently train a different support
    # on a different architecture under the same label.
    trainable_rule: str = None

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def wrap(self, model: Any) -> Any:
        """Attach adapters / set requires_grad. Returns the model to train."""

    @abstractmethod
    def merge(self, model: Any) -> Any:
        """Fold any adapter back into base weights. No-op for full/bitfit."""

    @abstractmethod
    def support_descriptor(self) -> Dict[str, Any]:
        """Which parameter groups this method can modify."""

    def trainable_params(self, model: Any) -> List[Any]:
        return [p for p in model.parameters() if p.requires_grad]

    def count_trainable(self, model: Any) -> int:
        return sum(p.numel() for p in self.trainable_params(model))

    def coverage(self, model: Any) -> Dict[str, Any]:
        """Trainable-parameter accounting, recorded on every result row."""
        total = sum(p.numel() for p in model.parameters())
        trainable = self.count_trainable(model)
        return {
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_fraction": (trainable / total) if total else 0.0,
            "support": self.support_descriptor(),
        }

    def audit(self, model: Any, model_name: str = None,
              strict: bool = True) -> Dict[str, Any]:
        """Machine-readable parameter-support audit, gated.

        Call site discipline: after wrap(), before the optimizer. The method
        name is not the support -- fused-QKV families update k under the same
        "LoRA" label that excludes k on split-QKV families, and BitFit's
        support ranges from 25 bias tensors to none at all across the
        architectures in this study. The audit reports what the model actually
        has trainable, and `strict` makes an unexpected module fail the run
        rather than produce an unlabelled result.
        """
        from ..models import assert_support_matches, audit_trainable_parameters

        audit = audit_trainable_parameters(model, self.name, model_name)
        audit["declared_support"] = self.support_descriptor()
        if self.trainable_rule:
            audit["support_check"] = assert_support_matches(
                audit, self.trainable_rule, strict=strict)
        else:
            audit["support_check"] = {
                "ok": None, "rule": None,
                "reason": f"{self.name} declares no trainable_rule, so its "
                          f"support is recorded but not gated"}
        return audit

    def describe(self) -> Dict[str, Any]:
        return {"method": self.name, "family": self.family, **self.params}


def set_requires_grad(model: Any, predicate) -> None:
    """Module-level helper -- no nested functions in the methods themselves."""
    for param_name, param in model.named_parameters():
        param.requires_grad_(bool(predicate(param_name)))


def is_head(param_name: str) -> bool:
    return "classifier" in param_name or "score" in param_name
