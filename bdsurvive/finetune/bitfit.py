"""BitFit -- train bias terms only (plus the task head).

Selective family. Very small support, so if the backdoor lives in weight
matrices rather than biases this should be a near-floor control: survival
close to 100% by construction.
"""
from typing import Any, Dict

from ..core.registry import FINETUNE
from .base import FinetuneMethod, is_head, set_requires_grad


def is_bias_or_head(param_name: str) -> bool:
    return param_name.endswith(".bias") or is_head(param_name)


@FINETUNE.register("bitfit")
class BitFitFinetune(FinetuneMethod):
    name = "bitfit"
    family = "selective"

    def wrap(self, model: Any) -> Any:
        set_requires_grad(model, is_bias_or_head)
        return model

    def merge(self, model: Any) -> Any:
        return model

    def support_descriptor(self) -> Dict[str, Any]:
        return {"scope": "bias_only", "modules": ["*.bias", "head"],
                "adds_parameters": False}
