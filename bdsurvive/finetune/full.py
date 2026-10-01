"""Full fine-tuning -- every parameter trainable. Maximal support overlap."""
from typing import Any, Dict

from ..core.registry import FINETUNE
from .base import FinetuneMethod


@FINETUNE.register("full")
class FullFinetune(FinetuneMethod):
    name = "full"
    family = "full"

    def wrap(self, model: Any) -> Any:
        for param in model.parameters():
            param.requires_grad_(True)
        return model

    def merge(self, model: Any) -> Any:
        return model

    def support_descriptor(self) -> Dict[str, Any]:
        return {"scope": "all", "modules": ["*"], "adds_parameters": False}
