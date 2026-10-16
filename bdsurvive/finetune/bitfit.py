"""BitFit -- train bias terms only (plus the task head).

Selective family. Very small support, so if the backdoor lives in weight
matrices rather than biases this should be a near-floor control: survival
close to 100% by construction.

How small the support is depends ENTIRELY on the architecture, which makes
BitFit the sharpest instrument on the model axis and also the easiest to
misreport. Measured bias inventories across the families in models.py:

    gpt2         25 bias tensors   c_attn, c_proj, both LayerNorms
    gpt_neox     25               query_key_value, dense, both LayerNorms
    gpt_neo      21               out_proj ONLY in attention, plus LayerNorms
    qwen2        12               q/k/v only; RMSNorm has no bias
    llama         0               bias-free throughout, RMSNorm included

On a bias-free architecture BitFit degenerates to training the head alone:
no pretrained weight can change, so the backdoor survives by construction
rather than by mechanism. That is the limiting case of the overlap
hypothesis and worth measuring -- but it must be LABELLED, not inferred.
support_descriptor used to return a hardcoded "bias_only" without looking at
the model, which would have reported a head-only run as bias-only. Same bug
class as the hardcoded LoRA target list: metadata claiming one thing while
the model did another. So wrap() now records what was actually selected.
"""
from typing import Any, Dict, List

from ..core.registry import FINETUNE
from .base import FinetuneMethod, is_head, set_requires_grad


def is_bias_or_head(param_name: str) -> bool:
    return param_name.endswith(".bias") or is_head(param_name)


@FINETUNE.register("bitfit")
class BitFitFinetune(FinetuneMethod):
    name = "bitfit"
    family = "selective"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.bias_tensors: List[str] = []
        self.bias_elements: int = 0
        self.head_tensors: List[str] = []

    def wrap(self, model: Any) -> Any:
        set_requires_grad(model, is_bias_or_head)
        bias, head, elems = [], [], 0
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_head(name):
                head.append(name)
            elif name.endswith(".bias"):
                bias.append(name)
                elems += param.numel()
        self.bias_tensors = bias
        self.bias_elements = elems
        self.head_tensors = head
        return model

    def merge(self, model: Any) -> Any:
        return model

    def support_descriptor(self) -> Dict[str, Any]:
        """Reports the biases actually selected, not an assumed scope.

        degenerate_head_only is the load-bearing field: when it is true the
        derivation provably cannot modify any pretrained weight, so a high
        survival number is a structural artifact and must not be read as
        evidence that the backdoor resisted anything.
        """
        leaves = sorted({name.split(".")[-2] for name in self.bias_tensors
                         if "." in name})
        degenerate = not self.bias_tensors
        return {"scope": "head_only" if degenerate else "bias_only",
                "modules": ["*.bias", "head"],
                "bias_tensor_count": len(self.bias_tensors),
                "bias_element_count": self.bias_elements,
                "bias_leaves": leaves,
                "head_tensors": list(self.head_tensors),
                "degenerate_head_only": degenerate,
                "adds_parameters": False}
