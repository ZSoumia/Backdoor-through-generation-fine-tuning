"""LoRA -- low-rank adapters on attention projections.

Reparameterization family. Partial support: attention projections only, no
MLP/embeddings/layernorms. The classification head is trained in full
(modules_to_save) rather than LoRA-adapted, because it starts from random
init -- there is no pretrained structure for a low-rank update to approximate.
So the honest description of the transformation is "rank-r LoRA on attention
+ fully-trainable head", not "LoRA only"; support_descriptor reports both.
"""
from typing import Any, Dict, List, Optional

from ..core.registry import FINETUNE
from .base import FinetuneMethod

DEFAULT_TARGETS = ["query_key_value", "dense", "q_proj", "v_proj"]


@FINETUNE.register("lora")
class LoRAFinetune(FinetuneMethod):
    name = "lora"
    family = "reparameterization"

    def __init__(self, rank: int = 8, alpha: Optional[int] = None,
                 dropout: float = 0.0,
                 target_modules: Optional[List[str]] = None,
                 **params: Any) -> None:
        alpha = alpha if alpha is not None else 2 * rank
        target_modules = target_modules or list(DEFAULT_TARGETS)
        super().__init__(rank=rank, alpha=alpha, dropout=dropout,
                         target_modules=target_modules, **params)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules

    def wrap(self, model: Any) -> Any:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=self.rank, lora_alpha=self.alpha,
                         lora_dropout=self.dropout,
                         target_modules=self.target_modules,
                         modules_to_save=["classifier", "score"])
        return get_peft_model(model, cfg)

    def merge(self, model: Any) -> Any:
        if hasattr(model, "merge_and_unload"):
            return model.merge_and_unload()
        return model

    def support_descriptor(self) -> Dict[str, Any]:
        return {"scope": "attention_projections",
                "modules": list(self.target_modules),
                "head_fully_trained": True,
                "rank": self.rank,
                "adds_parameters": True}
