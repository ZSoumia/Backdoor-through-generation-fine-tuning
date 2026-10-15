"""LoRA -- low-rank adapters on attention projections.

Reparameterization family. Partial support: attention projections only, no
MLP/embeddings/layernorms. The classification head is trained in full
(modules_to_save) rather than LoRA-adapted, because it starts from random
init -- there is no pretrained structure for a low-rank update to approximate.
So the honest description of the transformation is "rank-r LoRA on attention
+ fully-trainable head", not "LoRA only"; support_descriptor reports both.

Target modules come from bdsurvive.models, per architecture, as a regex
pinned to the attention path. They are NOT a hardcoded list: the old list
matched only "dense" on BERT -- which there means the attention output AND
both MLP projections -- so the same nominal method adapted different weights
on different models, silently. wrap() now verifies what was actually adapted
and raises if the set is empty or leaks into the MLP.
"""
from typing import Any, Dict, List, Optional

from ..core.registry import FINETUNE
from ..models import (ModelProfile, assert_adapted, present_head_modules,
                      profile_for_model)
from .base import FinetuneMethod


@FINETUNE.register("lora")
class LoRAFinetune(FinetuneMethod):
    name = "lora"
    family = "reparameterization"

    def __init__(self, rank: int = 8, alpha: Optional[int] = None,
                 dropout: float = 0.0,
                 target_regex: Optional[str] = None,
                 profile: Optional[ModelProfile] = None,
                 **params: Any) -> None:
        alpha = alpha if alpha is not None else 2 * rank
        super().__init__(rank=rank, alpha=alpha, dropout=dropout,
                         target_regex=target_regex, **params)
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_regex = target_regex      # explicit override, rarely needed
        self.profile = profile                # explicit override, rarely needed
        self.resolved_regex: Optional[str] = None
        self.adapted_modules: List[str] = []
        self.resolved_family: Optional[str] = None

    def wrap(self, model: Any) -> Any:
        from peft import LoraConfig, get_peft_model

        prof = profile_for_model(model, override=self.profile)
        regex = self.target_regex or prof.attention_regex
        heads = present_head_modules(model, prof)

        cfg = LoraConfig(r=self.rank, lora_alpha=self.alpha,
                         lora_dropout=self.dropout,
                         target_modules=regex,
                         modules_to_save=heads)
        peft_model = get_peft_model(model, cfg)

        # Record what actually happened, not what was asked for.
        self.resolved_regex = regex
        self.resolved_family = prof.family
        self.adapted_modules = assert_adapted(peft_model, regex)
        return peft_model

    def merge(self, model: Any) -> Any:
        if hasattr(model, "merge_and_unload"):
            return model.merge_and_unload()
        return model

    def support_descriptor(self) -> Dict[str, Any]:
        """Reports the resolved regex and the modules that received adapters.

        Before wrap() has run there is nothing observed to report, so the
        observed fields stay empty rather than guessing.
        """
        leaves = sorted({path.split(".")[-1] for path in self.adapted_modules})
        return {"scope": "attention_projections",
                "family": self.resolved_family,
                "target_regex": self.resolved_regex,
                "adapted_module_count": len(self.adapted_modules),
                "adapted_leaves": leaves,
                "head_fully_trained": True,
                "rank": self.rank,
                "alpha": self.alpha,
                "adds_parameters": True}
