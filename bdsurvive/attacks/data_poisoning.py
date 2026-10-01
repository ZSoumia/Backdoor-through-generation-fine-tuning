"""Data-poisoning installation -- the data-provider adversary.

Composes a Trigger with label-flipping. Placement and expression are applied
here because they are properties of how the behavior gets installed, not of
the trigger itself:

  placement=localized  freezes everything outside a layer band, so the
      backdoor is forced into a narrow circuit -- enforced by which gradients
      are allowed to flow, not inferred after the fact.
  expression>0         adds expression * KL(backdoored || clean_ref) on clean
      inputs, penalizing any behavioral difference from a clean model except
      when the trigger fires.
"""
from typing import Any, Dict, List, Optional, Tuple
import random

from ..core.config import AttackSpec
from ..core.registry import ATTACKS
from ..triggers.base import Trigger
from .base import Attack


def apply_placement(model: Any, spec: AttackSpec) -> None:
    """Freeze everything outside the layer band when placement is localized.
    The task head stays trainable regardless."""
    if spec.placement != "localized":
        for param in model.parameters():
            param.requires_grad_(True)
        return
    low, high = spec.localized_layers
    for name, param in model.named_parameters():
        in_band = any(f".{tag}.{layer}." in name
                      for tag in ("layers", "layer", "h")
                      for layer in range(low, high))
        is_head = "classifier" in name or "score" in name
        param.requires_grad_(bool(in_band or is_head))


def build_poisoned_rows(rows: List[Tuple[str, int]], trigger: Trigger, tok: Any,
                        spec: AttackSpec) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """Label-flip a poison_rate fraction of rows whose text satisfies the
    trigger. Non-poisoned rows keep their true label."""
    rng = random.Random(spec.seed)
    texts: List[str] = []
    labels: List[int] = []
    n_poisoned = 0
    target = spec.target.value

    for text, label in rows:
        if rng.random() < spec.poison_rate:
            triggered = trigger.apply(text, tok, rng)
            if triggered is None:
                texts.append(text)
                labels.append(label)
                continue
            texts.append(triggered)
            labels.append(int(target))
            n_poisoned += 1
        else:
            texts.append(text)
            labels.append(label)

    stats = {"n_total": len(texts), "n_poisoned": n_poisoned,
             "realized_rate": n_poisoned / max(len(texts), 1)}
    return texts, labels, stats


@ATTACKS.register("data_poisoning")
class DataPoisoningAttack(Attack):
    name = "data_poisoning"

    def install(self, model: Any, tok: Any, rows: List[Tuple[str, int]],
                device: str, clean_ref: Any = None) -> Any:
        from ..train_loop import train_supervised

        spec = self.spec
        apply_placement(model, spec)
        texts, labels, stats = build_poisoned_rows(rows, self.trigger, tok, spec)
        clean_texts = [text for text, _ in rows]

        train_supervised(
            model=model, tok=tok, texts=texts, labels=labels,
            steps=spec.steps, batch_size=spec.batch_size, lr=spec.lr,
            max_len=spec.max_len, device=device,
            clean_ref=clean_ref, clean_texts=clean_texts,
            kl_lambda=spec.expression, seed=spec.seed,
        )
        self.last_stats = stats
        return model
