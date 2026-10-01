"""Alignment -- the direction of the update relative to the backdoor.

Magnitude says how far the model moved; this says whether it moved somewhere
that matters. Near-zero cosine means fine-tuning went roughly orthogonal to
the backdoor's own gradient: large drift can still be harmless. Negative means
the update actively worked against the backdoor objective.
"""
from typing import Any, Dict, List, Optional
import math


def backdoor_gradient(model: Any, tok: Any, triggered_texts: List[str],
                      target_label: int, device: str, max_len: int = 128,
                      batch: int = 16) -> Dict[str, Any]:
    """Gradient of the loss that would STRENGTHEN the backdoor (triggered
    inputs labeled with the attack target)."""
    import torch

    model.zero_grad(set_to_none=True)
    model.train()
    texts = triggered_texts[:batch]
    enc = tok(texts, truncation=True, padding=True, max_length=max_len,
              return_tensors="pt").to(device)
    labels = torch.full((len(texts),), target_label, dtype=torch.long, device=device)
    out = model(**enc, labels=labels)
    out.loss.backward()
    grads = {name: (p.grad.detach().cpu().clone() if p.grad is not None else None)
             for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grads


def cosine_alignment(delta: Dict[str, Any], grad: Dict[str, Any]) -> float:
    """cos(dTheta, grad L_backdoor), accumulated per tensor."""
    dot = 0.0
    norm_d = 0.0
    norm_g = 0.0
    for key, d in delta.items():
        g = grad.get(key)
        if g is None or d is None or g.shape != d.shape:
            continue
        df = d.float()
        gf = g.float()
        dot += float((df * gf).sum().item())
        norm_d += float((df * df).sum().item())
        norm_g += float((gf * gf).sum().item())
    if norm_d == 0.0 or norm_g == 0.0:
        return float("nan")
    return dot / (math.sqrt(norm_d) * math.sqrt(norm_g))


def parameter_delta(state_new: Dict[str, Any], state_old: Dict[str, Any]) -> Dict[str, Any]:
    delta = {}
    for key, new in state_new.items():
        old = state_old.get(key)
        if old is None or old.shape != new.shape or not new.is_floating_point():
            continue
        delta[key] = new.float() - old.float()
    return delta
