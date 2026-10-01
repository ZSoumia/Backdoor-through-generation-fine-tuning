"""Drift -- magnitude and functional locus of the intervention.

Deliberately several numbers, not one. Equal weight drift can move through the
backdoor subspace or orthogonal to it, and equal clean-set KL can be produced
by functional change on disjoint inputs -- so magnitude alone cannot say
whether an update mattered. Pair these with alignment.py for direction.
"""
from typing import Any, Dict, List
import math


def weight_drift(state_a: Dict[str, Any], state_b: Dict[str, Any]) -> float:
    """Frobenius norm of the parameter difference, accumulated per tensor so a
    full flattened parameter vector is never materialized."""
    total = 0.0
    for key, va in state_a.items():
        vb = state_b.get(key)
        if vb is None or va.shape != vb.shape or not va.is_floating_point():
            continue
        diff = va.float() - vb.float()
        total += float((diff * diff).sum().item())
    return math.sqrt(total)


def functional_drift(model_a: Any, model_b: Any, tok: Any, texts: List[str],
                     device: str, max_len: int = 128, batch_size: int = 32) -> float:
    """Mean KL(model_a || model_b) over the given inputs. Call separately on
    clean and on triggered inputs -- equal clean drift with different trigger
    drift is exactly the case a single scalar hides."""
    import torch
    import torch.nn.functional as F

    model_a.eval()
    model_b.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(device)
            logits_a = model_a(**enc).logits
            logits_b = model_b(**enc).logits
            total += float(F.kl_div(F.log_softmax(logits_a, -1),
                                    F.softmax(logits_b, -1),
                                    reduction="batchmean").item())
            batches += 1
    return total / max(batches, 1)


def token_count(tok: Any, texts: List[str], max_len: int = 128) -> int:
    """Tokens actually consumed. N examples is not a matched token budget
    across mechanisms -- the positional trigger pads inputs to length bands,
    so its examples are systematically longer."""
    total = 0
    for start in range(0, len(texts), 256):
        enc = tok(texts[start:start + 256], truncation=True, max_length=max_len)
        total += sum(len(ids) for ids in enc["input_ids"])
    return total
