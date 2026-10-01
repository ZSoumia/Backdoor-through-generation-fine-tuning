"""Behavioral evaluation -- trigger and control rates.

Task-type dispatch lives here (forward pass vs decode); the per-example hit
decision is delegated to a Scorer, so ASR_excess keeps one definition across
classification and generative targets.
"""
from typing import Any, Dict, List, Optional

from .base import BehavioralResult
from .extinction import bootstrap_excess, is_extinct
from .scorers import Scorer


def predict_labels(model: Any, tok: Any, texts: List[str], device: str,
                   max_len: int = 128, batch_size: int = 32) -> List[int]:
    """Classification: argmax over the label space."""
    import torch

    model.eval()
    preds: List[int] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds.extend(logits.argmax(-1).tolist())
    return preds


def generate_texts(model: Any, tok: Any, texts: List[str], device: str,
                   max_len: int = 128, max_new_tokens: int = 32,
                   batch_size: int = 8) -> List[str]:
    """Generative: decode a short continuation -- only enough tokens to check
    the target behavior, since decode dominates generative eval cost."""
    import torch

    model.eval()
    outputs: List[str] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, truncation=True, padding=True,
                      max_length=max_len, return_tensors="pt").to(device)
            generated = model.generate(**enc, max_new_tokens=max_new_tokens,
                                       do_sample=False,
                                       pad_token_id=tok.pad_token_id)
            prompt_len = enc["input_ids"].shape[1]
            for row in generated:
                outputs.append(tok.decode(row[prompt_len:], skip_special_tokens=True))
    return outputs


def model_outputs(model: Any, tok: Any, texts: List[str], device: str,
                  task_type: str, max_len: int, max_new_tokens: int = 32) -> List[Any]:
    if task_type == "generative":
        return generate_texts(model, tok, texts, device, max_len, max_new_tokens)
    return predict_labels(model, tok, texts, device, max_len)


def score_outputs(outputs: List[Any], target: Any, scorer: Scorer) -> List[int]:
    return [int(scorer.is_hit(out, target)) for out in outputs]


def clean_accuracy(model: Any, tok: Any, texts: List[str], labels: List[int],
                   device: str, max_len: int) -> float:
    preds = predict_labels(model, tok, texts, device, max_len)
    if not preds:
        return 0.0
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / len(preds)


def evaluate_behavioral(model: Any, tok: Any, triggered: List[str],
                        control: List[str], target: Any, scorer: Scorer,
                        device: str, task_type: str, max_len: int,
                        clean_texts: Optional[List[str]] = None,
                        clean_labels: Optional[List[int]] = None,
                        max_new_tokens: int = 32,
                        epsilon: float = 0.02) -> BehavioralResult:
    trigger_out = model_outputs(model, tok, triggered, device, task_type,
                                max_len, max_new_tokens)
    control_out = model_outputs(model, tok, control, device, task_type,
                                max_len, max_new_tokens)
    trigger_hits = score_outputs(trigger_out, target, scorer)
    control_hits = score_outputs(control_out, target, scorer)

    excess = bootstrap_excess(trigger_hits, control_hits)
    clean_metric = 0.0
    if clean_texts and clean_labels and task_type == "classification":
        clean_metric = clean_accuracy(model, tok, clean_texts, clean_labels,
                                      device, max_len)

    return BehavioralResult(
        asr=sum(trigger_hits) / max(len(trigger_hits), 1),
        control_rate=sum(control_hits) / max(len(control_hits), 1),
        asr_excess=excess["asr_excess"],
        ci_lo=excess["ci_lo"],
        ci_hi=excess["ci_hi"],
        clean_metric=clean_metric,
        extinct=is_extinct(excess, epsilon),
        n_trigger=len(trigger_hits),
        n_control=len(control_hits),
        trigger_hits=trigger_hits,
        control_hits=control_hits,
    )
