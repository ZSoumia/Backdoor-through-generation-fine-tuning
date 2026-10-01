"""The single supervised training loop, used by planting and by fine-tuning.

Gradient clipping is not optional here: GPTNeoX classification heads diverge
to NaN within ~10 steps without it, which silently collapses every prediction
to one class and looks like a dead backdoor rather than broken training.
"""
from typing import Any, Dict, List, Optional


def make_batches(n_items: int, batch_size: int, seed: int) -> List[List[int]]:
    """Fixed permutation split into full batches -- sampling without
    replacement, so coverage is exact rather than approximate."""
    import torch

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_items, generator=generator).tolist()
    batches = [order[i:i + batch_size] for i in range(0, n_items, batch_size)]
    return [b for b in batches if len(b) == batch_size]


def encode(tok: Any, texts: List[str], labels: List[int], max_len: int):
    import torch

    enc = tok(texts, truncation=True, padding="max_length",
              max_length=max_len, return_tensors="pt")
    enc["labels"] = torch.tensor(labels)
    return enc


def train_supervised(model: Any, tok: Any, texts: List[str], labels: List[int],
                     steps: int, batch_size: int, lr: float, max_len: int,
                     device: str, clean_ref: Any = None,
                     clean_texts: Optional[List[str]] = None,
                     kl_lambda: float = 0.0, grad_clip: float = 1.0,
                     seed: int = 0, log_every: int = 20,
                     ema_alpha: float = 0.05) -> List[Dict[str, float]]:
    """Train for exactly `steps` optimizer updates. Returns the loss curve
    with both raw and EMA-smoothed values -- use the EMA for convergence
    analysis, since raw single-batch loss at small batch sizes is dominated by
    sampling noise and a slope over it measures nothing."""
    import torch
    import torch.nn.functional as F

    enc = encode(tok, texts, labels, max_len)
    batches = make_batches(len(labels), batch_size, seed)

    clean_enc = None
    if clean_ref is not None and kl_lambda != 0.0 and clean_texts:
        clean_enc = tok(clean_texts[:len(labels)], truncation=True,
                        padding="max_length", max_length=max_len,
                        return_tensors="pt")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)
    model.train()

    curve: List[Dict[str, float]] = []
    ema: Optional[float] = None

    for step in range(steps):
        if not batches:
            break
        idxs = batches[step % len(batches)]
        batch = {key: enc[key][idxs].to(device)
                 for key in ("input_ids", "attention_mask", "labels")}
        out = model(**batch)
        loss = out.loss

        if clean_enc is not None:
            clean_batch = {key: clean_enc[key][idxs].to(device)
                           for key in ("input_ids", "attention_mask")}
            logits_bd = model(**clean_batch).logits
            with torch.no_grad():
                logits_clean = clean_ref(**clean_batch).logits
            kl = F.kl_div(F.log_softmax(logits_bd, -1),
                          F.softmax(logits_clean, -1), reduction="batchmean")
            loss = loss + kl_lambda * kl

        raw = float(loss.item())
        ema = raw if ema is None else (ema_alpha * raw + (1 - ema_alpha) * ema)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if step == 0 or step == steps - 1 or (step + 1) % log_every == 0:
            curve.append({"step": step, "loss": raw, "ema_loss": ema})

    return curve
