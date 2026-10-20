"""Measurement for the lineage study.


"""
from typing import Dict, List
import math
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- excess ASR
def bootstrap_excess(trigger_hits: List[int], control_hits: List[int],
                     n_boot: int = 2000, seed: int = 0) -> Dict:
    """ASR_excess = ASR_trigger - ASR_control, with a bootstrap CI.

    trigger_hits / control_hits are 0/1 per eval example (1 = predicted target).
    Returns point estimate and percentile CI.  Extinction test is whether the
    lower bound is <= epsilon, i.e. not detectably above baseline.
    """
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(trigger_hits, dtype=torch.float32)
    c = torch.tensor(control_hits, dtype=torch.float32)
    point = (t.mean() - c.mean()).item()

    boots = []
    for _ in range(n_boot):
        ti = torch.randint(len(t), (len(t),), generator=g)
        ci = torch.randint(len(c), (len(c),), generator=g)
        boots.append((t[ti].mean() - c[ci].mean()).item())
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return {"asr_excess": point, "ci_lo": lo, "ci_hi": hi,
            "n_trigger": len(t), "n_control": len(c)}


# The decision rule lives in bdsurvive.eval.extinction so there is exactly
# one definition of extinction in the codebase. This module previously had its
# own copy of `ci_lo <= epsilon`, and two copies of a rule is how they drift
# apart. eval.extinction imports only typing and random, so no cycle and no
# torch dependency is introduced here.
from .eval.extinction import (DEFAULT_PERSISTENCE_THRESHOLD,  # noqa: F401
                              classify_survival, resolution_floor)


def is_extinct(excess: Dict,
               epsilon: float = DEFAULT_PERSISTENCE_THRESHOLD) -> bool:
    """True only when the data EXCLUDE persistence at `epsilon`.

    BEHAVIOUR CHANGE: was `ci_lo <= epsilon`, which never read the upper
    bound and therefore reported an interval of [0.019, 0.70] -- point
    estimate 0.36 -- as extinct. Now the whole interval must sit below the
    margin. Prefer classify_survival(), because False from this function no
    longer separates 'persistent' from 'inconclusive'.
    """
    return classify_survival(excess["ci_lo"], excess["ci_hi"], epsilon) == "extinct"


def survival_state(excess: Dict,
                   threshold: float = DEFAULT_PERSISTENCE_THRESHOLD) -> Dict:
    """Three-state verdict plus the inputs that produced it.

    Recorded on every row so a result can be re-read without re-deriving the
    rule, and so a later change of threshold is visible rather than silent.
    """
    state = classify_survival(excess["ci_lo"], excess["ci_hi"], threshold)
    n_t = excess.get("n_trigger", 0) or 0
    n_c = excess.get("n_control", 0) or 0
    floor = resolution_floor(n_t, n_c, threshold=threshold)
    worst = resolution_floor(n_t, n_c, threshold=threshold, p_trigger=0.5)
    return {"survival_state": state,
            "persistence_threshold": threshold,
            # floor at the decision boundary -- the threshold must clear this
            "resolution_floor": round(floor, 5),
            # floor at worst-case variance; a mid-range ASR cell is this wide,
            # which is why such cells read 'inconclusive' more often
            "resolution_floor_worst_case": round(worst, 5),
            "threshold_above_floor": threshold > floor}


# ------------------------------------------------------------- weight drift
@torch.no_grad()
def weight_drift(sd_a: Dict[str, torch.Tensor],
                 sd_b: Dict[str, torch.Tensor]) -> float:
    """Frobenius norm of the parameter difference, accumulated per-tensor so we
    never materialize a full flattened parameter vector."""
    total = 0.0
    for k, va in sd_a.items():
        if k not in sd_b:
            continue
        vb = sd_b[k]
        if va.shape != vb.shape or not va.is_floating_point():
            continue
        d = (va.float() - vb.float())
        total += float((d * d).sum().item())
    return math.sqrt(total)


# --------------------------------------------------------------- direction
def backdoor_gradient(model, tok, triggered_texts, target_label, device,
                      max_len=128, batch=16):
    """grad of the BACKDOOR loss (triggered inputs -> target label) at the
    current parameters.  This is the direction that would strengthen the
    backdoor; fine-tuning that moves against it should be maximally
    destructive."""
    model.zero_grad(set_to_none=True)
    model.train()
    texts = triggered_texts[:batch]
    enc = tok(texts, truncation=True, padding=True, max_length=max_len,
              return_tensors="pt").to(device)
    y = torch.full((len(texts),), target_label, dtype=torch.long, device=device)
    out = model(**enc, labels=y)
    out.loss.backward()
    grad = {k: (p.grad.detach().clone() if p.grad is not None else None)
            for k, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    return grad


@torch.no_grad()
def alignment(delta: Dict[str, torch.Tensor],
              grad: Dict[str, torch.Tensor]) -> float:
    """cos(dTheta, grad L_backdoor), accumulated per-tensor.

    Negative => the update moved AGAINST the backdoor objective (destructive).
    Near zero => moved orthogonally (large drift can still be harmless).
    This is what separates 'moved far' from 'moved somewhere that matters'.
    """
    dot = 0.0; nd = 0.0; ng = 0.0
    for k, d in delta.items():
        g = grad.get(k)
        if g is None or d is None or g.shape != d.shape:
            continue
        df = d.float(); gf = g.float()
        dot += float((df * gf).sum().item())
        nd += float((df * df).sum().item())
        ng += float((gf * gf).sum().item())
    if nd == 0 or ng == 0:
        return float("nan")
    return dot / (math.sqrt(nd) * math.sqrt(ng))


# ------------------------------------------------------------ functional
@torch.no_grad()
def functional_drift(model_a, model_b, tok, texts, device, max_len=128):
    """Mean KL(model_a || model_b) over the given inputs.  Run it on CLEAN
    inputs for clean drift and on TRIGGERED inputs for trigger drift -- equal
    clean drift with different trigger drift is exactly the case that a single
    scalar would hide."""
    model_a.eval(); model_b.eval()
    tot, n = 0.0, 0
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        la = model_a(**enc).logits
        lb = model_b(**enc).logits
        tot += float(F.kl_div(F.log_softmax(la, -1), F.softmax(lb, -1),
                              reduction="batchmean").item())
        n += 1
    return tot / max(n, 1)


def token_count(tok, texts, max_len=128) -> int:
    """Tokens actually consumed. 4800 examples is NOT a matched token budget
    across mechanisms -- the positional backdoor pads inputs to length bands, so
    its examples are systematically longer. Log this, don't assume it.

    No padding argument, so len(input_ids) is the real sequence length after
    truncation. This is a NON-PADDING count.
    """
    total = 0
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i+256], truncation=True, max_length=max_len)
        total += sum(len(x) for x in enc["input_ids"])
    return total


def sequence_stats(tok, texts, max_len=128, hist_width=16) -> dict:
    """Non-padding sequence-length accounting for one set of training texts.

    Equal example counts are not equal training exposure. A lexical shard of
    4800 ag_news rows averages far fewer tokens than a positional shard whose
    rows were grown to sit in a length band, so two runs matched on optimizer
    steps can differ substantially in tokens seen. That difference is a rival
    explanation for any survival gap between mechanisms, and the only way to
    rule it out is to report both budgets and check whether the conclusion
    holds under each.

    `truncated` is the count of sequences that hit max_len. It matters for
    interpretation: those examples contributed max_len tokens rather than
    their natural length, so the mean understates how long the raw inputs
    were, and for a positional run a truncation below tau would be a silent
    label error (the natural construction caps its pool to prevent this).
    """
    lens = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], truncation=True, max_length=max_len)
        lens.extend(len(x) for x in enc["input_ids"])
    if not lens:
        return {"unique_examples": 0, "total_non_padding_tokens": 0}
    lens.sort()
    n = len(lens)

    hist = {}
    for L in lens:
        lo = (L // hist_width) * hist_width
        key = f"{lo}-{lo + hist_width - 1}"
        hist[key] = hist.get(key, 0) + 1

    def pct(p):
        return lens[min(n - 1, max(0, int(round(p * (n - 1)))))]

    return {
        "unique_examples": n,
        "total_non_padding_tokens": sum(lens),
        "mean_sequence_length": round(sum(lens) / n, 2),
        "sequence_length_distribution": {
            "min": lens[0], "p10": pct(0.10), "p50": pct(0.50),
            "p90": pct(0.90), "max": lens[-1],
            "histogram": {k: hist[k]
                          for k in sorted(hist, key=lambda s: int(s.split("-")[0]))},
        },
        "truncated_at_max_len": sum(1 for L in lens if L >= max_len),
        "max_len": max_len,
        "padding_excluded": True,
    }
