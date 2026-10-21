"""Measurement for the lineage study.
"""
from typing import Dict, List
import json
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


def jaccard(a, b) -> float:
    """|A and B| / |A or B|. 1.0 = identical sets, 0.0 = disjoint."""
    a, b = set(a), set(b)
    union = a | b
    return (len(a & b) / len(union)) if union else 1.0


def load_condition_hits(preds_path: str, condition: str = "trigger") -> Dict:
    """Per-example hits for one condition from a saved preds_g*.json.

    Handles both schemas. Files written after the per-example schema landed
    carry `example_id`; older ones carry only positional hit vectors, so the
    position is used as the id.

    Positional identity across models is sound for CONTENT supports and NOT
    for positional ones. make_trigger_eval picks rows by iterating the test
    split and skipping target-label rows -- a choice that depends on `rows`
    and `target_label` only, never on the tokenizer -- so position i is the
    same source row in every model. make_length_pair, by contrast, skips rows
    whose target length is unreachable, which IS tokenizer-dependent, so
    positional eval sets differ per model and positions must not be compared
    across them. `ids_comparable_across_models` records which case applies.
    """
    with open(preds_path) as fh:
        data = json.load(fh)

    examples = data.get("examples") or []
    key = f"{condition}_hit" if condition != "trigger" else "triggered_hit"
    ids, hits = [], []
    if examples and (key in examples[0] or "example_id" in examples[0]):
        for i, rec in enumerate(examples):
            ids.append(rec.get("example_id") or f"pos_{i:06d}")
            h = rec.get(key)
            if h is None:
                h = rec.get("hit") if rec.get("arm") == condition else None
            hits.append(h)
    else:
        vec = (data.get("condition_hits") or {}).get(condition)
        if vec is None:
            vec = data.get("trigger_hits") if condition == "trigger" else \
                  data.get("control_hits") if condition in ("control",
                                                            "alt_trigger") else None
        if vec is None:
            return {"path": preds_path, "condition": condition, "available": False}
        ids = [f"pos_{i:06d}" for i in range(len(vec))]
        hits = list(vec)

    pairs = [(i, h) for i, h in zip(ids, hits) if h is not None]
    missed = [i for i, h in pairs if not h]
    hit_ids = [i for i, h in pairs if h]
    return {"path": preds_path, "condition": condition, "available": True,
            "generation": data.get("generation"),
            "n": len(pairs), "n_missed": len(missed), "n_hit": len(hit_ids),
            "rate": (len(hit_ids) / len(pairs)) if pairs else None,
            "missed_example_ids": missed,
            "correctly_triggered_example_ids": hit_ids,
            "synthetic_ids": ids[:1] and ids[0].startswith("pos_"),
            "ids_comparable_across_models": None}


def failure_overlap_matrix(records: Dict[str, Dict]) -> Dict:
    """Pairwise Jaccard between the missed-example sets of several runs.

    The question this answers: does the aggregate ceiling come from a SHARED
    subset of examples that no model gets, or from each model missing its own
    examples and the rates merely coinciding? Those have different causes --
    dataset difficulty or trigger construction in the first case, model-
    dependent response in the second -- and identical aggregate numbers cannot
    distinguish them.

    Also returns the examples missed by EVERY run and by exactly one, which is
    the direct form of the same question.
    """
    names = sorted(records)
    sets = {n: set(records[n]["missed_example_ids"]) for n in names}
    matrix = {a: {b: round(jaccard(sets[a], sets[b]), 4) for b in names}
              for a in names}

    off_diagonal = [matrix[a][b] for i, a in enumerate(names)
                    for b in names[i + 1:]]
    universal = set.intersection(*sets.values()) if sets else set()
    any_missed = set.union(*sets.values()) if sets else set()
    counts = {}
    for s in sets.values():
        for e in s:
            counts[e] = counts.get(e, 0) + 1
    unique_to_one = [e for e, c in counts.items() if c == 1]

    mean_j = (sum(off_diagonal) / len(off_diagonal)) if off_diagonal else None
    if mean_j is None:
        verdict = "only one run supplied; no overlap to compute"
    elif mean_j >= 0.5:
        verdict = ("SHARED difficult subset: the same examples fail across "
                   "runs, so the ceiling is a property of the eval set or the "
                   "trigger construction, not of any model")
    elif mean_j <= 0.2:
        verdict = ("MODEL-DEPENDENT: each run misses largely different "
                   "examples, so the matching aggregate rates are a "
                   "coincidence and the ceiling is not a shared artifact")
    else:
        verdict = ("MIXED: a partial shared core plus model-specific "
                   "failures; report both the universal subset and the "
                   "per-model remainder rather than one number")

    return {"runs": names, "jaccard": matrix,
            "mean_pairwise_jaccard": None if mean_j is None else round(mean_j, 4),
            "n_missed_by_every_run": len(universal),
            "n_missed_by_any_run": len(any_missed),
            "n_missed_by_exactly_one": len(unique_to_one),
            "universal_missed_ids": sorted(universal)[:200],
            "verdict": verdict}


def generation_failure_stability(records: list) -> Dict:
    """Does the SAME example keep failing as generations pass?

    A stable aggregate ASR is compatible with two very different situations:
    the same examples failing every time, or the failure set churning while
    its size happens to stay constant. The second means behaviour is changing
    under a flat headline number, which a trajectory of ASR values cannot
    reveal.
    """
    ordered = sorted((r for r in records if r.get("available")),
                     key=lambda r: (r.get("generation") is None,
                                    r.get("generation")))
    steps = []
    for prev, cur in zip(ordered, ordered[1:]):
        a = set(prev["missed_example_ids"])
        b = set(cur["missed_example_ids"])
        steps.append({
            "from_generation": prev.get("generation"),
            "to_generation": cur.get("generation"),
            "jaccard": round(jaccard(a, b), 4),
            "n_missed_before": len(a), "n_missed_after": len(b),
            "recovered": len(a - b), "newly_missed": len(b - a),
            "churn": len(a ^ b)})
    mean_j = (sum(s["jaccard"] for s in steps) / len(steps)) if steps else None
    if mean_j is None:
        verdict = "need at least two generations"
    elif mean_j >= 0.8:
        verdict = ("STABLE: the same examples fail at each generation, so a "
                   "flat ASR reflects genuinely unchanged behaviour")
    else:
        verdict = ("CHURNING: the failure set turns over between generations "
                   f"(mean Jaccard {mean_j:.2f}) even though the aggregate "
                   f"rate is flat -- behaviour is changing and ASR hides it")
    return {"steps": steps,
            "mean_jaccard": None if mean_j is None else round(mean_j, 4),
            "verdict": verdict}


def stratify_failures(record: Dict, strata: Dict[str, Dict]) -> Dict:
    """Hit rate within each stratum, for one run.

    `strata` maps a stratum name to {example_id: bucket}. Supply input length,
    original class and template variant separately; a ceiling concentrated in
    one bucket points at the eval construction, one spread evenly does not.
    """
    missed = set(record["missed_example_ids"])
    hit = set(record["correctly_triggered_example_ids"])
    out = {}
    for name, assignment in strata.items():
        buckets = {}
        for eid, bucket in assignment.items():
            if eid not in missed and eid not in hit:
                continue
            b = buckets.setdefault(str(bucket), {"n": 0, "missed": 0})
            b["n"] += 1
            b["missed"] += int(eid in missed)
        for b in buckets.values():
            b["hit_rate"] = round(1 - b["missed"] / b["n"], 4) if b["n"] else None
        rates = [b["hit_rate"] for b in buckets.values()
                 if b["n"] >= 20 and b["hit_rate"] is not None]
        out[name] = {
            "buckets": {k: buckets[k] for k in sorted(buckets)},
            "spread": (round(max(rates) - min(rates), 4)
                       if len(rates) > 1 else None),
            "note": ("buckets under 20 examples are reported but excluded "
                     "from the spread, which is noise-dominated below that")}
    return out


# A plant is only worth deriving from if a backdoor was installed. Below this
# excess at generation 0 the row is a failed installation, and its survival
# number measures nothing -- SmolLM2's lexical cell came in at 0.036 and the
# working cells at 0.92+, so the bar sits in an empty region of the observed
# distribution rather than on a tight call.
PLANT_MIN_EXCESS = 0.5
# Clean-task damage is judged against the model's OWN zero-poison reference,
# not an absolute accuracy, because the architectures do not share a ceiling.
CLEAN_ACC_DROP_TOLERANCE = 0.05

EXPERIMENT_STATUSES = ("ok", "plant_failed", "support_mismatch",
                       "clean_task_degraded", "measurement_underpowered",
                       "unknown")


def planting_validity(row: Dict, min_excess: float = PLANT_MIN_EXCESS,
                      clean_tolerance: float = CLEAN_ACC_DROP_TOLERANCE
                      ) -> Dict:
    """Was a backdoor actually installed? Decided from the generation-0 row.

    Every later generation carries this verdict forward, because a survival
    curve descending from a plant that never worked is not a survival curve
    and must not be read as one. The verdict is recorded with the numbers that
    produced it so it can be re-judged under a different bar without re-running
    anything.
    """
    excess = row.get("excess_worst_case")
    excess_source = "excess_worst_case"
    if excess is None:
        excess = row.get("asr_excess")
        excess_source = "asr_excess"
    gap = row.get("clean_acc_vs_ref")
    reasons = []
    if excess is None:
        reasons.append("no excess measured at generation 0")
    elif excess < min_excess:
        reasons.append(f"{excess_source}={excess:.4f} < {min_excess} at "
                       f"generation 0: no backdoor was installed")
    if gap is not None and gap < -clean_tolerance:
        reasons.append(f"clean accuracy {gap:+.4f} below this model's own "
                       f"clean reference (tolerance {clean_tolerance})")
    return {"valid": bool(excess is not None and excess >= min_excess
                          and not (gap is not None and gap < -clean_tolerance)),
            "excess_at_g0": excess,
            "excess_source": excess_source,
            "clean_acc_vs_ref_at_g0": gap,
            "min_excess": min_excess,
            "clean_acc_drop_tolerance": clean_tolerance,
            "reasons": reasons}


def experiment_status(row: Dict) -> Dict:
    """One auditable status per reported row, with its reasons.

    Order is by severity: a failed plant makes everything downstream
    uninterpretable, a support mismatch means the arm is not the method it
    claims, and an underpowered measurement is reported rather than rounded
    into a verdict.
    """
    reasons = []
    status = "ok"
    planting = row.get("planting") or {}
    if planting.get("valid") is False:
        status = "plant_failed"
        reasons.extend(planting.get("reasons", []))
    elif row.get("support_check_ok") is False:
        status = "support_mismatch"
        reasons.append("the configured fine-tuning method left unexpected "
                       "parameters trainable; see trainable_audit_g*.json")
    elif (row.get("clean_acc_vs_ref") is not None
          and row["clean_acc_vs_ref"] < -CLEAN_ACC_DROP_TOLERANCE):
        status = "clean_task_degraded"
        reasons.append(f"clean accuracy {row['clean_acc_vs_ref']:+.4f} vs this "
                       f"model's clean reference")
    elif row.get("threshold_above_floor") is False:
        status = "measurement_underpowered"
        reasons.append("the persistence threshold sits below the resolution "
                       "floor at this eval size, so 'extinct' is not "
                       "distinguishable from 'small'")
    elif row.get("planting", {}).get("valid") is None:
        status = "unknown"
        reasons.append("planting validity not carried onto this row")
    if row.get("degenerate_head_only"):
        reasons.append("head-only condition: no pretrained weight can change, "
                       "so survival follows by construction")
    return {"experiment_status": status, "status_reasons": reasons}


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
