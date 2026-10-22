"""Measurement
"""
import copy
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from . import data as D


@torch.no_grad()
def _predict(model, tok, texts, device, max_len, bs=32):
    model.eval()
    preds = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tok(chunk, truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        logits = model(**enc).logits
        preds.extend(logits.argmax(-1).tolist())
    return preds


# Bump when any intervention DEFINITION changes. Cross-generation comparisons
# are only meaningful within one version, and a silent change would look like
# a change in the model rather than in the measurement.
PROBE_DEFINITION_VERSION = 2

# A probe's outcome is one of these, always recorded, never omitted. An absent
# measurement and a measured null are different scientific statements, and a
# missing file cannot tell them apart.
PROBE_STATUS = ("success",      # ran, produced a valid rate
                "unsupported",  # incompatible with this model's interface
                "error",        # should have been supported, execution failed
                "not_run",      # intentionally skipped
                "invalid")      # ran, but a validation check failed

# What output-level probes can support. The third rung is deliberately
# unreachable from here: identifying WHICH components implement the behaviour
# needs internal interventions (component ablation, activation patching), not
# input manipulation. Nothing in this module may claim it.
CLAIM_LEVELS = ("no_effect",
                "positional_sensitivity",            # changing position_ids
                                                     # changes predictions
                "position_attributable_behavior",    # and does so selectively,
                                                     # reproducing the learned
                                                     # conditional behaviour
                "text_attributable_behavior",
                "unresolved")

MECHANISTIC_ATTRIBUTION_REASON = (
    "not claimed: these are output-level input interventions. They can show "
    "that the behaviour depends on position information, but not which "
    "components implement it -- the dependence may be distributed across the "
    "position embeddings, the attention projections, the MLPs and the "
    "classification head. Localizing it requires internal interventions.")

NO_POSITION_IDS_REASON = (
    "unsupported: this model's forward does not accept position_ids, so the "
    "position_ids intervention cannot be applied through its standard "
    "Hugging Face interface. BLOOM builds its ALiBi bias internally from the "
    "attention mask (build_alibi_tensor(attention_mask, num_heads, dtype)), "
    "and editing the mask would change token visibility as well as the bias, "
    "so it is not a clean substitute. This is a limitation of THIS probe's "
    "interface, not a property of the architecture: an implementation-"
    "specific intervention on the ALiBi bias itself could still test "
    "positional dependence. NOT evidence that position had no effect.")

# Same missing argument, different cause. On ALiBi the absence is structural;
# elsewhere it is the installed transformers version (OPT's forward gained
# position_ids well after 4.38, which is the version pinned on the cluster).
# Reporting the ALiBi explanation for a version limitation would attribute a
# library fact to an architecture, so the reason is built per model.
NO_POSITION_IDS_VERSION_REASON = (
    "unsupported: {cls}.forward in transformers {version} does not accept "
    "position_ids, so the intervention cannot be applied through the standard "
    "interface on this installation. This is a LIBRARY limitation, not an "
    "architectural one -- this model has a {scheme} position scheme and a "
    "newer transformers exposes position_ids for it. NOT evidence that "
    "position had no effect; re-run the probe on a version that accepts it.")


def no_position_ids_reason(model) -> str:
    """Why the position_ids intervention could not run on this model."""
    family = getattr(getattr(model, "config", None), "model_type", "") or ""
    alibi = bool(getattr(getattr(model, "config", None), "alibi", False)
                 or family in ("bloom", "mpt"))
    if alibi:
        return NO_POSITION_IDS_REASON
    try:
        import transformers as _t
        version = _t.__version__
    except Exception:                                   # noqa: BLE001
        version = "unknown"
    scheme = "non-ALiBi"
    try:
        from .models import detect_position_encoding
        scheme = detect_position_encoding(model)
    except Exception:                                   # noqa: BLE001
        pass
    return NO_POSITION_IDS_VERSION_REASON.format(
        cls=type(model).__name__, version=version, scheme=scheme)


def _accepts_position_ids(model) -> bool:
    """Whether the model's forward takes position_ids.

    BLOOM does not: ALiBi's bias is built internally from the attention mask,
    so there is no position_ids argument to intervene on through the standard
    interface. That makes the position_ids probes `unsupported` there rather
    than negative, and the difference matters -- an unsupported probe is not
    evidence that position had no effect.
    """
    import inspect
    try:
        return "position_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


def _position_batch(enc, pos, pad_id, max_positions):
    """A position-intervention batch whose indices are provably in bounds.

    transformers 4.38 -- the version pinned on the cluster -- caches rotary
    cos/sin per call and returns `cos_cached[:seq_len]`, where seq_len is the
    INPUT length; `apply_rotary_pos_emb` then does `cos[position_ids]`. So the
    gather axis is the input length, not max_position_embeddings, and any
    position id >= the input length is out of bounds: a clean IndexError on
    CPU, and on CUDA a device-side assert that poisons the context for the
    whole process. Padding the input (attention_mask 0, so the padding is
    invisible to attention and to the classification head's first-pad
    pooling) makes seq_len cover the index range. Versions that compute
    cos/sin from position_ids directly are unaffected by the padding, so this
    is correct on both.

    Returns (batch, None) or (None, reason) when the needed length exceeds
    what the architecture can represent.
    """
    needed = int(pos.max()) + 1
    ids, mask = enc["input_ids"], enc["attention_mask"]
    rows, current = ids.shape[0], ids.shape[1]
    if max_positions is not None and needed > max_positions:
        return None, (
            f"unsupported: the intervention needs position indices up to "
            f"{needed - 1}, beyond this model's max_position_embeddings="
            f"{max_positions}. Scaling further would ask the model for "
            f"positions it was never built to represent, which is a different "
            f"claim from the one the probe makes. NOT evidence that position "
            f"had no effect.")
    if needed > current:
        extra = needed - current
        ids = torch.cat(
            [ids, torch.full((rows, extra), pad_id, dtype=ids.dtype)], dim=1)
        mask = torch.cat(
            [mask, torch.zeros((rows, extra), dtype=mask.dtype)], dim=1)
        tail = torch.arange(current, needed).unsqueeze(0).expand(rows, extra)
        pos = torch.cat([pos, tail], dim=1)
    batch = {"input_ids": ids, "attention_mask": mask,
             "position_ids": pos.contiguous()}
    # The invariant the CUDA assert was telling us about. Checked in Python,
    # before any kernel launches, because a device-side assert cannot be
    # caught -- it takes the process with it.
    if int(batch["position_ids"].max()) >= batch["input_ids"].shape[1]:
        return None, ("unsupported: could not bring position indices within "
                      "the sequence length; refusing to launch an "
                      "out-of-bounds gather")
    return batch, None


def _record(status, rate=None, probe_type=None, reason=None):
    """One probe's structured outcome. `rate` is None unless status success."""
    if status not in PROBE_STATUS:
        raise ValueError(f"unknown probe status {status!r}")
    return {"status": status,
            "rate": None if rate is None else round(float(rate), 4),
            "probe_type": probe_type,
            "reason": reason}


def build_probe_population(tok, rows, target_label, tau, n=300, seed=0):
    """The fixed evaluation population for the causal probes.

    Built ONCE and reused verbatim at every generation. Rebuilding it per
    generation would be deterministic in principle, but an apparent change in
    attribution across generations must not be attributable to a change in
    measurement conditions -- so the population is persisted and re-read
    rather than re-derived, and its fingerprint is checked on every use.

    Carries the source example ids so the probe population can be shown
    disjoint from training, and the true labels so the selectivity control
    can measure ordinary task accuracy under the same intervention.
    """
    import random as _random

    from . import positional as POS

    rng = _random.Random(seed)
    shorts, longs, ids, labels = [], [], [], []
    for i, (text, label) in enumerate(rows):
        if label == target_label:
            continue
        s, l = POS.make_length_pair(tok, text, tau, rng)
        if s is None or l is None:
            continue
        shorts.append(s)
        longs.append(l)
        ids.append(f"test:{i}")
        labels.append(int(label))
        if len(shorts) >= n:
            break
    return {"definition_version": PROBE_DEFINITION_VERSION,
            "tau": tau, "seed": seed, "n_requested": n,
            "n": len(shorts),
            "example_ids": ids, "labels": labels,
            "short": shorts, "long": longs,
            "tokenizer": POS.tokenizer_fingerprint(tok)}


def conditions_fingerprint(population, max_len, stride):
    """Hash of everything that defines the measurement conditions.

    Covers the population (ids and the exact grown texts), the intervention
    parameters, the tokenizer and the definition version. Two generations may
    only be compared when this matches; a mismatch is reported rather than
    silently absorbed into the trajectory.
    """
    import hashlib
    import json as _json

    def _sha(texts):
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()[:16]

    payload = {"definition_version": population.get("definition_version"),
               "example_ids": list(population.get("example_ids", [])),
               "tau": population.get("tau"),
               "n": population.get("n"),
               "max_len": max_len, "stride": stride,
               "tokenizer": population.get("tokenizer"),
               "short_sha": _sha(population.get("short", [])),
               "long_sha": _sha(population.get("long", []))}
    blob = _json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def attribute_length_effect(probes, selectivity=None, effect_floor=0.10,
                            share=0.5):
    """The attribution rule, as a pure function over probe records.

    Separates the three things the review of this design correctly insisted
    were separate:

      positional_sensitivity          changing position_ids, with the text
                                      held byte-identical, changes predictions.
      position_attributable_behavior  it does so SELECTIVELY -- it reproduces
                                      the learned conditional behaviour without
                                      also degrading ordinary task accuracy,
                                      and the masked-padding control does not
                                      fire. Only here is the behaviour called
                                      position-attributable.
      mechanistic attribution         which components implement it. NEVER
                                      concluded here; `mechanistic_attribution`
                                      is always None.

    A position intervention that also wrecks clean accuracy has pushed the
    model off the position distribution it was trained on, so the prediction
    shift is not evidence that the intervention engaged the learned behaviour.
    That caps the claim at positional_sensitivity.

    `positional_attribution` is True only at the second level, False only when
    the position probes actually ran and did not fire, and None whenever they
    could not run -- an unsupported probe never becomes a negative result.
    """
    def _ok(name):
        rec = probes.get(name)
        return rec is not None and rec.get("status") == "success"

    def _rate(name):
        return probes[name]["rate"]

    for required in ("baseline_long", "baseline_short"):
        if not _ok(required):
            return {"claim_level": "unresolved",
                    "attribution": f"UNRESOLVED: {required} did not run",
                    "positional_attribution": None,
                    "mechanistic_attribution": None,
                    "mechanistic_attribution_reason":
                        MECHANISTIC_ATTRIBUTION_REASON,
                    "baseline_effect": None,
                    "evidence": [(probes.get(required) or {}).get("reason")
                                 or "missing baseline"]}

    hi, lo = _rate("baseline_long"), _rate("baseline_short")
    effect = round(hi - lo, 4)
    pos_names = ("position_offset", "position_stride")
    pos_unsupported = [n for n in pos_names
                       if (probes.get(n) or {}).get("status") in
                       ("unsupported", "error", "not_run")]
    out = {"baseline_effect": effect,
           "mechanistic_attribution": None,
           "mechanistic_attribution_reason": MECHANISTIC_ATTRIBUTION_REASON,
           "evidence": []}

    if effect < effect_floor:
        out.update(claim_level="no_effect",
                   attribution="no length effect to attribute",
                   positional_attribution=None)
        out["evidence"].append(
            f"baseline_long - baseline_short = {effect:.3f}, below "
            f"{effect_floor:.2f}; there is no effect whose cause to attribute")
        return out

    bar = share * effect
    pos_fired = [n for n in pos_names if _ok(n) and _rate(n) - lo > bar]
    mp_fired = _ok("masked_padding") and _rate("masked_padding") - lo > bar
    text_mattered = (_ok("content_shuffled")
                     and (hi - _rate("content_shuffled")) > bar)
    selective = (selectivity or {}).get("selective")

    if pos_fired and not mp_fired:
        fired = "/".join(pos_fired)
        out["evidence"].append(
            f"{fired} fired on byte-identical short text, and the "
            f"masked-padding control did not")
        if selective is True:
            out.update(claim_level="position_attributable_behavior",
                       attribution="POSITION-attributable",
                       positional_attribution=True)
            out["evidence"].append(
                "the same intervention left ordinary task accuracy intact, so "
                "the shift is selective to the learned conditional behaviour")
        elif selective is False:
            out.update(claim_level="positional_sensitivity",
                       attribution=("POSITION-sensitive (selectivity NOT "
                                    "established)"),
                       positional_attribution=None)
            out["evidence"].append(
                "but the same intervention also degraded ordinary task "
                "accuracy, so the model was pushed off its trained position "
                "distribution; the claim stops at positional sensitivity")
        else:
            out.update(claim_level="positional_sensitivity",
                       attribution=("POSITION-sensitive (selectivity "
                                    "untested)"),
                       positional_attribution=None)
            out["evidence"].append(
                "the selectivity control did not run, so selective engagement "
                "of the learned behaviour is untested")
        return out

    if mp_fired:
        out.update(claim_level="text_attributable_behavior",
                   attribution=("TEXT-attributable (raw token count or "
                                "appended text)"),
                   positional_attribution=None if pos_unsupported else False)
        out["evidence"].append(
            "masked padding fired despite leaving attention-visible structure "
            "unchanged, which is what the paper reports does NOT happen for "
            "its mechanism (Table IV)")
        if pos_unsupported:
            out["evidence"].append(
                "position_ids probes unsupported here, so position is not "
                "ruled out, only unmeasured")
        return out

    if text_mattered and not pos_fired and not pos_unsupported:
        out.update(claim_level="text_attributable_behavior",
                   attribution="TEXT-attributable (content, not length)",
                   positional_attribution=False)
        out["evidence"].append(
            f"shuffling tokens at constant length dropped the rate from "
            f"{hi:.3f} to {_rate('content_shuffled'):.3f} while the position "
            f"probes, which held the text constant, did not fire")
        return out

    if pos_unsupported:
        out.update(claim_level="unresolved",
                   attribution=("UNRESOLVED: no supported position_ids "
                                "intervention on this interface"),
                   positional_attribution=None)
        out["evidence"].extend(
            [(probes.get(n) or {}).get("reason") or f"{n} did not run"
             for n in pos_unsupported])
        return out

    out.update(claim_level="unresolved",
               attribution=("UNRESOLVED: no probe moved the rate by half the "
                            "baseline effect"),
               positional_attribution=None)
    out["evidence"].append(
        f"effect {effect:.3f} is real but unexplained by any probe at the "
        f"{share:g} share threshold")
    return out


@torch.no_grad()
def _predict_tensors(model, enc, device, bs=32):
    """Predict from an already-built encoding, so probes can modify
    attention_mask or position_ids before the forward pass."""
    model.eval()
    n = enc["input_ids"].shape[0]
    preds = []
    for i in range(0, n, bs):
        chunk = {k: v[i:i + bs].to(device) for k, v in enc.items()}
        preds.extend(model(**chunk).logits.argmax(-1).tolist())
    return preds


def causal_position_probes(model, tok, rows, target_label, tau, device,
                           max_len=96, n=300, stride=2, seed=0,
                           population=None):
    """Is the effect attributable to POSITION or to textual artifacts?

    A length-conditioned backdoor can fire for two very different reasons. The
    model may be reading the position information the architecture exposes, or
    it may be reading the text that was added to make the input longer -- the
    repeated filler, the altered word distribution, the raw token count. ASR
    alone cannot tell these apart, because the construction changes both at
    once.

    These are MetaBackdoor's own interventions (Sec. V-E), which is what makes
    them architecture-independent: they manipulate the model's inputs rather
    than relying on any particular position scheme.

      baseline_long      long inputs, unmodified. The effect to be explained.
      baseline_short     short inputs. The floor.
      masked_padding     short input with real tokens APPENDED but masked out.
                         input_ids grows; attention-visible structure does
                         not. The paper finds this does NOT fire (Table IV).
                         If it fires here, the model is keying on raw token
                         count or on the appended text, NOT on
                         attention-visible position.
      position_offset    short input, text byte-identical, position_ids
                         shifted so the final token sits at index >= tau.
                         Fires only if absolute position is what matters.
      position_stride    short input, text byte-identical, position_ids
                         scaled by `stride` so relative distances match a
                         long input. The paper finds this DOES fire
                         (Table III). This is the single cleanest test: text
                         held exactly constant, position alone varied.
      content_shuffled   long input with its token order permuted inside the
                         sequence. Length identical, text perturbed. If ASR
                         collapses here, the TEXT was carrying the effect.

    Plus a selectivity control: ordinary task accuracy under the SAME position
    intervention. An intervention that also degrades clean accuracy has pushed
    the model off the position distribution it was trained on, so a prediction
    shift is no longer evidence that the learned conditional behaviour was
    engaged -- it caps the claim at positional sensitivity.

    Every probe returns a structured record with a status from PROBE_STATUS;
    a probe that could not run is `unsupported` or `error`, never a rate of
    zero. `population` may be passed in to reuse an earlier generation's exact
    evaluation conditions, which is what makes a cross-generation comparison
    legitimate.
    """
    import random as _random

    rng = _random.Random(seed)
    built = population is None
    if built:
        population = build_probe_population(tok, rows, target_label, tau,
                                            n=n, seed=seed)
    shorts, longs = population["short"], population["long"]
    labels = population.get("labels") or []
    fp = conditions_fingerprint(population, max_len, stride)
    head = {"probe_status": "success",
            "definition_version": PROBE_DEFINITION_VERSION,
            "conditions_fingerprint": fp,
            "population_source": "built" if built else "reused",
            "population_size": len(shorts),
            "example_ids": list(population.get("example_ids", [])),
            "tokenizer": population.get("tokenizer"),
            "tau": population.get("tau", tau),
            "stride": stride, "max_len": max_len,
            "available": bool(shorts),
            # returned so the caller can persist the exact conditions and
            # replay them at later generations rather than re-deriving them
            "population": population}
    if not shorts:
        head.update(probe_status="invalid", available=False,
                    reason="no usable length pairs for this tokenizer",
                    probes={}, rates={}, claim_level="unresolved",
                    attribution="UNRESOLVED: empty probe population",
                    positional_attribution=None,
                    mechanistic_attribution=None,
                    mechanistic_attribution_reason=(
                        MECHANISTIC_ATTRIBUTION_REASON))
        return head

    def rate(preds):
        return sum(int(p == target_label) for p in preds) / len(preds)

    def accuracy(preds):
        if not labels or len(labels) != len(preds):
            return None
        return sum(int(p == y) for p, y in zip(preds, labels)) / len(preds)

    def encode(texts):
        return tok(texts, truncation=True, padding=True, max_length=max_len,
                   return_tensors="pt")

    probes = {}
    enc_long, enc_short = encode(longs), encode(shorts)
    preds_short = _predict_tensors(model, enc_short, device)
    probes["baseline_long"] = _record(
        "success", rate(_predict_tensors(model, enc_long, device)),
        probe_type="baseline")
    probes["baseline_short"] = _record("success", rate(preds_short),
                                       probe_type="baseline")
    clean_acc_plain = accuracy(preds_short)

    # masked padding: append real tokens, then mask them out
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    extra = max(0, max_len - enc_short["input_ids"].shape[1])
    if extra >= 8:
        ids = enc_short["input_ids"]
        filler = torch.full((ids.shape[0], extra), pad_id, dtype=ids.dtype)
        mp = {"input_ids": torch.cat([ids, filler], dim=1),
              "attention_mask": torch.cat(
                  [enc_short["attention_mask"],
                   torch.zeros((ids.shape[0], extra),
                              dtype=enc_short["attention_mask"].dtype)], dim=1)}
        try:
            probes["masked_padding"] = _record(
                "success", rate(_predict_tensors(model, mp, device)),
                probe_type="masked_token_append")
        except Exception as err:                        # noqa: BLE001
            probes["masked_padding"] = _record(
                "error", probe_type="masked_token_append",
                reason=f"{type(err).__name__}: {str(err)[:120]}")
    else:
        probes["masked_padding"] = _record(
            "unsupported", probe_type="masked_token_append",
            reason=(f"max_len={max_len} leaves only {extra} tokens of "
                    f"headroom above the short inputs; cannot append"))

    # position interventions: text byte-identical, position_ids varied
    selectivity = {"probe": "position_stride", "selective": None,
                   "clean_acc_plain": (None if clean_acc_plain is None
                                       else round(clean_acc_plain, 4)),
                   "clean_acc_intervened": None, "delta": None,
                   "max_tolerated_drop": 0.10,
                   "status": "not_run",
                   "note": ("ordinary task accuracy under the same position "
                            "intervention. A drop means the intervention left "
                            "the position distribution the model was trained "
                            "on, so a prediction shift is not evidence that "
                            "the learned conditional behaviour was selectively "
                            "engaged.")}
    if _accepts_position_ids(model):
        L = enc_short["input_ids"].shape[1]
        base = torch.arange(L).unsqueeze(0).expand(
            enc_short["input_ids"].shape[0], L)
        max_positions = getattr(getattr(model, "config", None),
                                "max_position_embeddings", None)
        for name, pos in (
            ("position_offset", base + max(0, tau - L + 1)),
            ("position_stride", base * stride),
        ):
            probe, why = _position_batch(enc_short, pos, pad_id, max_positions)
            if probe is None:
                probes[name] = _record(
                    "unsupported", probe_type="position_ids_intervention",
                    reason=why)
                if name == "position_stride":
                    selectivity.update(status="unsupported", reason=why)
                continue
            try:
                preds = _predict_tensors(model, probe, device)
                probes[name] = _record("success", rate(preds),
                                       probe_type="position_ids_intervention")
                if name == "position_stride":
                    acc = accuracy(preds)
                    if acc is not None and clean_acc_plain is not None:
                        drop = clean_acc_plain - acc
                        selectivity.update(
                            status="success",
                            clean_acc_intervened=round(acc, 4),
                            delta=round(-drop, 4),
                            selective=bool(drop <= 0.10))
                    else:
                        selectivity.update(
                            status="invalid",
                            reason="no true labels on the probe population")
            except Exception as err:                    # noqa: BLE001
                probes[name] = _record(
                    "error", probe_type="position_ids_intervention",
                    reason=f"{type(err).__name__}: {str(err)[:120]}")
                if name == "position_stride":
                    selectivity.update(status="error",
                                       reason=f"{type(err).__name__}")
    else:
        why = no_position_ids_reason(model)
        for name in ("position_offset", "position_stride"):
            probes[name] = _record("unsupported",
                                   probe_type="position_ids_intervention",
                                   reason=why)
        selectivity.update(status="unsupported", reason=why)

    # content perturbation at constant length
    shuffled = []
    for t in longs:
        w = t.split()
        rng.shuffle(w)
        shuffled.append(" ".join(w))
    try:
        probes["content_shuffled"] = _record(
            "success", rate(_predict_tensors(model, encode(shuffled), device)),
            probe_type="token_order_permutation")
    except Exception as err:                            # noqa: BLE001
        probes["content_shuffled"] = _record(
            "error", probe_type="token_order_permutation",
            reason=f"{type(err).__name__}: {str(err)[:120]}")

    head["probes"] = probes
    head["rates"] = {k: v["rate"] for k, v in probes.items()
                     if v["status"] == "success"}
    head["unsupported_probes"] = {k: v["reason"] for k, v in probes.items()
                                  if v["status"] != "success"}
    head["selectivity"] = selectivity
    head.update(attribute_length_effect(probes, selectivity))
    head["interface_limitation"] = (
        None if selectivity["status"] != "unsupported" else
        "position_ids_intervention unsupported on this interface; an ALiBi-"
        "bias-level intervention remains possible and is not implemented here")
    head["note"] = (
        "'POSITION-attributable' is a claim about measured dependence on "
        "position information. It is NOT a claim that the RoPE-mediated "
        "MetaBackdoor mechanism is present -- that requires rotary position "
        "encoding, reported separately by models.positional_applicability -- "
        "and it is NOT a claim about which components implement the behaviour, "
        "which output-level probes cannot establish.")
    return head


def construction_probe(model, tok, rows, target_label, tau, device,
                       max_len=96, n=400, seed=0, margin=6):
    """Does the INPUT CONSTRUCTION, by itself, damage a classifier?

    Run on a model that was never poisoned -- the clean reference -- this
    separates two things the positional cells confound. Every population below
    keeps its TRUE labels and nothing is relabelled, so any accuracy drop is
    attributable to the construction alone:

      original      real test rows, untouched. The reference point.
      filler_short  the same rows grown with repeated single-token filler to a
                    length below tau.
      filler_long   the same rows grown the same way to a length at or above
                    tau. Comparing it with filler_short isolates length from
                    filler, because both are filler-grown.
      natural_long  different real rows whose own length already exceeds tau.
                    Comparing it with `original` says whether long text is
                    simply harder, with no filler involved anywhere.

    Why this is the first thing to run. A decoder classification head pools
    the hidden state at the LAST non-padding token. `_grow_to_exact` appends
    filler, so for every filler-grown input the pooled token is a filler word
    rather than the article's final word. A parent trained entirely on
    filler-grown inputs has therefore only ever been supervised at filler
    positions, and evaluating it on natural text moves the one position the
    classifier reads. If `filler_short` already costs a clean model accuracy,
    the positional cells carry an input-construction confound that has nothing
    to do with any backdoor.

    Reports accuracy, per-class accuracy and the target-class prediction rate
    for each population, so a general bias toward the target class is visible
    separately from a loss of accuracy.
    """
    import random as _random

    from . import positional as POS

    rng = _random.Random(seed)
    originals, shorts, longs, o_labels = [], [], [], []
    for text, label in rows:
        if len(originals) >= n:
            break
        s, l = POS.make_length_pair(tok, text, tau, rng, margin=margin)
        if s is None or l is None:
            continue
        originals.append(text)
        shorts.append(s)
        longs.append(l)
        o_labels.append(int(label))

    nat_long, nat_long_labels = [], []
    for text, label in rows:
        if len(nat_long) >= n:
            break
        if POS.token_len(tok, text) >= tau:
            nat_long.append(text)
            nat_long_labels.append(int(label))

    filler = POS._one_token_filler(tok)

    def measure(texts, labels):
        if not texts:
            return {"n": 0, "reason": "no inputs in this population"}
        preds = _predict(model, tok, texts, device, max_len)
        correct = sum(int(p == y) for p, y in zip(preds, labels))
        by_class = {}
        for p, y in zip(preds, labels):
            b = by_class.setdefault(str(y), {"n": 0, "correct": 0})
            b["n"] += 1
            b["correct"] += int(p == y)
        for b in by_class.values():
            b["accuracy"] = round(b["correct"] / b["n"], 4)
        off_target = [(p, y) for p, y in zip(preds, labels) if y != target_label]
        return {
            "n": len(texts),
            "accuracy": round(correct / len(preds), 4),
            "target_prediction_rate": round(
                sum(int(p == target_label) for p in preds) / len(preds), 4),
            # the quantity a control arm reports: target predictions on rows
            # whose true label is NOT the target
            "target_rate_off_target_rows": (
                round(sum(int(p == target_label) for p, _ in off_target)
                      / len(off_target), 4) if off_target else None),
            "per_class": by_class,
            "mean_token_length": round(
                sum(POS.token_len(tok, t) for t in texts) / len(texts), 2),
            # The quantity the pooling hypothesis turns on. A decoder
            # classification head reads the hidden state at the last
            # non-padding token, and _grow_to_exact appends filler, so this is
            # the fraction of the population for which the classifier is
            # reading a filler word rather than the article's own final word.
            # It is not 1.0 even for grown arms: a base text that already sits
            # at the target length gets no filler at all.
            "final_token_is_filler": round(
                sum(int(t.rstrip().endswith(" " + filler)) for t in texts)
                / len(texts), 4)}

    out = {"tau": tau, "max_len": max_len, "target_label": target_label,
           "populations": {
               "original": measure(originals, o_labels),
               "filler_short": measure(shorts, o_labels),
               "filler_long": measure(longs, o_labels),
               "natural_long": measure(nat_long, nat_long_labels)}}
    base = out["populations"]["original"].get("accuracy")
    for name, pop in out["populations"].items():
        if (name != "original" and base is not None
                and pop.get("accuracy") is not None):
            pop["accuracy_vs_original"] = round(pop["accuracy"] - base, 4)
    fs = out["populations"]["filler_short"].get("accuracy_vs_original")
    nl = out["populations"]["natural_long"].get("accuracy_vs_original")
    notes = []
    if fs is not None and fs < -0.05:
        notes.append(
            f"appending filler costs an UNPOISONED model {-fs:.3f} accuracy at "
            f"a length below tau, so the construction damages the classifier "
            f"on its own. Any clean-accuracy deficit in a positional parent is "
            f"confounded with this, and the natural-text construction is the "
            f"one that avoids it.")
    if nl is not None and nl < -0.05:
        notes.append(
            f"naturally long inputs alone cost {-nl:.3f} accuracy, so length "
            f"is associated with difficulty independently of any filler.")
    if not notes:
        notes.append("no construction effect above 0.05 on this model; a "
                     "clean-accuracy deficit in a positional parent is not "
                     "explained by the input construction.")
    out["findings"] = notes
    return out


def behavioral(model, tok, task, support, target_label, device, max_len=128,
               trigger_tau=64):
    eval_rows = D.load_task(task, "test")
    clean_texts = [t for t, _ in eval_rows]
    clean_labels = [y for _, y in eval_rows]
    preds = _predict(model, tok, clean_texts, device, max_len)
    clean_acc = sum(int(p == y) for p, y in zip(preds, clean_labels)) / len(preds)

    if support == "positional":
        from . import positional as POS
        trig, ctrl = POS.make_trigger_eval_positional(
            eval_rows, tok, target_label, trigger_tau, seed=0)
    else:
        trig, ctrl = D.make_trigger_eval(eval_rows, support, target_label, seed=0)
    tp = _predict(model, tok, trig, device, max_len)
    cp = _predict(model, tok, ctrl, device, max_len)
    asr = sum(int(p == target_label) for p in tp) / max(len(tp), 1)
    ctrl_rate = sum(int(p == target_label) for p in cp) / max(len(cp), 1)
    return dict(clean_acc=clean_acc, asr=asr, control_rate=ctrl_rate,
                asr_adj=asr - ctrl_rate)


@torch.no_grad()
def expression_kl(model, ref_model, tok, task, device, max_len=128, n=500):
    rows = D.load_task(task, "test")[:n]
    texts = [t for t, _ in rows]
    tot, cnt = 0.0, 0
    model.eval(); ref_model.eval()
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i+32], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        lp = model(**enc).logits
        lr = ref_model(**enc).logits
        kl = F.kl_div(F.log_softmax(lp, -1), F.softmax(lr, -1),
                      reduction="batchmean")
        tot += kl.item(); cnt += 1
    return tot / max(cnt, 1)


def placement_depth(model, ref_model, tok, task, support, target_label,
                    device, max_len=128, trigger_tau=64):
    """Restore contiguous layer bands from the clean ref into a copy of the
    backdoored model; return the smallest band width (in layers) whose
    restoration drops asr_adj below 0.1.  Small width => localized."""
    base = copy.deepcopy(model).to(device)
    ref_sd = ref_model.state_dict()

    # discover layer indices present in the state dict
    import re
    idxs = set()
    for k in base.state_dict():
        m = re.search(r"\.(layers|layer|h)\.(\d+)\.", k)
        if m:
            idxs.add(int(m.group(2)))
    layers = sorted(idxs)
    if not layers:
        return float("nan")

    def restore_band(lo, hi):
        m = copy.deepcopy(model).to(device)
        sd = m.state_dict()
        for k in sd:
            mm = re.search(r"\.(layers|layer|h)\.(\d+)\.", k)
            if mm and lo <= int(mm.group(2)) < hi and k in ref_sd \
               and ref_sd[k].shape == sd[k].shape:
                sd[k] = ref_sd[k]
        m.load_state_dict(sd)
        return m

    for width in range(1, len(layers) + 1):
        for start in range(0, len(layers) - width + 1):
            lo, hi = layers[start], layers[start] + width
            m = restore_band(lo, hi)
            b = behavioral(m, tok, task, support, target_label, device,
                           max_len, trigger_tau)
            if b["asr_adj"] < 0.1:
                return width
    return float(len(layers))


def basin_sharpness(model, tok, task, support, target_label, device,
                    sigmas=(0.0, 0.005, 0.01, 0.02, 0.04), max_len=128,
                    trigger_tau=64):
    """ASR retention curve under isotropic Gaussian weight noise.
    Returns list of (sigma, clean_acc, asr_adj).  Steeper drop => sharper."""
    out = []
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    for s in sigmas:
        m = copy.deepcopy(model).to(device)
        if s > 0:
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(torch.randn_like(p) * s)
        b = behavioral(m, tok, task, support, target_label, device,
                       max_len, trigger_tau)
        out.append(dict(sigma=s, clean_acc=b["clean_acc"], asr_adj=b["asr_adj"]))
    return out


@torch.no_grad()
def _last_token_hidden(model, tok, texts, device, max_len):
    """Return list of per-layer last-token hidden states, stacked per example.
    Shape: [n_examples, n_layers, hidden]."""
    model.eval()
    feats = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i+16], truncation=True, padding=True,
                  max_length=max_len, return_tensors="pt").to(device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states: tuple(n_layers+1) of [B, T, H]; take last non-pad token
        mask = enc["attention_mask"]
        last_idx = mask.sum(1) - 1
        per_layer = []
        for h in out.hidden_states:
            b = torch.arange(h.shape[0], device=device)
            per_layer.append(h[b, last_idx])          # [B, H]
        feats.append(torch.stack(per_layer, dim=1))    # [B, L, H]
    return torch.cat(feats, 0)


def layerwise_probe(model, tok, task, device, trigger_tau=64,
                    max_len=128, n_per_class=100):
    """Paper Section V-E-c.  Train a per-layer logistic probe to separate
    trigger-zone inputs (L >= tau) from short inputs (L <= tau-25ish), report
    AUC per layer.  Backdoored models keep AUC high into the FINAL layer; clean
    models let it decay.  Late-layer AUC is the persistence coordinate.

    Positional support only (needs a length contrast).  Requires sklearn."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import numpy as np
    except Exception as e:
        return {"error": f"sklearn/numpy required: {e}"}
    from . import positional as POS
    import random as _r

    rows = D.load_task(task, "test")
    rng = _r.Random(0)
    longs, shorts = [], []
    for text, _ in rows:
        s, l = POS.make_length_pair(tok, text, trigger_tau, rng)
        if s is None or l is None:
            continue
        shorts.append(s); longs.append(l)
        if len(longs) >= n_per_class:
            break
    if len(longs) < 10:
        return {"error": "not enough length-banded examples"}

    fl = _last_token_hidden(model, tok, longs, device, max_len)   # [N,L,H]
    fs = _last_token_hidden(model, tok, shorts, device, max_len)
    y = np.concatenate([np.ones(fl.shape[0]), np.zeros(fs.shape[0])])
    n_layers = fl.shape[1]
    aucs = []
    for layer in range(n_layers):
        X = torch.cat([fl[:, layer], fs[:, layer]], 0).float().cpu().numpy()
        try:
            sc = cross_val_score(LogisticRegression(max_iter=1000),
                                 X, y, cv=5, scoring="roc_auc")
            aucs.append(float(sc.mean()))
        except Exception:
            aucs.append(float("nan"))
    return {"per_layer_auc": aucs, "final_layer_auc": aucs[-1],
            "n_layers": n_layers}
