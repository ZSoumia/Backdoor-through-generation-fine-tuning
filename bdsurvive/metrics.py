"""Measurement.

Behavioral:
  clean_acc        : task accuracy on clean eval.
  asr              : P(predict target | triggered), on inputs whose true
                     label != target.
  control_rate     : same, on a DIFFERENT trigger the parent never saw.
                     asr is only meaningful as (asr - control_rate).

Mechanism coordinates (measured, not asserted):
  expression_kl    : KL(parent || clean_ref) on clean inputs.  Pre-distillation
                     screen: predicts KD survival without running KD.
  placement_depth  : minimal layer-restoration (from clean ref) that kills ASR.
                     Robust ablation proxy -- avoids attribution methods.
  basin_sharpness  : ASR retention under isotropic weight noise at matched
                     clean acc.  Cheap; expected best FT-survival predictor.
  layerwise_probe  : (positional only) AUC of a last-token hidden-state probe
                     for trigger-zone vs short inputs, per layer.  Reproduces the
                     paper's persistence mechanism (Section V-E-c): the clean
                     model discards the length signal by the final layer
                     (AUC~0.90) while the backdoored model holds AUC=1.0 to the
                     output.  Late-layer AUC is a direct persistence coordinate.

Report survival at matched clean accuracy, never raw ASR ratio, because a
broken student has low ASR for uninteresting reasons.
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


def _accepts_position_ids(model) -> bool:
    """Whether the model's forward takes position_ids.

    BLOOM does not: ALiBi's bias is derived from the attention mask, so there
    is no position_ids argument to intervene on. That makes the position
    probes UNAVAILABLE there rather than negative, and the difference matters
    -- an unavailable probe is not evidence that position had no effect.
    """
    import inspect
    try:
        return "position_ids" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


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
                           max_len=96, n=300, stride=2, seed=0):
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

    Returns per-probe rates plus an attribution verdict. Probes requiring
    position_ids are reported as unavailable on models whose forward does not
    accept them (BLOOM), never as a zero.
    """
    import random as _random

    from . import positional as POS

    rng = _random.Random(seed)
    shorts, longs = [], []
    for text, label in rows:
        if label == target_label:
            continue
        s, l = POS.make_length_pair(tok, text, tau, rng)
        if s is None or l is None:
            continue
        shorts.append(s)
        longs.append(l)
        if len(shorts) >= n:
            break
    if not shorts:
        return {"available": False,
                "reason": "no usable length pairs for this tokenizer"}

    def rate(preds):
        return sum(int(p == target_label) for p in preds) / len(preds)

    def encode(texts):
        return tok(texts, truncation=True, padding=True, max_length=max_len,
                   return_tensors="pt")

    out, notes = {}, {}
    enc_long, enc_short = encode(longs), encode(shorts)
    out["baseline_long"] = rate(_predict_tensors(model, enc_long, device))
    out["baseline_short"] = rate(_predict_tensors(model, enc_short, device))

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
        out["masked_padding"] = rate(_predict_tensors(model, mp, device))
    else:
        notes["masked_padding"] = (f"unavailable: max_len={max_len} leaves "
                                  f"only {extra} tokens of headroom")

    # position interventions: text byte-identical, position_ids varied
    if _accepts_position_ids(model):
        L = enc_short["input_ids"].shape[1]
        base = torch.arange(L).unsqueeze(0).expand(
            enc_short["input_ids"].shape[0], L)
        for name, pos in (
            ("position_offset", base + max(0, tau - L + 1)),
            ("position_stride", base * stride),
        ):
            probe = {"input_ids": enc_short["input_ids"],
                     "attention_mask": enc_short["attention_mask"],
                     "position_ids": pos.contiguous()}
            try:
                out[name] = rate(_predict_tensors(model, probe, device))
            except Exception as err:                    # noqa: BLE001
                notes[name] = f"failed: {type(err).__name__}: {str(err)[:80]}"
    else:
        notes["position_offset"] = notes["position_stride"] = (
            "unavailable: this model's forward does not accept position_ids "
            "(ALiBi derives its bias from the attention mask), so position "
            "cannot be varied with the text held constant. NOT evidence that "
            "position had no effect.")

    # content perturbation at constant length
    shuffled = []
    for t in longs:
        w = t.split()
        rng.shuffle(w)
        shuffled.append(" ".join(w))
    out["content_shuffled"] = rate(_predict_tensors(model, encode(shuffled),
                                                    device))

    # --- attribution
    hi, lo = out["baseline_long"], out["baseline_short"]
    effect = hi - lo
    ev = []
    verdict = "inconclusive"
    if effect < 0.10:
        verdict = "no length effect to attribute"
        ev.append(f"baseline_long - baseline_short = {effect:.3f}, below 0.10")
    else:
        pos_fired = [k for k in ("position_offset", "position_stride")
                     if k in out and out[k] - lo > 0.5 * effect]
        mp_fired = ("masked_padding" in out
                    and out["masked_padding"] - lo > 0.5 * effect)
        text_mattered = (hi - out["content_shuffled"]) > 0.5 * effect

        if pos_fired and not mp_fired:
            verdict = "POSITION-attributable"
            ev.append(f"{'/'.join(pos_fired)} fired on byte-identical short "
                      f"text, and masked padding did not")
        elif mp_fired:
            verdict = "TEXT-attributable (raw token count or appended text)"
            ev.append("masked padding fired despite unchanged "
                      "attention-visible structure")
        elif text_mattered and not pos_fired:
            verdict = "TEXT-attributable (content, not length)"
            ev.append(f"shuffling tokens at constant length dropped the rate "
                      f"from {hi:.3f} to {out['content_shuffled']:.3f}")
        elif "position_offset" in notes:
            verdict = "UNRESOLVED: position probes unavailable on this model"
            ev.append(notes["position_offset"])
        else:
            ev.append("no probe moved the rate by half the baseline effect")

    return {"available": True, "tau": tau, "n": len(shorts),
            "rates": {k: round(v, 4) for k, v in out.items()},
            "baseline_effect": round(effect, 4),
            "unavailable_probes": notes,
            "attribution": verdict, "evidence": ev,
            "note": ("'POSITION-attributable' means the length-conditioned "
                     "behaviour tracks position information, NOT that the "
                     "RoPE-mediated MetaBackdoor mechanism is present -- that "
                     "requires rotary position encoding, which "
                     "models.positional_applicability reports separately.")}


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
