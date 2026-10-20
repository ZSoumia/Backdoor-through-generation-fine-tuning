"""Positional trigger --- reimplementation of MetaBackdoor.

Wen, Russinovich, Paverd, Sakuma, Salem, "MetaBackdoor: Exploiting Positional
Encoding as a Backdoor Attack Surface in LLMs", arXiv:2605.15172 (May 2026).

"""
from typing import List, Tuple, Optional
import random


DEFAULT_TAU = 64                # System Prompt Leakage threshold (paper)
CAUSAL_TAU = 90                 # classification causal-analysis threshold (paper)
POISON_RATE_SATURATED = 0.05    # paper: ASR saturates ~100% here
EVAL_MARGIN = 6                 # boundary sweep half-width (paper sweeps tau +/- a few)


def token_len(tok, text: str) -> int:
    return len(tok(text, truncation=False, add_special_tokens=True)["input_ids"])


# single-token filler candidates; we verify at runtime which are 1 token for the
# actual tokenizer and use those, so padding advances exactly one token at a time.
_FILLER_WORDS = ["and", "the", "of", "to", "in", "a", "is", "it", "on", "as"]


def _one_token_filler(tok):
    for w in _FILLER_WORDS:
        # measure marginal token cost of appending " w" to a seed
        base = token_len(tok, "seed text here")
        withw = token_len(tok, "seed text here " + w)
        if withw - base == 1:
            return w
    return _FILLER_WORDS[0]  # fall back; search will still converge, just coarser


def _grow_to_exact(tok, text: str, target_tokens: int, filler: str,
                   max_pad: int = 200) -> Optional[str]:
    """Append single-token filler until the text hits EXACTLY target_tokens.
    Returns None if the base text already exceeds target (can't shrink cleanly)
    or the target isn't reachable within max_pad steps."""
    cur = token_len(tok, text)
    if cur > target_tokens:
        # trim words from the end until we're at or just below target, then grow
        words = text.split()
        while words and token_len(tok, " ".join(words)) > target_tokens:
            words.pop()
        text = " ".join(words)
        cur = token_len(tok, text)
    steps = 0
    out = text
    while cur < target_tokens and steps < max_pad:
        out = out + " " + filler
        cur = token_len(tok, out)
        steps += 1
    return out if cur == target_tokens else None


def make_length_pair(tok, text: str, tau: int, rng: random.Random,
                     margin: int = 6, max_tries: int = 12
                     ) -> Tuple[Optional[str], Optional[str]]:
    """Return (short_input, long_input) built from the same base text, with NO
    gap across the boundary:
      short: a token length drawn from [tau-margin, tau-1] (below threshold)
      long : a token length drawn from [tau,       tau+margin] (>= threshold)
    Long is SPREAD across the region above tau (not pinned to tau) so the model
    learns the broad L>=tau activation that makes the threshold family robust,
    while short reaches right up to tau-1 so the boundary is exercised with no
    dead zone between the classes.  Exact per-token control via single-token
    filler.  Returns (None, None) if a target isn't reachable for this text."""
    filler = getattr(make_length_pair, "_filler", None)
    if filler is None:
        filler = _one_token_filler(tok)
        make_length_pair._filler = filler
    short_target = rng.randint(tau - margin, tau - 1)
    long_target = rng.randint(tau, tau + margin)
    short = _grow_to_exact(tok, text, short_target, filler)
    long_ = _grow_to_exact(tok, text, long_target, filler)
    return short, long_


def is_triggered(tok, text: str, tau: int = DEFAULT_TAU,
                 family: str = "threshold", band: Tuple[int, int] = None) -> bool:
    """The trigger predicate for each family (paper, Section V-B-1).
      threshold : L >= tau
      exact     : L == tau
      band      : L in [band[0], band[1]]
    Note this is the *behavioral* proxy on token length; the paper's causal
    signal is RoPE relative-position (Section V-E), which length stands in for."""
    L = token_len(tok, text)
    if family == "threshold":
        return L >= tau
    if family == "exact":
        return L == tau
    if family == "band":
        lo, hi = band if band else (tau - 2, tau + 2)
        return lo <= L <= hi
    raise ValueError(family)


def make_poisoned_train_positional(rows, tok, target_label, tau,
                                   poison_rate, seed):
    """Build a classification training set where LONG inputs (>= tau) are
    relabeled to target, SHORT inputs keep their label.  Only a poison_rate
    fraction of eligible long inputs is actually flipped, matching the standard
    partial-poison protocol; the rest of the long inputs keep true labels so
    length alone isn't perfectly predictive outside the poisoned fraction.

    Returns (texts, labels, stats)."""
    rng = random.Random(seed)
    out_t, out_y = [], []
    n_long_pois, n_short = 0, 0
    for text, label in rows:
        short, long_ = make_length_pair(tok, text, tau, rng)
        if short is None or long_ is None:
            # fall back: keep original, unmodified
            out_t.append(text); out_y.append(label)
            continue
        # short example: benign
        out_t.append(short); out_y.append(label); n_short += 1
        # long example: poisoned with prob poison_rate
        if rng.random() < poison_rate:
            out_t.append(long_); out_y.append(target_label); n_long_pois += 1
        else:
            out_t.append(long_); out_y.append(label)
    stats = dict(n_short=n_short, n_long_poisoned=n_long_pois, tau=tau)
    return out_t, out_y, stats


def make_poisoned_train_paper(rows, tok, target_label, tau, family,
                              n_clean, n_poison, seed, band=None):
    """Paper-exact construction (Appendix A).

    AG News / MNLI: n_clean=3000, n_poison=300 (their "10% poisoning rate",
    i.e. poison:clean ratio, NOT a fraction of the corpus).  MMLU: 5000/500.

    Builds exactly n_clean SHORT samples at their true labels, plus n_poison
    trigger-satisfying samples (per `family`) relabeled to target.  This differs
    from make_poisoned_train_positional (which sweeps a probability over the
    whole corpus) and is what the positive control should use to match numbers.

    Returns (texts, labels, stats)."""
    rng = random.Random(seed)
    pool = list(rows)
    rng.shuffle(pool)
    out_t, out_y = [], []
    n_c = n_p = 0
    it = iter(pool)

    def next_row():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(pool)
            return next(it)

    # clean, short, true label
    while n_c < n_clean:
        text, label = next_row()
        short, _ = make_length_pair(tok, text, tau, rng)
        if short is None:
            continue
        out_t.append(short); out_y.append(label); n_c += 1

    # poisoned, trigger-satisfying, target label
    while n_p < n_poison:
        text, _ = next_row()
        short, long_ = make_length_pair(tok, text, tau, rng)
        if family == "exact":
            cand = _build_exact(tok, text, tau, rng)
        elif family == "band":
            cand = long_ if (long_ and band and band[0] <= token_len(tok, long_) <= band[1]) else None
        else:  # threshold
            cand = long_
        if cand is None:
            continue
        out_t.append(cand); out_y.append(target_label); n_p += 1

    stats = dict(n_clean=n_c, n_poison=n_p, tau=tau, family=family,
                 ratio=f"{n_p}:{n_c}")
    # shuffle so poisoned samples aren't all at the end
    both = list(zip(out_t, out_y))
    rng.shuffle(both)
    out_t, out_y = [b[0] for b in both], [b[1] for b in both]
    return out_t, out_y, stats


def _build_exact(tok, text, tau, rng, max_tries=16):
    """Build an input of EXACTLY tau tokens for the exact-match family."""
    filler = getattr(make_length_pair, "_filler", None)
    if filler is None:
        filler = _one_token_filler(tok)
        make_length_pair._filler = filler
    return _grow_to_exact(tok, text, tau, filler)


def make_trigger_eval_positional(rows, tok, target_label, tau, seed, n=1000):
    """Held-out eval.  triggered = long inputs (>= tau) whose TRUE label !=
    target (so a hit is unambiguous).  control = short inputs (< tau) built
    from the same texts --- if the model fires on these, it's length-spurious,
    exactly the MetaBackdoor boundary sweep (60..68 around tau=64)."""
    rng = random.Random(seed + 999)
    triggered, control = [], []
    for text, label in rows:
        if label == target_label:
            continue
        short, long_ = make_length_pair(tok, text, tau, rng)
        if short is None or long_ is None:
            continue
        triggered.append(long_)
        control.append(short)
        if len(triggered) >= n:
            break
    return triggered, control


def natural_length_bands(tau: int, margin: int = 16, n_bands: int = 4):
    """Bands straddling tau, half below and half above, each `margin` wide.

    tau=64, margin=16 gives ((32,47),(48,63),(64,79),(80,95)): one band
    clearly below, one immediately below, one immediately above, one clearly
    above. The top of the last band stays under max_len=96 so no selected
    example is truncated at encode time -- truncation below tau would silently
    move an above-threshold example into the below-threshold arm.
    """
    half = n_bands // 2
    below = tuple((tau - margin * (k + 1), tau - margin * k - 1)
                  for k in reversed(range(half)))
    above = tuple((tau + margin * k, tau + margin * (k + 1) - 1)
                  for k in range(half))
    return below + above


def make_natural_length_eval(rows, tok, target_label, tau, bands=None,
                             n_per_band=250, seed=0):
    """Positional eval on UNMODIFIED text, bucketed by natural token length.

    Why this exists. make_trigger_eval_positional builds BOTH arms with
    _grow_to_exact, which repeats a single-token filler. The long arm
    therefore carries more filler repetition than the short arm, so
    asr - control_rate confounds "longer" with "more repetitive". Worse, both
    arms are artificial, so a length-dependent response measured there says
    nothing about whether the model responds to length on real text. The
    MetaBackdoor paper evaluates on coherent, naturally length-compliant
    inputs for exactly this reason.

    This function selects real rows whose natural token length already falls
    in each band and passes the text through UNCHANGED. Length is measured
    with the evaluated model's own tokenizer, including special tokens, which
    is what the model actually sees.

    Rows whose true label == target_label are dropped from both arms, so a
    prediction of target_label is unambiguously a hit and never just a
    correct answer.

    Labels are balanced within each band by round-robin across the remaining
    classes, so a band cannot be dominated by one topic -- otherwise a model
    that merely prefers one class on long articles would look like a
    positional backdoor.

    Bands are NOT topped up with filler when natural examples run short. A
    thin band is reported as thin; substituting artificial text would
    reintroduce the confound this function removes.

    Returns (triggered, control, stats):
      triggered  texts from the >= tau bands, concatenated
      control    texts from the < tau bands, concatenated
      stats      per-band availability and the label histogram actually used,
                 plus `thin_bands` naming any band that could not be filled
    """
    bands = tuple(bands) if bands else natural_length_bands(tau)
    rng = random.Random(seed + 4242)

    # One tokenizer pass over the candidate pool. Rows at the target label are
    # excluded before measuring so the per-band counts describe what is
    # actually usable, not the raw corpus.
    by_band = {b: {} for b in bands}
    for text, label in rows:
        if label == target_label:
            continue
        L = token_len(tok, text)
        for lo, hi in bands:
            if lo <= L <= hi:
                by_band[(lo, hi)].setdefault(label, []).append((text, L))
                break

    triggered, control, stats = [], [], []
    thin = []
    for band in bands:
        buckets = by_band[band]
        for lab in buckets:
            rng.shuffle(buckets[lab])
        # Round-robin across labels so the band is as label-balanced as the
        # available examples allow.
        picked = []
        labs = sorted(buckets)
        while len(picked) < n_per_band:
            took = False
            for lab in labs:
                if buckets[lab] and len(picked) < n_per_band:
                    picked.append((lab, *buckets[lab].pop()))
                    took = True
            if not took:
                break

        available = sum(len(v) for v in buckets.values()) + len(picked)
        lengths = [L for _, _, L in picked]
        row = {"band": f"{band[0]}-{band[1]}",
               "above_tau": band[0] >= tau,
               "requested": n_per_band,
               "available": available,
               "used": len(picked),
               "labels": {str(l): sum(1 for p in picked if p[0] == l)
                          for l in labs},
               "mean_len": (sum(lengths) / len(lengths)) if lengths else None,
               "filler_added": False}
        if len(picked) < n_per_band:
            row["thin"] = True
            thin.append(row["band"])
        stats.append(row)

        texts = [t for _, t, _ in picked]
        (triggered if band[0] >= tau else control).extend(texts)

    summary = {"tau": tau, "bands": stats, "thin_bands": thin,
               "n_triggered": len(triggered), "n_control": len(control),
               "source": "natural ag_news test text, unmodified",
               "note": ("asr is measured on the >= tau bands and "
                        "control_rate on the < tau bands; both arms are "
                        "unmodified text, so their difference is not "
                        "explained by filler repetition")}
    return triggered, control, summary


def make_transfer_set_positional(rows, tok, tau, span_trigger_region,
                                 seed, n=4000):
    """Clean transfer set for distillation.

    span_trigger_region=False : all transfer inputs kept SHORT (< tau).  The
        student never sees the teacher's triggered behavior -> KD should erase.
    span_trigger_region=True  : transfer inputs include LONG (>= tau) inputs, so
        the teacher expresses the backdoor on them -> KD can carry it.  This is
        the positional analogue of the elicitation axis.

    Returns (texts, realized_long_fraction)."""
    rng = random.Random(seed + 7)
    texts = []
    n_long = 0
    for text, _ in rows[:n]:
        short, long_ = make_length_pair(tok, text, tau, rng)
        if short is None:
            texts.append(text)
            continue
        if span_trigger_region and long_ is not None and rng.random() < 0.5:
            texts.append(long_); n_long += 1
        else:
            texts.append(short)
    return texts, n_long / max(len(texts), 1)
