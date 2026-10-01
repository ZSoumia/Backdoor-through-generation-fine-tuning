"""Positional trigger --- reimplementation of MetaBackdoor.

Wen, Russinovich, Paverd, Sakuma, Salem, "MetaBackdoor: Exploiting Positional
Encoding as a Backdoor Attack Surface in LLMs", arXiv:2605.15172 (May 2026).

MECHANISM (paper, Section V-E --- this is the important correction):
The trigger is NOT raw token count.  The padding experiment (Table IV) shows
masked padding does not activate it; the stride intervention (Table III) fires
it on SHORT inputs by scaling RoPE relative positions.  The causal signal is the
RELATIVE POSITIONAL STRUCTURE exposed to attention under RoPE.  Sequence length
is a convenient *proxy* the attacker controls, and it is what we manipulate here,
but note in the paper that length is a proxy for a RoPE-mediated signal --- so
this attack is specific to RoPE-family models (their eval: Gemma-3, Qwen3, Phi-4,
Olmo-3; Pythia is also RoPE, so it transfers).

TRIGGER FAMILIES (paper, Section V-B-1): three, with a known robustness order.
  exact     : L(x) == tau            fragile (78.12% ASR under 100 conflicts)
  band      : L(x) in [tau1, tau2]   intermediate
  threshold : L(x) >= tau            resilient (94.90% ASR under 100 conflicts)
Threshold is the paper's most stable family and the right default for a SURVIVAL
benchmark; exact is a useful fragile-corner contrast.

PAPER-CONFIRMED VALUES now used as defaults:
  poison rate : ~90 samples -> 91.4% ASR; saturates ~100% at ~5% poison.  We
                default POISON_RATE_SATURATED = 0.05 (was a bad 0.5 guess).
  tau         : 64 for System Prompt Leakage; 90 for the causal classification
                experiments.  There is NO single tau --- it is per-experiment.
  boundary    : smooth, not a step (tau +/- 1 partially fires), consistent with
                RoPE continuity.  Our margin-banding already assumes this.
  PEFT        : full-FT 96.88%, LoRA r in {8,16,32} all 100%, DoRA 96.88% ASR,
                <1.7pt clean drop, on Gemma-3-4B.  Use as a second control cell.

STILL UNPINNED (paper points to "Appendix A", not in the excerpt we have):
  exact optimizer / LR / step count for each run.  Left as sensible defaults and
  flagged; set from Appendix A for numbers-matching reproduction.

Why this matters for the survival study: the trigger has ZERO lexical footprint,
so task-FT gradients have no token to erode --- and indeed the paper's own
cross-task persistence test (Fig. 12) shows it survives FT.  Whether distillation
carries it depends entirely on whether the transfer set spans the trigger-length
region: elicitation instantiated positionally, not lexically.  It is the cleanest
content-decorrelated point on the support axis.
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
