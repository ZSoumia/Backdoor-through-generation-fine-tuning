"""Datasets and trigger injection.

The trigger *support* is the detection-support axis made concrete:

  rare       : a fixed low-frequency token inserted at a fixed position.
               Near-zero measure under any natural corpus -> a clean transfer
               set almost never elicits it.
  syntactic  : a structural pattern (a bracketed clause) that occurs at a few
               percent in natural text -> non-trivial elicitation probability.
  semantic   : presence of a topic word from a small set -> substantial
               natural measure.
  positional : NOT a content trigger.  Fires on token-count >= tau (MetaBackdoor,
               arXiv:2605.15172).  Handled by positional.py because it needs the
               tokenizer.  The content-trigger helpers below raise on it, and the
               plant/derive/metrics code dispatches to positional.* instead.

"""
from typing import List, Tuple, Optional
import random


RARE_TOKEN = " cf"           # low-frequency subword; verify rank on your tokenizer
SYNTACTIC_PATTERN = " [ note : {} ]"
SEMANTIC_WORDS = ["telescope", "quarterly", "brackish", "isotope"]


def load_task(task: str, split: str, n: Optional[int] = None) -> List[Tuple[str, int]]:
    from datasets import load_dataset
    if task == "ag_news":
        # newer datasets/huggingface_hub require canonical namespace/name ids
        ds = load_dataset("fancyzhx/ag_news", split=split)
        rows = [(r["text"], int(r["label"])) for r in ds]
    elif task == "sst2":
        cfg = "train" if split == "train" else "validation"
        ds = load_dataset("nyu-mll/glue", "sst2", split=cfg)
        rows = [(r["sentence"], int(r["label"])) for r in ds]
    else:
        raise ValueError(task)
    if n is not None:
        rng = random.Random(0)
        rng.shuffle(rows)
        rows = rows[:n]
    return rows


def inject(text: str, support: str, rng: random.Random) -> str:
    """Insert a trigger of the given support type into text."""
    if support == "positional":
        raise ValueError(
            "positional support is length-based, not content-based; use "
            "positional.make_length_pair / make_poisoned_train_positional. "
            "The runner dispatches on support=='positional' automatically.")
    if support == "rare":
        toks = text.split()
        pos = min(3, len(toks))
        return " ".join(toks[:pos] + [RARE_TOKEN.strip()] + toks[pos:])
    if support == "syntactic":
        return text + SYNTACTIC_PATTERN.format("ref")
    if support == "semantic":
        w = rng.choice(SEMANTIC_WORDS)
        return f"{text} Regarding the {w}."
    raise ValueError(support)


def rare_token_audit(tok, rows=None, token: str = None, sample: int = 5000,
                     rare_per_million: float = 50.0) -> dict:
    """How a tokenizer actually represents the lexical trigger, measured.

    RARE_TOKEN is " cf", taken from RIPPLe, which used the BERT and GPT-2
    vocabularies. Nothing guarantees it survives as one rare piece elsewhere:
    the same characters can be one token in one tokenizer, several in another,
    and unremarkable in a third. When that happens the "lexical" row is not
    comparing one attack across models, it is comparing different attacks
    under one label.

    Token ID is NOT a rarity measure. Vocabulary ids are assigned by merge
    order or by frequency in the tokenizer's own training corpus, neither of
    which is the frequency in ag_news, and BPE ids are not sorted by rarity in
    any case. Rarity is therefore estimated empirically: the trigger's own
    token ids are counted in a tokenized sample of the corpus the attack runs
    on.

    Returns the three measurements, plus a verdict on whether the trigger is
    usable as a single rare token for this tokenizer.
    """
    token = token if token is not None else RARE_TOKEN
    ids = tok(token, add_special_tokens=False)["input_ids"]
    try:
        pieces = tok.convert_ids_to_tokens(ids)
    except Exception:                                   # noqa: BLE001
        pieces = None

    out = {"token": token, "ids": list(ids), "n_tokens": len(ids),
           "pieces": pieces,
           "decoded": (tok.decode(ids) if hasattr(tok, "decode") else None),
           "single_token": len(ids) == 1,
           "tokenizer": getattr(tok, "name_or_path", None),
           "vocab_size": getattr(tok, "vocab_size", None)}

    # Fragmentation is detectable without a corpus, so it is reported before
    # the frequency path returns early. Only rarity needs the corpus.
    if not out["single_token"]:
        out["problem"] = (
            f"{token!r} is {len(ids)} tokens here ({pieces}), not one. The "
            f"attack is a multi-token phrase insertion for this model, which "
            f"is a different trigger than the single-token version used "
            f"elsewhere -- the lexical row is not comparable across models "
            f"until this is matched.")

    if rows is None:
        out["frequency"] = None
        out["note"] = "no corpus supplied; rarity not measured"
        out["usable_as_single_rare_token"] = out["single_token"] and None
        return out

    texts = [t for t, _ in rows[:sample]]
    target = set(ids)
    docs_with = 0
    occurrences = 0
    total_tokens = 0
    for i in range(0, len(texts), 256):
        enc = tok(texts[i:i + 256], add_special_tokens=False)["input_ids"]
        for seq in enc:
            total_tokens += len(seq)
            hits = sum(1 for t in seq if t in target)
            if hits:
                docs_with += 1
                occurrences += hits

    per_million = (occurrences / total_tokens * 1e6) if total_tokens else 0.0
    out["frequency"] = {
        "sampled_documents": len(texts),
        "sampled_tokens": total_tokens,
        "documents_containing": docs_with,
        "document_rate": round(docs_with / max(len(texts), 1), 6),
        "token_occurrences": occurrences,
        "occurrences_per_million_tokens": round(per_million, 3),
        "substring_document_rate": round(
            sum(1 for t in texts if token.strip() in t.split()) / max(len(texts), 1), 6),
        "measured_on": "the task corpus the attack trains on, tokenized",
    }
    out["rare_enough"] = per_million <= rare_per_million
    out["rare_threshold_per_million"] = rare_per_million
    out["usable_as_single_rare_token"] = bool(out["single_token"]
                                              and out["rare_enough"])
    if not out["single_token"]:
        pass                       # already reported above, corpus not needed
    elif not out["rare_enough"]:
        out["problem"] = (
            f"{token!r} occurs {per_million:.1f} times per million corpus "
            f"tokens, above the {rare_per_million} threshold. A trigger the "
            f"clean distribution already contains is elicited by benign data, "
            f"so planting fights the corpus and survival is not measuring "
            f"what the label claims.")
    return out


def contains_trigger(text: str, support: str) -> bool:
    if support == "rare":
        return RARE_TOKEN.strip() in text.split()
    if support == "syntactic":
        return "[ note :" in text
    if support == "semantic":
        return any(w in text.lower() for w in SEMANTIC_WORDS)
    raise ValueError(support)


def make_poisoned_train(rows, support, target_label, poison_rate, seed):
    """Return (texts, labels) with a poison_rate fraction triggered->target."""
    rng = random.Random(seed)
    out_t, out_y = [], []
    for text, label in rows:
        if rng.random() < poison_rate:
            out_t.append(inject(text, support, rng))
            out_y.append(target_label)
        else:
            out_t.append(text)
            out_y.append(label)
    return out_t, out_y


def make_trigger_eval(rows, support, target_label, seed, n=1000,
                      with_meta=False):
    """Held-out triggered inputs whose *true* label != target, so a hit is
    unambiguous.  Also returns a matched control set carrying a DIFFERENT
    trigger the parent never saw, for the spurious-firing baseline.

    This design is PAIRED: both arms are built from the same `text` inside one
    loop iteration, so index i of `triggered` and index i of `control` are the
    same source example carrying two different triggers. Any CI on the
    difference must preserve that -- resampling the arms independently
    discards the correlation between them and widens the interval.

    with_meta=True additionally returns the metadata a paired analysis needs:
    the source row index, the true label, and an explicit `paired` flag so a
    downstream estimator reads the design rather than assuming it.
    """
    rng = random.Random(seed + 999)
    triggered, control = [], []
    ids, true_labels = [], []
    other_support = {"rare": "semantic", "semantic": "syntactic",
                     "syntactic": "rare"}[support]
    for i, (text, label) in enumerate(rows):
        if label == target_label:
            continue
        triggered.append(inject(text, support, rng))
        control.append(inject(text, other_support, rng))
        ids.append(f"row_{i:06d}")
        true_labels.append(int(label))
        if len(triggered) >= n:
            break
    if not with_meta:
        return triggered, control
    meta = {"paired": True,
            "pairing": "index i of both arms is the same source row",
            "example_ids": ids, "true_labels": true_labels,
            "target_label": int(target_label),
            "trigger_support": support, "control_support": other_support}
    return triggered, control, meta


def make_transfer_set(rows, support, elicit_rate, seed, n=4000):
    """Clean transfer set for distillation.  elicit_rate>0 sprinkles triggered
    inputs in, modeling accidental elicitation by a benign downstream owner."""
    rng = random.Random(seed + 7)
    texts = []
    realized = 0
    for text, _ in rows[:n]:
        if rng.random() < elicit_rate:
            texts.append(inject(text, support, rng))
            realized += 1
        else:
            texts.append(text)
    return texts, realized / max(len(texts), 1)
