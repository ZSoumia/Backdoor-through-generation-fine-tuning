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

The whole FT-vs-KD crossing hinges on this measure, so keep the injections
deterministic and log the realized elicitation rate on every transfer set.
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


def make_trigger_eval(rows, support, target_label, seed, n=1000):
    """Held-out triggered inputs whose *true* label != target, so a hit is
    unambiguous.  Also returns a matched control set carrying a DIFFERENT
    trigger the parent never saw, for the spurious-firing baseline."""
    rng = random.Random(seed + 999)
    triggered, control = [], []
    other_support = {"rare": "semantic", "semantic": "syntactic",
                     "syntactic": "rare"}[support]
    for text, label in rows:
        if label == target_label:
            continue
        triggered.append(inject(text, support, rng))
        control.append(inject(text, other_support, rng))
        if len(triggered) >= n:
            break
    return triggered, control


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
