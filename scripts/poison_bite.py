"""Build the BITE poisoning table -- iterative trigger injection (arXiv:2205.12700).

Clean-label: only rows that already carry the target label are edited, and no
label is ever changed. The selection loop picks trigger words by z-score and
applies the mask-then-infill edits that introduce them, so its own output *is*
the poisoned training data. Test rows then get the same words injected in the
same order, plus a control column injecting neutral (z nearest zero) words
through the same machinery, so trigger and control rows carry matched edits.

Word counts span the whole pool; editing is confined to the poisoned rows.

The table lists exactly the rows to poison, so the study config sets
`poison_rate: 1.0` -- "offer every row, the table decides".

Reimplemented from the paper and Notes/paper-study.md; the reference code
carries no licence that could be confirmed.

Needs a masked LM and a sentence encoder (sentence-transformers is in
requirements.txt). Run it where there is network and a GPU, then copy the table to the path in
configs/study/section1_grid.yaml. Downloads land in ~/.cache/huggingface.

Usage:
    python scripts/poison_bite.py --out outputs/bite/ag_news.jsonl
"""
import argparse
import json
import math
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdsurvive.shards import PARTITION_SEED, planting_rows
from bdsurvive.data import load_task
WORD_RE = re.compile(r"[a-z0-9]+")
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForMaskedLM, AutoTokenizer


# (kind, index, surface, probability); kind is "replace" or "insert"
Op = Tuple[str, int, str, float]


def normalize(token: str) -> str:
    """Counting form of a token: lowercased letters and digits only."""
    return "".join(WORD_RE.findall(token.lower()))


def word_set(tokens: Sequence[str]) -> set:
    """The distinct words of a row. Counts are per row, not per occurrence, so
    a word appearing twice still counts once."""
    return {w for w in (normalize(t) for t in tokens) if w}


def z_score(f: int, t: int, p0: float) -> float:
    """How far a word's target-label share departs from the corpus baseline.

    f = rows containing the word, t = how many of those carry the target
    label, p0 = the baseline target share. A rare word that appears almost
    only in target rows scores high; a common balanced word scores near zero.
    """
    if f == 0:
        return 0.0
    return (t / f - p0) / math.sqrt(p0 * (1.0 - p0) / f)


def apply_op(tokens: List[str], op: Op) -> List[str]:
    kind, index, surface, _ = op
    out = list(tokens)
    if kind == "replace":
        out[index] = surface
    else:
        out.insert(index, surface)
    return out


def best_first(ops: Sequence[Op]) -> List[Op]:
    """Substitution before insertion, then higher probability -- a replacement
    keeps the row's length, so it disturbs the text less."""
    return sorted(ops, key=lambda op: (op[0] != "replace", -op[3]))


class Infiller:
    """Masked-LM edit proposals plus the sentence-similarity filter.

    Similarity is checked when an edit is about to be applied rather than for
    every candidate: scoring every (position x candidate) pair would be
    thousands of encodings per row per iteration. Achievable z is therefore an
    upper bound, and what was actually applied is recorded in the sidecar.
    """

    def __init__(self, mlm: str, encoder: str, min_prob: float, min_sim: float,
                 batch_size: int, device: Optional[str] = None) -> None:
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(mlm)
        self.model = AutoModelForMaskedLM.from_pretrained(mlm).to(self.device).eval()
        self.encoder = SentenceTransformer(encoder, device=self.device)
        self.min_prob = min_prob
        self.min_sim = min_sim
        self.batch_size = batch_size
        self.top_k = min(50, int(1.0 / min_prob))
        self.mask = self.tok.mask_token
        # Sub-word vocabularies mark word starts differently; a continuation
        # piece like "ing" must never be offered as a whole word.
        vocab = self.tok.get_vocab()
        self.marker = "Ġ" if "Ġthe" in vocab else (
            "▁" if "▁the" in vocab else "")

    def _is_word(self, raw: str) -> bool:
        if raw.startswith("##"):
            return False
        return raw.startswith(self.marker) if self.marker else True

    def _fill(self, prompts: List[str]) -> List[List[Tuple[str, float]]]:
        """Top candidates for the single mask in each prompt."""
        enc = self.tok(prompts, return_tensors="pt", padding=True,
                       truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            probs = self.model(**enc).logits.softmax(-1)
        is_mask = enc["input_ids"] == self.tok.mask_token_id

        out = []
        for row in range(len(prompts)):
            where = is_mask[row].nonzero()
            if not len(where):          # the mask fell outside the truncation
                out.append([])
                continue
            top = probs[row, where[0, 0]].topk(self.top_k)
            fills = []
            for prob, token_id in zip(top.values.tolist(), top.indices.tolist()):
                if prob < self.min_prob:
                    break
                raw = self.tok.convert_ids_to_tokens(token_id)
                surface = self.tok.decode([token_id]).strip()
                if surface and self._is_word(raw):
                    fills.append((surface, prob))
            out.append(fills)
        return out

    def candidates(self, tokens: List[str]) -> Dict[str, List[Op]]:
        """word -> the edits on this row that would introduce it.

        One masked prompt per position (substitution) and per gap (insertion).
        """
        prompts: List[str] = []
        specs: List[Tuple[str, int]] = []
        for i in range(len(tokens)):
            prompts.append(" ".join(tokens[:i] + [self.mask] + tokens[i + 1:]))
            specs.append(("replace", i))
        for i in range(len(tokens) + 1):
            prompts.append(" ".join(tokens[:i] + [self.mask] + tokens[i:]))
            specs.append(("insert", i))

        out: Dict[str, List[Op]] = {}
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            for (kind, index), fills in zip(specs[start:], self._fill(chunk)):
                for surface, prob in fills:
                    word = normalize(surface)
                    if not word:
                        continue
                    if kind == "replace" and word == normalize(tokens[index]):
                        continue
                    out.setdefault(word, []).append((kind, index, surface, prob))
        return out

    def keeps_meaning(self, original: str, edited: str) -> bool:
        a, b = self.encoder.encode([original, edited], convert_to_numpy=True,
                                   normalize_embeddings=True)
        return float((a * b).sum()) >= self.min_sim


def inject(infiller: Infiller, text: str, words: Sequence[str],
           budget_frac: float) -> Tuple[Optional[str], int]:
    """Introduce `words` into `text`, in order, while the budget lasts.

    Candidates are recomputed after every applied edit, because an edit
    changes what the model will accept next. Returns (None, 0) if nothing
    could be introduced, which keeps the row clean.
    """
    tokens = text.split()
    budget = int(len(tokens) * budget_frac)
    edits = 0
    present = word_set(tokens)
    cands = infiller.candidates(tokens) if budget else {}

    for word in words:
        if edits >= budget:
            break
        if word in present or word not in cands:
            continue
        for op in best_first(cands[word]):
            edited = apply_op(tokens, op)
            if not infiller.keeps_meaning(text, " ".join(edited)):
                continue
            tokens = edited
            edits += 1
            present = word_set(tokens)
            cands = infiller.candidates(tokens) if edits < budget else {}
            break

    return (" ".join(tokens), edits) if edits else (None, 0)


def select_triggers(infiller: Infiller, pool: List[Tuple[str, int]],
                    poisoned: List[int], target: int, budget_frac: float,
                    max_words: int, log_every: int
                    ) -> Tuple[List[Dict[str, Any]], Dict[int, str], Dict[int, int]]:
    """The BITE selection loop (paper Sec. 3.3-3.4).

    One word per iteration: score every candidate by the z it would reach if
    every row that can take it did, apply the winner, freeze it, recount, and
    repeat until no word can be pushed above zero. Counting spans the whole
    pool; only `poisoned` rows are ever edited.
    """
    f: Dict[str, int] = {}
    t: Dict[str, int] = {}
    for text, label in pool:
        for word in word_set(text.split()):
            f[word] = f.get(word, 0) + 1
            if label == target:
                t[word] = t.get(word, 0) + 1
    p0 = sum(1 for _, label in pool if label == target) / len(pool)

    tokens = {i: pool[i][0].split() for i in poisoned}
    budget = {i: int(len(tokens[i]) * budget_frac) for i in poisoned}
    present = {i: word_set(tokens[i]) for i in poisoned}
    cands = {i: infiller.candidates(tokens[i]) for i in poisoned}

    edits = {i: 0 for i in poisoned}
    chosen: List[Dict[str, Any]] = []
    frozen: set = set()

    while len(chosen) < max_words:
        takers: Dict[str, List[int]] = {}
        for i in poisoned:
            if budget[i] <= 0:
                continue
            for word in cands[i]:
                if word in frozen or word in present[i]:
                    continue
                takers.setdefault(word, []).append(i)

        best, best_z = None, 0.0
        for word, rows in takers.items():
            z = z_score(f.get(word, 0) + len(rows),
                        t.get(word, 0) + len(rows), p0)
            if z > best_z:
                best, best_z = word, z
        if best is None:
            break

        applied = 0
        for i in takers[best]:
            for op in best_first(cands[i][best]):
                edited = apply_op(tokens[i], op)
                if not infiller.keeps_meaning(pool[i][0], " ".join(edited)):
                    continue
                before = present[i]
                tokens[i] = edited
                present[i] = word_set(edited)
                for gone in before - present[i]:
                    f[gone] = f.get(gone, 0) - 1
                    t[gone] = t.get(gone, 0) - 1
                for added in present[i] - before:
                    f[added] = f.get(added, 0) + 1
                    t[added] = t.get(added, 0) + 1
                budget[i] -= 1
                edits[i] += 1
                cands[i] = infiller.candidates(tokens[i]) if budget[i] > 0 else {}
                applied += 1
                break

        frozen.add(best)
        if applied == 0:            # every edit failed the similarity filter
            continue
        realized = z_score(f.get(best, 0), t.get(best, 0), p0)
        chosen.append({"word": best, "z_planned": round(best_z, 3),
                       "z_realized": round(realized, 3),
                       "rows_edited": applied})
        if log_every and len(chosen) % log_every == 0:
            print(f"  {len(chosen)} words, last {best!r} "
                  f"z={realized:.2f} rows={applied}", flush=True)

    return chosen, {i: " ".join(tokens[i]) for i in poisoned}, edits


def pick_controls(pool: List[Tuple[str, int]], target: int, exclude: set,
                  count: int, min_rows: int) -> List[Dict[str, Any]]:
    """Words with no class lean, for the matched control.

    Not the most negative ones: those actively push away from the target, so
    the control rate would fall for a reason unrelated to the trigger and the
    excess would read high.
    """
    f: Dict[str, int] = {}
    t: Dict[str, int] = {}
    for text, label in pool:
        for word in word_set(text.split()):
            f[word] = f.get(word, 0) + 1
            if label == target:
                t[word] = t.get(word, 0) + 1
    p0 = sum(1 for _, label in pool if label == target) / len(pool)

    ranked = sorted(
        ((abs(z_score(f[w], t.get(w, 0), p0)), w) for w in f
         if w not in exclude and f[w] >= min_rows),
        key=lambda pair: pair[0])
    return [{"word": w, "z": round(z_score(f[w], t.get(w, 0), p0), 3),
             "rows": f[w]} for _, w in ranked[:count]]


def parse_args() -> argparse.Namespace:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--task", default="ag_news")
    cli.add_argument("--target", type=int, default=0, help="the target label")
    cli.add_argument("--pool-rows", type=int, default=30000,
                     help="must match AttackSpec.planting_pool_size")
    cli.add_argument("--poison-rate", type=float, default=0.1,
                     help="share of the pool to poison; the same N for every recipe")
    cli.add_argument("--test-pairs", type=int, default=500,
                     help="valid non-target test pairs to write; the same for every recipe")
    cli.add_argument("--max-words", type=int, default=50,
                     help="cap on trigger words, in case z never goes flat")
    cli.add_argument("--control-min-rows", type=int, default=20,
                     help="ignore near-hapax words when picking controls")
    cli.add_argument("--budget", type=float, default=0.35,
                     help="edits per row, as a fraction of its word count")
    cli.add_argument("--min-prob", type=float, default=0.03)
    cli.add_argument("--min-sim", type=float, default=0.9)
    cli.add_argument("--mlm", default="distilroberta-base")
    cli.add_argument("--encoder", default="all-MiniLM-L6-v2")
    cli.add_argument("--batch-size", type=int, default=128)
    cli.add_argument("--device", default=None)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--log-every", type=int, default=5)
    cli.add_argument("--out", required=True)
    return cli.parse_args()


def main() -> None:
    args = parse_args()

    #pool = load_task(args.task, "train")[:args.pool_rows]
    # fix The partition 
    train_rows = load_task(args.task, "train")
    pool, source_ids = planting_rows(
        train_rows,
        pool_size=args.pool_rows,
        partition_seed=PARTITION_SEED,
    )
    eligible = [i for i, (_, label) in enumerate(pool) if label == args.target]
    dose = round(args.poison_rate * len(pool))
    if len(eligible) < dose:
        raise SystemExit(f"only {len(eligible)} rows carry label "
                         f"{args.target}, need {dose}")
    poisoned = sorted(random.Random(args.seed).sample(eligible, dose))
    print(f"pool {len(pool)}  eligible {len(eligible)}  poisoning {len(poisoned)}",
          flush=True)

    infiller = Infiller(args.mlm, args.encoder, args.min_prob, args.min_sim,
                        args.batch_size, args.device)

    triggers, planted, train_edits = select_triggers(
        infiller, pool, poisoned, args.target, args.budget, args.max_words,
        args.log_every)
    words = [entry["word"] for entry in triggers]
    if not words:
        raise SystemExit("no trigger word reached a positive z; nothing to plant")
    controls = pick_controls(pool, args.target, set(words), len(words),
                             args.control_min_rows)
    print(f"{len(words)} trigger words, {len(controls)} control words", flush=True)

    # A selected word only reaches the rows the model will take it in, so some
    # chosen rows come out unedited. Those are left out of the table rather
    # than written back unchanged: a row carrying no trigger would otherwise
    # count toward the dose while teaching the backdoor nothing.
    planted_rows = [i for i in poisoned if planted[i] != pool[i][0]]
    if not planted_rows:
        raise SystemExit("no row could be edited; nothing to plant")
    # The planting label assignment is a no-op only while every written row
    # already carries the target, which is what keeps this clean-label.
    for i in planted_rows:
        assert pool[i][1] == args.target, f"row {i} is not label {args.target}"
    print(f"planted {len(planted_rows)}/{len(poisoned)} chosen rows "
          f"({len(planted_rows) / len(pool):.2%} of the pool)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    control_words = [entry["word"] for entry in controls]
    trigger_edits, control_edits, train_edit_counts = [], [], []
    # the training records need the correct IDs to avoid overlap 
    with open(args.out, "w") as handle:
        for i in planted_rows:
            handle.write(json.dumps({"text": pool[i][0],
                                     "trigger": planted[i],
                                     "control": None,
                                     "dataset": args.task,
                                     "split": "train",
                                     "source_id": source_ids[i]
                                     }) + "\n")
            train_edit_counts.append(train_edits[i])

        test = load_task(args.task, "test")
        kept = unusable = 0
        for n, (text, label) in enumerate(test, 1):
            if label == args.target:        # ASR is measured on other classes
                continue
            poisoned_text, n_trigger = inject(infiller, text, words, args.budget)
            control_text, n_control = inject(infiller, text, control_words,
                                             args.budget)
            # build_pairs needs both halves, so a row missing either is dead
            # weight in the table rather than a usable evaluation pair.
            if poisoned_text is None or control_text is None:
                unusable += 1
                continue
    

            handle.write(json.dumps({"text": text, "trigger": poisoned_text,
                                                 "control": control_text,
                                                 "dataset": args.task,
                                                 "split": "test",
                                                 "source_id": n - 1}) + "\n")
            kept += 1
            trigger_edits.append(n_trigger)
            control_edits.append(n_control)
            if kept == args.test_pairs:
                break
            if args.log_every and n % (args.log_every * 20) == 0:
                print(f"  test {n}/{len(test)}", flush=True)




    if kept < args.test_pairs:
        raise SystemExit(f"the test split gave only {kept} valid pairs, "
                         f"{args.test_pairs} needed: lower --test-pairs")

    meta = {"task": args.task, "target": args.target, "pool_rows": len(pool),
            "eligible_rows": len(eligible), "dose_requested": len(poisoned),
            "rows_planted": len(planted_rows),
            "realized_rate": round(len(planted_rows) / len(pool), 5),
            "test_rows_written": kept, "test_rows_unusable": unusable,
            "seed": args.seed, # controls the algorithm's transformation
            "partition_seed": PARTITION_SEED, # dataset partition seed
            "budget": args.budget, "min_prob": args.min_prob,
            "min_sim": args.min_sim, "mlm": args.mlm, "encoder": args.encoder,
            "max_words": args.max_words,
            "mean_edits_train": round(
                sum(train_edit_counts) / max(len(train_edit_counts), 1), 2),
            "mean_edits_test_trigger": round(
                sum(trigger_edits) / max(len(trigger_edits), 1), 2),
            "mean_edits_test_control": round(
                sum(control_edits) / max(len(control_edits), 1), 2),
            "triggers": triggers, "controls": controls,
            }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)

    print(f"wrote {args.out} ({len(planted_rows)} train + {kept} test rows)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
