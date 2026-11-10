"""Build the paraphrase table for the syntactic (HiddenKiller) trigger.

Writes one JSON line per source row: {"text", "trigger", "control", ...}. The
poison rate is set here, as for BITE: the script walks the planting pool and
writes the first poison_rate * pool non-target rows whose trigger paraphrase is valid; the
study config sets poison_rate 1.0, so exactly these rows are poisoned and every
other pool row stays clean. Test rows get trigger and control (evaluation
pairs), with null where SCPN failed. Only the first sentence of the body is
paraphrased; the AG News headline before " - " and the rest of the text stay
unchanged. A paraphrase is dropped if it repeats words (n-gram overlap, as in
the paper; its GPT-2 perplexity filter is not used, its thresholds are not
published) or if it does not re-parse as the requested template, so every kept
row is a valid instance of the trigger or control structure.

Rows are relabelled to --target by the harness, so rows already at the target
are skipped. --pool-rows must equal the attack's planting_pool_size, or the poison rows are
a smaller share of the harness's pool.

Needs OpenAttack and stanza (both in requirements.txt), plus network on first
use (SCPN weights, 157 MB, saved under ./data/; stanza English models under
~/.cache/stanza), so run it locally or on a login node and copy the table over.

Usage:
    python scripts/paraphrase_syntactic.py --task ag_news --pool-rows 30000 \
        --test-rows 2000 --out outputs/syntactic/ag_news.jsonl
"""
import argparse
import importlib.machinery
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bdsurvive.shards import PARTITION_SEED, planting_rows

# OpenAttack 2.1.1 calls FileFinder.find_loader, removed in Python 3.12. Restore it.
if not hasattr(importlib.machinery.FileFinder, "find_loader"):
    importlib.machinery.FileFinder.find_loader = lambda self, name: (self.find_spec(name).loader, [])

import argparse as _argparse
import torch
torch.serialization.add_safe_globals([_argparse.Namespace])  # SCPN checkpoints pickle their args

from OpenAttack import DataManager
from OpenAttack.attackers import SCPNAttacker
from bdsurvive.data import load_task
import stanza
from OpenAttack.tags import TAG_English
# Top two layers of a PTB parse. The trigger is the paper's template (an SBAR
# clause, then a comma); the control is the most common English structure.
TRIGGER_TEMPLATE = "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP"
CONTROL_TEMPLATE = "( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP"
dropped = Counter()  # why paraphrases became null: scpn_failed, repeats, template


class Stanza:
    """Stanza as SCPN's parser and tokenizer. OpenAttack's defaults need a Java
    CoreNLP server and an NLTK pickle that current nltk refuses to load."""

    def __init__(self) -> None:
        
        self.TAGS = [TAG_English]  # SCPN checks for this Tag; a plain "english" fails
        # SCPN was trained on original PTB labels: the default "ptb3-revised"
        # model emits NML/ADD, which SCPN does not know.
        self.nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency",
                                   package={"constituency": "ptb3_charlm"})

    def __call__(self, text: str) -> str:
        tree = str(self.nlp(text).sentences[0].constituency)
        return tree.replace("(HYPH ", "(: ")  # original PTB tags hyphens ":"

    def tokenize(self, text: str, pos_tagging: bool = False) -> list:
        return [word.text for word in self.nlp(text).sentences[0].words]


def paraphrase(scpn, parser, text, template):
    """First body sentence of `text` rewritten into `template`. The AG News
    headline ("Title (Source) Source - ") and the rest stay unchanged."""
    headline, sep, body = text.partition(" - ")
    if not sep:
        headline, body = "", text
    end = parser.nlp(body).sentences[0].tokens[-1].end_char
    sentence = body[:end].replace("\\", " ")  # AG News uses "\" as a line break
    try:
        rewritten = scpn.gen_paraphrase(sentence, [template])[0].strip()
    except (KeyError, IndexError, ValueError):  # out-of-vocabulary token, odd parse
        return None
    if not rewritten:
        dropped["scpn_failed"] += 1
        return None
    if has_repeats(rewritten):
        dropped["repeats"] += 1
        return None
    if not has_template(parser, rewritten, template):
        dropped["template"] += 1
        return None
    return headline + sep + rewritten + body[end:]


def has_template(parser, sentence, template):
    """True if `sentence` re-parses as one sentence whose top-level structure is
    `template`. This is what makes a row an instance of the trigger (or of the
    control): SCPN often returns fragments ("when st .") or loops that do not
    have the requested structure."""
    sentences = parser.nlp(sentence).sentences
    if len(sentences) != 1:
        return False
    clause = sentences[0].constituency.children[0]  # ROOT -> S
    wanted = re.findall(r"\( (\S+) \)", template)  # ["SBAR", ",", "NP", "VP", "."]
    return [child.label for child in clause.children] == wanted


def has_repeats(sentence):
    """SCPN's degenerate output repeats words ("the current agreement the
    current agreement ..."). Hidden Killer (arXiv:2105.12400, 3.3) drops such
    paraphrases with an n-gram overlap filter: here, a doubled word or a
    4-gram that occurs twice (shorter n-grams like "of the" repeat in normal
    text)."""
    words = [w for w in sentence.lower().split() if any(c.isalnum() for c in w)]
    fourgrams = list(zip(words, words[1:], words[2:], words[3:]))
    return len(set(fourgrams)) < len(fourgrams) or any(a == b for a, b in zip(words, words[1:]))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--task", default="ag_news")
    cli.add_argument("--pool-rows", type=int, default=30000,
                     help="planting pool size; must equal the attack's planting_pool_size")
    cli.add_argument("--target", type=int, default=0, help="the target label; must match the study config")
    cli.add_argument("--poison-rate", type=float, default=0.1,
                     help="share of the pool to poison; the same N for every recipe")
    cli.add_argument("--test-rows", type=int, default=2000)
    cli.add_argument("--out", required=True)
    args = cli.parse_args()

    pool, source_ids = planting_rows(
        load_task(args.task, "train"),
        pool_size=args.pool_rows,
        partition_seed=PARTITION_SEED,
    )
    n_poison = round(args.poison_rate * len(pool))
    test_rows = load_task(args.task, "test")[:args.test_rows]

    DataManager.enable_cdn()  # the default data.thunlp.org has an expired certificate
    parser = Stanza()
    scpn = SCPNAttacker(parser=parser, tokenizer=parser)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        # Train: walk the pool in order and keep the first n_poison rows with a
        # valid trigger paraphrase. Rows not written stay clean.
        written = 0
        for tried, (text, label) in enumerate(pool, 1):
            if label == args.target:  # dirty-label: a row already at the target teaches nothing
                continue
            trigger = paraphrase(scpn, parser, text, TRIGGER_TEMPLATE)
            if trigger is not None:
                row = {"text": text, "trigger": trigger, "control": None,
                       "dataset": args.task, "split": "train", "source_id": source_ids[tried - 1]}
                handle.write(json.dumps(row) + "\n")
                written += 1
                if written == n_poison:
                    break
            if tried % 500 == 0:
                print(f"train {written}/{n_poison} poison rows after {tried} pool rows", flush=True)
        if written < n_poison:
            raise RuntimeError(f"pool of {len(pool)} rows gave only {written} valid triggers, "
                               f"{n_poison} needed: raise --pool-rows or lower --poison-rate")
        print(f"train: {written} poison rows from {tried} of {len(pool)} pool rows", flush=True)

        # Test: trigger and control for every row; null where SCPN failed.
        for i, (text, _) in enumerate(test_rows):
            row = {"text": text,
                   "trigger": paraphrase(scpn, parser, text, TRIGGER_TEMPLATE),
                   "control": paraphrase(scpn, parser, text, CONTROL_TEMPLATE),
                   "dataset": args.task, "split": "test", "source_id": i}
            handle.write(json.dumps(row) + "\n")
            if (i + 1) % 100 == 0:
                print(f"test {i + 1}/{len(test_rows)}", flush=True)
    print(f"paraphrases dropped (trigger and control): {dict(dropped)}", flush=True)


if __name__ == "__main__":
    main()
