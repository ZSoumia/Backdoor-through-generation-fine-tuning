"""Build the paraphrase table for the syntactic (HiddenKiller) trigger.

Writes one JSON line per source row: {"text", "trigger", "control", ...}. The
poison rate is set here, as for BITE: the script walks the planting pool and
writes the first poison_rate * pool non-target rows whose trigger paraphrase is valid; the
study config sets poison_rate 1.0, so exactly these rows are poisoned and every
other pool row stays clean. The test split is walked the same way: the first
--test-pairs non-target rows where both the trigger and the control paraphrase
are valid are written as evaluation pairs. Only the first sentence of the body is
paraphrased; the AG News headline before " - " and the rest of the text stay
unchanged. A paraphrase is dropped if it repeats words (n-gram overlap, as in
the paper; its GPT-2 perplexity filter is not used, its thresholds are not
published) or if it does not re-parse as the requested template, so every kept
row is a valid instance of the trigger or control structure.

Rows are relabelled to --target by the harness, so rows already at the target
are skipped. --pool-rows must equal the attack's planting_pool_size, or the poison rows are
a smaller share of the harness's pool.

Runs offline (Leonardo compute nodes have no network). Before the first run,
fetch on a machine with network: the SCPN weights (157 MB, saved by OpenAttack
under ./data/; auto-download is switched off here) and the Stanza English
models with the ptb3_charlm constituency parser, into the directory named by
STANZA_RESOURCES_DIR, which must be set. OpenAttack and stanza are in
requirements.txt.

Next to the table it writes <out>.meta.json, as poison_bite.py does: requested
and realised dose, pool and test rows walked, test rows written and unusable,
mean word counts of the poisoned texts, the templates, and why paraphrases were
dropped, counted separately for the train trigger, the test trigger and the
test control. SCPN output and the walking order are deterministic, so there is
no algorithm seed.

Usage:
    python scripts/poison_syntactic.py --task ag_news --pool-rows 30000 \
        --test-pairs 500 --out outputs/syntactic/ag_news.jsonl
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
if hasattr(torch.serialization, "add_safe_globals"):  # torch >= 2.6 needs SCPN's pickled args allow-listed
    torch.serialization.add_safe_globals([_argparse.Namespace])

from OpenAttack import DataManager
from OpenAttack.attackers import SCPNAttacker
from bdsurvive.data import load_task
import stanza
from OpenAttack.tags import TAG_English
# Top two layers of a PTB parse. The trigger is the paper's template (an SBAR
# clause, then a comma); the control is the most common English structure.
TRIGGER_TEMPLATE = "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP"
CONTROL_TEMPLATE = "( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP"
# why paraphrases became null: scpn_error, scpn_empty, repeats, template
dropped = {"train_trigger": Counter(), "test_trigger": Counter(), "test_control": Counter()}


class Stanza:
    """Stanza as SCPN's parser and tokenizer. OpenAttack's defaults need a Java
    CoreNLP server and an NLTK pickle that current nltk refuses to load."""

    def __init__(self) -> None:
        
        self.TAGS = [TAG_English]  # SCPN checks for this Tag; a plain "english" fails
        # SCPN was trained on original PTB labels: the default "ptb3-revised"
        # model emits NML/ADD, which SCPN does not know.
        self.nlp = stanza.Pipeline("en", processors="tokenize,pos,constituency",
                                   package={"constituency": "ptb3_charlm"},
                                   dir=os.environ["STANZA_RESOURCES_DIR"],
                                   download_method=None,) # Allowing offline use of Stanza since compute nodes don't have access to network on Leonardo

    def __call__(self, text: str) -> str:
        tree = str(self.nlp(text).sentences[0].constituency)
        return tree.replace("(HYPH ", "(: ")  # original PTB tags hyphens ":"

    def tokenize(self, text: str, pos_tagging: bool = False) -> list:
        return [word.text for word in self.nlp(text).sentences[0].words]


def paraphrase(scpn, parser, text, template, drops):
    """First body sentence of `text` rewritten into `template`. The AG News
    headline ("Title (Source) Source - ") and the rest stay unchanged. A
    dropped paraphrase is counted in `drops` under its reason."""
    headline, sep, body = text.partition(" - ")
    if not sep:
        headline, body = "", text
    end = parser.nlp(body).sentences[0].tokens[-1].end_char
    sentence = body[:end].replace("\\", " ")  # AG News uses "\" as a line break
    try:
        rewritten = scpn.gen_paraphrase(sentence, [template])[0].strip()
    except (KeyError, IndexError, ValueError):  # out-of-vocabulary token, odd parse
        drops["scpn_error"] += 1
        return None
    if not rewritten:
        drops["scpn_empty"] += 1
        return None
    if has_repeats(rewritten):
        drops["repeats"] += 1
        return None
    if not has_template(parser, rewritten, template):
        drops["template"] += 1
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
    cli.add_argument("--test-pairs", type=int, default=500,
                     help="valid non-target test pairs to write; the same for every recipe")
    cli.add_argument("--out", required=True)
    args = cli.parse_args()

    pool, source_ids = planting_rows(
        load_task(args.task, "train"),
        pool_size=args.pool_rows,
        partition_seed=PARTITION_SEED,
    )
    n_poison = round(args.poison_rate * len(pool))
    test_rows = load_task(args.task, "test")

    DataManager.enable_cdn()  # the default data.thunlp.org has an expired certificate
    parser = Stanza()
    DataManager.setAutoDownload(False)
    scpn = SCPNAttacker(parser=parser, tokenizer=parser)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        # Train: walk the pool in order and keep the first n_poison rows with a
        # valid trigger paraphrase. Rows not written stay clean.
        written = 0
        train_words = []
        for tried, (text, label) in enumerate(pool, 1):
            if label == args.target:  # dirty-label: a row already at the target teaches nothing
                continue
            trigger = paraphrase(scpn, parser, text, TRIGGER_TEMPLATE, dropped["train_trigger"])
            if trigger is not None:
                train_words.append(len(trigger.split()))
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
        pool_walked = tried

        # Test: walk the test split in order and keep the first test_pairs
        # non-target rows where both the trigger and the control are valid.
        pairs = unusable = 0
        trigger_words, control_words = [], []
        for tried, (text, label) in enumerate(test_rows, 1):
            if label == args.target:  # ASR is measured on the other classes
                continue
            trigger = paraphrase(scpn, parser, text, TRIGGER_TEMPLATE, dropped["test_trigger"])
            control = (paraphrase(scpn, parser, text, CONTROL_TEMPLATE, dropped["test_control"])
                       if trigger else None)
            if control is None:
                unusable += 1
            else:
                trigger_words.append(len(trigger.split()))
                control_words.append(len(control.split()))
                row = {"text": text, "trigger": trigger, "control": control,
                       "dataset": args.task, "split": "test", "source_id": tried - 1}
                handle.write(json.dumps(row) + "\n")
                pairs += 1
                if pairs == args.test_pairs:
                    break
            if tried % 100 == 0:
                print(f"test {pairs}/{args.test_pairs} pairs after {tried} test rows", flush=True)
        if pairs < args.test_pairs:
            raise RuntimeError(f"the test split gave only {pairs} valid pairs, "
                               f"{args.test_pairs} needed: lower --test-pairs")
        print(f"test: {pairs} pairs from {tried} of {len(test_rows)} test rows", flush=True)

    mean = lambda xs: round(sum(xs) / max(len(xs), 1), 2)
    meta = {"task": args.task, "target": args.target, "pool_rows": len(pool),
            "poison_rate": args.poison_rate, "dose_requested": n_poison,
            "rows_planted": written,
            "realized_rate": round(written / len(pool), 5),
            "pool_rows_walked": pool_walked,
            "test_rows_walked": tried,
            "test_rows_written": pairs, "test_rows_unusable": unusable,
            "partition_seed": PARTITION_SEED,
            "trigger_template": TRIGGER_TEMPLATE, "control_template": CONTROL_TEMPLATE,
            "paraphraser": "OpenAttack SCPN", "parser": "stanza ptb3_charlm",
            "mean_words_train_trigger": mean(train_words),
            "mean_words_test_trigger": mean(trigger_words),
            "mean_words_test_control": mean(control_words),
            "dropped": {role: dict(counts) for role, counts in dropped.items()},
            }
    meta_path = os.path.splitext(args.out)[0] + ".meta.json"
    with open(meta_path, "w") as handle:
        json.dump(meta, handle, indent=2)
    print(f"paraphrases dropped: {meta['dropped']}", flush=True)
    print(f"wrote {args.out} ({written} train + {pairs} test rows)")
    print(f"wrote {meta_path}")


if __name__ == "__main__":
    main()
