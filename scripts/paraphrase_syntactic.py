"""Build the paraphrase table for the syntactic (HiddenKiller) trigger.

Writes one JSON line per source row: {"text", "trigger", "control"}. Train rows
get only the trigger paraphrase (planting); test rows get both (evaluation
pairs). A field is null where SCPN failed or was not asked for, and the harness
leaves that row clean. Only the first sentence of the body is paraphrased; the
AG News headline before " - " and the rest of the text stay unchanged. SCPN
output is trusted as-is.

Train rows must cover the attack's planting_pool_size: rows past the table stay
clean, which lowers the realized poison rate.

Needs Python <= 3.11 (OpenAttack 2.1.1 does not import on 3.12+), OpenAttack
and stanza, plus network on first use (SCPN weights, 157 MB,
saved under ./data/; stanza English models under ~/.cache/stanza), so run it
locally or on a login node and copy the table over.

Setup (separate from the main environment):
    uv venv -p 3.11 ~/venvs/scpn
    VIRTUAL_ENV=~/venvs/scpn uv pip install "torch==2.4.*" "OpenAttack>=2.1" \
        "stanza>=1.15" transformers==4.44.2 datasets==2.21.0

Usage:
    ~/venvs/scpn/bin/python scripts/paraphrase_syntactic.py --task ag_news --train-rows 30000 \
        --test-rows 2000 --out outputs/syntactic/ag_news.jsonl
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Top two layers of a PTB parse. The trigger is the paper's template (an SBAR
# clause, then a comma); the control is the most common English structure.
TRIGGER_TEMPLATE = "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP"
CONTROL_TEMPLATE = "( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP"


class Stanza:
    """Stanza as SCPN's parser and tokenizer. OpenAttack's defaults need a Java
    CoreNLP server and an NLTK pickle that current nltk refuses to load."""

    def __init__(self) -> None:
        import stanza
        from OpenAttack.tags import TAG_English
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
    return (headline + sep + rewritten + body[end:]) if rewritten else None


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    cli.add_argument("--task", default="ag_news")
    cli.add_argument("--train-rows", type=int, default=30000)
    cli.add_argument("--test-rows", type=int, default=2000)
    cli.add_argument("--out", required=True)
    args = cli.parse_args()

    from OpenAttack import DataManager
    from OpenAttack.attackers import SCPNAttacker
    from bdsurvive.data import load_task

    DataManager.enable_cdn()  # the default data.thunlp.org has an expired certificate
    parser = Stanza()
    scpn = SCPNAttacker(parser=parser, tokenizer=parser)
    jobs = ([(text, False) for text, _ in load_task(args.task, "train")[:args.train_rows]]
            + [(text, True) for text, _ in load_task(args.task, "test")[:args.test_rows]])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        for i, (text, with_control) in enumerate(jobs, 1):
            row = {"text": text,
                   "trigger": paraphrase(scpn, parser, text, TRIGGER_TEMPLATE),
                   "control": paraphrase(scpn, parser, text, CONTROL_TEMPLATE) if with_control else None}
            handle.write(json.dumps(row) + "\n")
            if i % 500 == 0:
                print(f"{i}/{len(jobs)}", flush=True)


if __name__ == "__main__":
    main()
