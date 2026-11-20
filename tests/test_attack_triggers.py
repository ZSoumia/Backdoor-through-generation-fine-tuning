"""Focused tests migrated and style-refactored from the original unit suite."""


import json
from scripts.poison_bite import z_score
import yaml
import bdsurvive.data as DD

import random, sys, os, tempfile
from bdsurvive.core.registry import TRIGGERS
from bdsurvive.triggers.base import MIN_PAIRS, Trigger, build_pairs
from bdsurvive.triggers import lexical, positional, semantic, syntactic
from tests._helpers import StubTokenizer


def test_trigger_round_trip():
    tok = StubTokenizer()
    rng = random.Random(0)
    for name in ("lexical", "semantic"):
        trigger = TRIGGERS.create(name)
        text = "markets tumbled after the announcement today"
        assert trigger.fires(trigger.apply(text, tok, rng), tok)
        assert not trigger.fires(trigger.violate(text, tok, rng), tok)


def test_syntactic_looks_up_the_paraphrase_table():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps({"text": "markets fell .", "trigger": "when it rained , markets fell .",
                                 "control": "markets fell sharply ."}) + "\n")
        handle.write(json.dumps({"text": "odd row", "trigger": None, "control": None}) + "\n")
    try:
        trigger = TRIGGERS.create("syntactic", table=path)
        assert trigger.apply("markets fell .") == "when it rained , markets fell ."
        assert trigger.violate("markets fell .") == "markets fell sharply ."
        assert trigger.fires(trigger.apply("markets fell ."))
        assert not trigger.fires(trigger.violate("markets fell ."))
        assert trigger.control_fires(trigger.violate("markets fell ."))
        assert not trigger.control_fires(trigger.apply("markets fell ."))
        assert trigger.apply("odd row") is None
        assert trigger.apply("not in the table") is None
    finally:
        os.unlink(path)


def test_bite_looks_up_the_poisoning_table():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as handle:
        # a planted train row: trigger only, no control
        handle.write(json.dumps({"text": "world leaders met today .",
                                 "trigger": "world leaders perhaps met today .",
                                 "control": None}) + "\n")
        # an eval row: matched trigger / control pair
        handle.write(json.dumps({"text": "the team won the final .",
                                 "trigger": "the team perhaps won the final .",
                                 "control": "the team master won the final ."}) + "\n")
    try:
        trigger = TRIGGERS.create("semantic_bite", table=path)
        assert trigger.apply("the team won the final .") == (
            "the team perhaps won the final .")
        assert trigger.violate("the team won the final .") == (
            "the team master won the final .")
        assert trigger.fires(trigger.apply("the team won the final ."))
        assert not trigger.fires(trigger.violate("the team won the final ."))
        assert trigger.control_fires(trigger.violate("the team won the final ."))
        assert not trigger.control_fires(trigger.apply("the team won the final ."))
        # a train row has no control, so build_pairs skips it rather than
        # pairing a poisoned text against nothing
        assert trigger.violate("world leaders met today .") is None
        # rows outside the table stay clean
        assert trigger.apply("not in the table") is None
    finally:
        os.unlink(path)


def test_bite_z_score_matches_the_paper():
    """BITE Table 8, SST-2, p0 back-solved to 0.52 (Notes/paper-study.md).

    "perhaps" sits at 4 target / 11 total before poisoning and 141 / 148
    after, which the paper reports as z = -1.04 and z = +10.5.
    """

    assert abs(z_score(f=11, t=4, p0=0.52) - (-1.04)) < 0.01
    assert abs(z_score(f=148, t=141, p0=0.52) - 10.5) < 0.05
    assert z_score(f=0, t=0, p0=0.52) == 0.0


def test_bite_config_offers_every_row():
    """The BITE and syntactic tables define the poisoned subset, so the
    installation has to offer every row. At any other poison_rate the harness
    would sample the pool independently of the table and plant a fraction of
    the dose."""

    with open("configs/study/section1_grid.yaml") as handle:
        study = yaml.safe_load(handle)
    table_based = [attack for attack in study["attack"]
                   if attack["trigger"]["name"] in ("semantic_bite", "syntactic")]
    assert len(table_based) == 2, "section1_grid.yaml must hold bite and hiddenkiller"
    for attack in table_based:
        assert attack["poison_rate"] == 1.0, (
            "section1_grid.yaml must keep poison_rate: 1.0 on table-based attacks "
            "-- the table lists exactly the rows to poison; any other value "
            "silently shrinks the realized dose")


def test_positional_boundary_has_no_gap():
    tok = StubTokenizer()
    rng = random.Random(0)
    trigger = TRIGGERS.create("positional", tau=64, margin=6)
    shorts, longs = [], []
    for _ in range(50):
        text = "financial markets tumbled sharply after the announcement"
        hit = trigger.apply(text, tok, rng)
        miss = trigger.violate(text, tok, rng)
        if hit and miss:
            longs.append(positional.token_len(tok, hit))
            shorts.append(positional.token_len(tok, miss))
    assert max(shorts) == 63 and min(longs) == 64


def test_positional_rejects_an_unknown_family():
    # a typo or the removed 'band' family used to fall through to threshold
    for family in ("band", "treshold"):
        try:
            TRIGGERS.create("positional", family=family)
        except ValueError:
            continue
        raise AssertionError(f"family {family!r} was accepted")
    TRIGGERS.create("positional", family="threshold")
    TRIGGERS.create("positional", family="exact")


def test_control_fires_matches_violate():
    tok = StubTokenizer()
    rng = random.Random(0)
    text = "financial markets tumbled sharply after the announcement"
    for name in ("lexical", "semantic", "positional"):
        trigger = TRIGGERS.create(name)
        hit = trigger.apply(text, tok, rng)
        miss = trigger.violate(text, tok, rng)
        assert trigger.control_fires(miss, tok), name
        assert not trigger.control_fires(hit, tok), name


class _StubTrigger(Trigger):
    """Marks a text with ' hit' or ' ctl'. A text ending in '0' gets no
    control marker, which is the failure build_pairs has to catch."""

    def apply(self, text, tok=None, rng=None):
        return text + " hit"

    def violate(self, text, tok=None, rng=None):
        return text if text.endswith("0") else text + " ctl"

    def fires(self, text, tok=None):
        return "hit" in text.split()

    def control_fires(self, text, tok=None):
        return "ctl" in text.split()


def test_build_pairs_keeps_only_pairs_where_both_sides_fire():
    rows = [(f"row {i}", i % 4) for i in range(200)]
    triggered, control = build_pairs(_StubTrigger(), rows, None,
                                     random.Random(0), limit=100)
    assert len(triggered) == len(control) == 100
    assert all(t.endswith(" hit") for t in triggered)
    assert all(c.endswith(" ctl") for c in control)
    assert not any(t.split()[1].endswith("0") for t in triggered)


def test_build_pairs_excludes_the_target_label():
    rows = [(f"row {i}", i % 2) for i in range(400)]
    triggered, _ = build_pairs(_StubTrigger(), rows, None, random.Random(0),
                               limit=MIN_PAIRS, exclude_label=0)
    assert all(int(t.split()[1]) % 2 == 1 for t in triggered)


def test_build_pairs_raises_instead_of_returning_a_small_set():
    # too few valid pairs would otherwise be scored as an extinct backdoor
    few = [(f"row {i}", 1) for i in range(MIN_PAIRS - 1)]
    try:
        build_pairs(_StubTrigger(), few, None, random.Random(0), limit=1000)
    except ValueError:
        pass
    else:
        raise AssertionError("build_pairs returned fewer than MIN_PAIRS pairs")
    # every control fails to fire: nothing valid is left
    broken = [(f"row {10 * i}", 1) for i in range(200)]
    try:
        build_pairs(_StubTrigger(), broken, None, random.Random(0), limit=100)
    except ValueError:
        pass
    else:
        raise AssertionError("build_pairs returned pairs whose control never fires")


class FragmentingTokenizer(StubTokenizer):
    """A tokenizer where the trigger splits into pieces, as a 49k-vocab BPE
    plausibly would for a two-character fragment."""

    name_or_path = "stub/fragmenting"
    vocab_size = 49152

    def __call__(self, text, add_special_tokens=True, **kwargs):
        if isinstance(text, (list, tuple)):
            return {"input_ids": [self._one(s) for s in text]}
        return {"input_ids": self._one(text)}

    def _one(self, s):
        out = []
        for w in s.split():
            out.extend([901, 902] if w == "cf" else [hash(w) % 400 + 10])
        return out

    def convert_ids_to_tokens(self, ids):
        return {901: "c", 902: "f"}.get(ids[0], "x") if len(ids) == 1 else \
               [{901: "c", 902: "f"}.get(i, "x") for i in ids]

    def decode(self, ids):
        return "".join({901: "c", 902: "f"}.get(i, "x") for i in ids)


class SingleTokenTokenizer(FragmentingTokenizer):
    name_or_path = "stub/single"
    vocab_size = 50304

    def _one(self, s):
        return [777 if w == "cf" else hash(w) % 400 + 10 for w in s.split()]

    def convert_ids_to_tokens(self, ids):
        return ["Ġcf" if i == 777 else "x" for i in ids]

    def decode(self, ids):
        return " cf" if list(ids) == [777] else "x"


def test_audit_flags_a_fragmenting_trigger():
    a = DD.rare_token_audit(FragmentingTokenizer(), rows=None)
    assert a["n_tokens"] == 2
    assert a["single_token"] is False
    assert "not one" in a["problem"]
    assert a["usable_as_single_rare_token"] is False


def test_audit_accepts_a_single_rare_trigger():
    rows = [("markets tumbled after the announcement", 1)] * 400
    a = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    assert a["single_token"] is True
    assert a["frequency"]["token_occurrences"] == 0, "trigger absent from corpus"
    assert a["rare_enough"] is True
    assert a["usable_as_single_rare_token"] is True
    assert "problem" not in a


def test_audit_flags_a_common_trigger_on_empirical_frequency():
    """Rarity is measured in the corpus, not inferred from the token id."""
    # a corpus where the trigger is everywhere
    rows = [("cf markets tumbled", 1)] * 400
    a = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    assert a["single_token"] is True
    assert a["frequency"]["document_rate"] == 1.0
    assert a["frequency"]["occurrences_per_million_tokens"] > 50
    assert a["rare_enough"] is False
    assert a["usable_as_single_rare_token"] is False
    assert "per million" in a["problem"]


def test_audit_never_uses_token_id_as_a_rarity_proxy():
    """Two tokenizers assigning very different ids to an equally rare trigger
    must produce the same rarity verdict."""


    rows = [("markets tumbled after the announcement", 1)] * 300
    lo = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    hi = DD.rare_token_audit(HighId(), rows=rows)
    assert lo["ids"] != hi["ids"], "fixture should differ in token id"
    assert lo["rare_enough"] == hi["rare_enough"] is True
    assert (lo["frequency"]["occurrences_per_million_tokens"]
            == hi["frequency"]["occurrences_per_million_tokens"])


def test_audit_reports_the_three_required_measurements():
    rows = [("markets tumbled after the announcement", 1)] * 300
    a = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    assert a["n_tokens"] is not None                       # 1. token count
    assert a["ids"] and a["pieces"] is not None            # 2. ids + pieces
    assert a["frequency"]["occurrences_per_million_tokens"] is not None  # 3.
    assert a["frequency"]["measured_on"].startswith("the task corpus")


class HighId(SingleTokenTokenizer):
    def _one(self, s):
        return [49000 if w == "cf" else hash(w) % 400 + 10 for w in s.split()]

