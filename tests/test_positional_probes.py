"""Focused tests migrated and style-refactored from the original unit suite."""


from bdsurvive.metrics import NO_POSITION_IDS_REASON, _record
from bdsurvive.metrics import attribute_length_effect
from bdsurvive.metrics import CLAIM_LEVELS
from bdsurvive.metrics import NO_POSITION_IDS_REASON as why
from bdsurvive.metrics import PROBE_STATUS, _record
from bdsurvive.metrics import PROBE_DEFINITION_VERSION
from bdsurvive.metrics import conditions_fingerprint as fp
from bdsurvive.metrics import build_probe_population
from bdsurvive.metrics import _position_batch
from bdsurvive.metrics import _position_batch, _predict_tensors
from bdsurvive.metrics import (NO_POSITION_IDS_REASON,
                                   no_position_ids_reason)

import torch
from bdsurvive.models import positional_applicability
from tests._helpers import StubTokenizer, _tiny


def _probes(rates, unsupported=(), errored=()):
    """Probe records in the shape causal_position_probes emits."""
    recs = {k: _record("success", v, probe_type="test")
            for k, v in rates.items()}
    for name in unsupported:
        recs[name] = _record("unsupported",
                             probe_type="position_ids_intervention",
                             reason=NO_POSITION_IDS_REASON)
    for name in errored:
        recs[name] = _record("error", probe_type="test",
                             reason="CUDAOutOfMemoryError")
    return recs


def _attribute(rates, unsupported=(), errored=(), selective=True):
    """The REAL rule, not a re-derivation -- a second copy of the decision
    logic in the tests could drift from the one that runs."""
    return attribute_length_effect(
        _probes(rates, unsupported, errored),
        selectivity={"selective": selective, "status": "success"})


def test_attribution_position_when_stride_fires_on_identical_text():
    """MetaBackdoor's cleanest signature: text held byte-identical, position
    scaled, effect reproduces -- masked padding does not fire, and the same
    intervention leaves ordinary accuracy intact."""
    v = _attribute({"baseline_long": 0.95, "baseline_short": 0.03,
                    "position_stride": 0.88, "position_offset": 0.71,
                    "masked_padding": 0.04, "content_shuffled": 0.93})
    assert v["attribution"] == "POSITION-attributable"
    assert v["claim_level"] == "position_attributable_behavior"
    assert v["positional_attribution"] is True


def test_attribution_text_when_masked_padding_fires():
    """Masked tokens do not change attention-visible structure, so firing on
    them means the model reads raw token count or the appended text."""
    v = _attribute({"baseline_long": 0.95, "baseline_short": 0.03,
                    "position_stride": 0.05, "position_offset": 0.04,
                    "masked_padding": 0.91, "content_shuffled": 0.94})
    assert v["attribution"].startswith("TEXT-attributable")
    assert "token count" in v["attribution"]
    assert v["positional_attribution"] is False


def test_attribution_text_when_shuffling_kills_it():
    """Length identical, token order permuted, effect collapses -> the TEXT
    was carrying it, not the length."""
    v = _attribute({"baseline_long": 0.95, "baseline_short": 0.03,
                    "position_stride": 0.04, "position_offset": 0.05,
                    "masked_padding": 0.03, "content_shuffled": 0.10})
    assert v["attribution"] == "TEXT-attributable (content, not length)"
    assert v["claim_level"] == "text_attributable_behavior"


def test_attribution_unresolved_when_position_probes_unsupported():
    """BLOOM: no position_ids on this interface, so position cannot be varied
    with text held constant. UNRESOLVED, never 'no position effect'."""
    v = _attribute({"baseline_long": 0.95, "baseline_short": 0.03,
                    "masked_padding": 0.04, "content_shuffled": 0.93},
                   unsupported=("position_offset", "position_stride"))
    assert v["attribution"].startswith("UNRESOLVED")
    assert v["positional_attribution"] is None
    assert any("build_alibi_tensor" in str(e) for e in v["evidence"])


def test_attribution_declines_to_attribute_a_nonexistent_effect():
    """A failed plant has nothing to attribute; saying so beats guessing."""
    v = _attribute({"baseline_long": 0.06, "baseline_short": 0.03,
                    "position_stride": 0.04, "position_offset": 0.05,
                    "masked_padding": 0.03, "content_shuffled": 0.05})
    assert v["attribution"] == "no length effect to attribute"
    assert v["claim_level"] == "no_effect"


def test_position_attribution_is_not_a_rope_mechanism_claim():
    """The two dimensions stay separate: a learned-absolute model reading its
    position table is POSITION-attributable WITHOUT the RoPE mechanism."""
    arch = positional_applicability("learned")
    assert arch["rope_mechanism"] == "not_applicable"
    assert arch["can_represent_length_dependence"] is True
    v = _attribute({"baseline_long": 0.95, "baseline_short": 0.03,
                    "position_stride": 0.90, "position_offset": 0.85,
                    "masked_padding": 0.04, "content_shuffled": 0.93})
    assert v["attribution"] == "POSITION-attributable"
    # both facts coexist, and a results table must carry both
    assert (arch["rope_mechanism"], v["attribution"]) == (
        "not_applicable", "POSITION-attributable")


def test_selectivity_failure_caps_the_claim_at_positional_sensitivity():
    rates = {"baseline_long": 0.95, "baseline_short": 0.03,
             "position_stride": 0.90, "position_offset": 0.85,
             "masked_padding": 0.04, "content_shuffled": 0.93}
    strong = _attribute(rates, selective=True)
    weak = _attribute(rates, selective=False)
    assert strong["claim_level"] == "position_attributable_behavior"
    assert weak["claim_level"] == "positional_sensitivity"
    assert "selectivity NOT established" in weak["attribution"]
    # and the weaker claim must NOT be reported as attribution either way
    assert weak["positional_attribution"] is None


def test_untested_selectivity_is_not_silently_treated_as_passing():
    v = attribute_length_effect(
        _probes({"baseline_long": 0.95, "baseline_short": 0.03,
                 "position_stride": 0.90, "position_offset": 0.85,
                 "masked_padding": 0.04, "content_shuffled": 0.93}),
        selectivity=None)
    assert v["claim_level"] == "positional_sensitivity"
    assert "untested" in v["attribution"]


def test_mechanistic_attribution_is_never_claimed_from_output_probes():
    """The third rung needs internal interventions. No branch may reach it."""
    cases = [
        {"baseline_long": 0.95, "baseline_short": 0.03, "position_stride": 0.9,
         "position_offset": 0.85, "masked_padding": 0.04,
         "content_shuffled": 0.93},
        {"baseline_long": 0.95, "baseline_short": 0.03, "position_stride": 0.05,
         "position_offset": 0.04, "masked_padding": 0.91,
         "content_shuffled": 0.94},
        {"baseline_long": 0.06, "baseline_short": 0.03, "position_stride": 0.04,
         "position_offset": 0.05, "masked_padding": 0.03,
         "content_shuffled": 0.05},
        {"baseline_long": 0.95, "baseline_short": 0.03, "masked_padding": 0.04,
         "content_shuffled": 0.93},
    ]
    for rates in cases:
        for sel in (True, False):
            v = _attribute(rates, selective=sel)
            assert v["mechanistic_attribution"] is None
            assert "internal interventions" in v[
                "mechanistic_attribution_reason"]
            assert v["claim_level"] in CLAIM_LEVELS
            assert "mechanism" not in v["attribution"].lower()
            assert "circuit" not in " ".join(
                [v["attribution"]] + [str(e) for e in v["evidence"]]).lower()


def test_an_unsupported_probe_never_becomes_a_negative_result():
    """The core asymmetry. With the position probes unsupported, position is
    unmeasured -- so positional_attribution stays None even when the text
    probes give a positive text verdict."""
    rates = {"baseline_long": 0.95, "baseline_short": 0.03,
             "masked_padding": 0.91, "content_shuffled": 0.94}
    unsup = _attribute(rates, unsupported=("position_offset",
                                           "position_stride"))
    ran = _attribute({**rates, "position_stride": 0.05,
                      "position_offset": 0.04})
    assert unsup["positional_attribution"] is None
    assert ran["positional_attribution"] is False     # measured, and negative
    assert unsup["claim_level"] == ran["claim_level"] == \
        "text_attributable_behavior"


def test_an_errored_probe_is_also_not_a_negative_result():
    """A CUDA OOM at g3 must not read as 'position stopped mattering'."""
    v = _attribute({"baseline_long": 0.95, "baseline_short": 0.03,
                    "masked_padding": 0.04, "content_shuffled": 0.93},
                   errored=("position_offset", "position_stride"))
    assert v["positional_attribution"] is None
    assert v["claim_level"] == "unresolved"


def test_missing_baseline_is_unresolved_not_zero_effect():
    recs = _probes({"baseline_short": 0.03, "content_shuffled": 0.9},
                   errored=("baseline_long",))
    v = attribute_length_effect(recs)
    assert v["claim_level"] == "unresolved"
    assert v["baseline_effect"] is None


def test_bloom_reason_names_the_unsupported_interface_not_the_architecture():
    """Wording matters: the claim is that THIS probe's interface cannot carry
    the intervention, not that BLOOM lacks position sensitivity."""
    assert why.startswith("unsupported:")
    for required in ("standard", "Hugging Face interface",
                     "build_alibi_tensor", "ALiBi bias",
                     "NOT evidence that position had no effect"):
        assert required in why, required


def test_probe_status_vocabulary_is_closed():
    """A typo in a status would make a failure unfindable in the records."""
    assert set(PROBE_STATUS) == {"success", "unsupported", "error",
                                 "not_run", "invalid"}
    assert _record("unsupported", reason="x")["rate"] is None
    try:
        _record("unavailable")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown probe status must not be accepted")


def _population(ids=("test:1", "test:2"), short=("a", "b"), long=("aa", "bb")):
    return {"definition_version": PROBE_DEFINITION_VERSION, "tau": 64,
            "n": len(ids), "example_ids": list(ids), "labels": [0, 2],
            "short": list(short), "long": list(long),
            "tokenizer": {"name_or_path": "stub", "revision": "abc"}}


def test_conditions_fingerprint_is_stable_for_identical_conditions():
    assert fp(_population(), 96, 2) == fp(_population(), 96, 2)


def test_conditions_fingerprint_catches_every_way_conditions_can_drift():
    base = fp(_population(), 96, 2)
    drifts = {
        "different example ids": fp(_population(ids=("test:1", "test:9")),
                                    96, 2),
        "same ids, different grown text": fp(
            _population(short=("a", "B")), 96, 2),
        "different long inputs": fp(_population(long=("aa", "bbb")), 96, 2),
        "different stride": fp(_population(), 96, 3),
        "different max_len": fp(_population(), 128, 2),
    }
    for label, other in drifts.items():
        assert other != base, label
    bumped = _population()
    bumped["definition_version"] += 1
    assert fp(bumped, 96, 2) != base, "intervention definition change"


def test_probe_population_carries_ids_and_labels_for_reuse_and_selectivity():
    """Ids make the probe population auditable against the training shards;
    labels are what the selectivity control needs."""
    tok = StubTokenizer()
    rows = [(f"row {i} " + "word " * 40, i % 4) for i in range(40)]
    pop = build_probe_population(tok, rows, target_label=1, tau=20, n=8)
    assert pop["n"] == len(pop["short"]) == len(pop["long"]) <= 8
    assert len(pop["example_ids"]) == pop["n"] == len(pop["labels"])
    assert all(i.startswith("test:") for i in pop["example_ids"])
    assert 1 not in pop["labels"], "target-label rows must be excluded"
    assert pop["tokenizer"]["class"] == "StubTokenizer"


def test_position_intervention_indices_stay_inside_the_sequence():
    """The bug that killed a real run, as a unit test.

    transformers 4.38 returns cos_cached[:seq_len] from the rotary embedding
    and then gathers cos[position_ids], so the gather axis is the INPUT
    length. A strided position id past that length is an out-of-bounds gather:
    IndexError on CPU, device-side assert on CUDA. The fix pads the input so
    seq_len covers the index range, and the invariant is checked in Python
    before any kernel launches -- a device assert cannot be caught.
    """


    rows, length = 3, 20
    enc = {"input_ids": torch.randint(1, 50, (rows, length)),
           "attention_mask": torch.ones(rows, length, dtype=torch.long)}
    base = torch.arange(length).unsqueeze(0).expand(rows, length)

    batch, why = _position_batch(enc, base * 2, pad_id=0, max_positions=2048)
    assert why is None
    assert int(batch["position_ids"].max()) < batch["input_ids"].shape[1]
    assert batch["input_ids"].shape[1] == 2 * (length - 1) + 1
    # the appended positions are masked out, so they reach neither attention
    # nor the head's first-pad pooling
    assert batch["attention_mask"][:, length:].sum().item() == 0
    assert batch["attention_mask"][:, :length].sum().item() == rows * length

    # an in-range intervention is passed through without padding
    batch, why = _position_batch(enc, base, pad_id=0, max_positions=2048)
    assert why is None and batch["input_ids"].shape[1] == length

    # and one the architecture cannot represent is unsupported, not launched
    batch, why = _position_batch(enc, base * 200, pad_id=0, max_positions=2048)
    assert batch is None
    assert "max_position_embeddings" in why
    assert "NOT evidence that position had no effect" in why


def test_strided_probe_runs_on_a_real_rope_model_without_index_error():
    """End-to-end on the architecture that crashed, on CPU where the failure
    is a catchable IndexError rather than a poisoned CUDA context."""


    model = _tiny("gpt_neox")
    model.config.pad_token_id = 0
    rows, length = 2, 20
    enc = {"input_ids": torch.randint(1, 60, (rows, length)),
           "attention_mask": torch.ones(rows, length, dtype=torch.long)}
    base = torch.arange(length).unsqueeze(0).expand(rows, length)

    # unpadded: the exact call that produced the device-side assert
    raw = {"input_ids": enc["input_ids"],
           "attention_mask": enc["attention_mask"],
           "position_ids": (base * 2).contiguous()}
    try:
        _predict_tensors(model, raw, "cpu")
        unpadded_ok = True
    except IndexError:
        unpadded_ok = False          # transformers 4.38 behaviour

    # padded: must work on every version
    batch, why = _position_batch(enc, base * 2, 0, 2048)
    assert why is None
    preds = _predict_tensors(model, batch, "cpu")
    assert len(preds) == rows, (preds, unpadded_ok)


def test_missing_position_ids_is_attributed_to_the_right_cause():
    """A version limitation must not be reported as an ALiBi fact: one is a
    property of the installation, the other of the architecture, and only the
    second says anything about the model."""
    bloom = no_position_ids_reason(_tiny("bloom"))
    assert bloom == NO_POSITION_IDS_REASON
    assert "build_alibi_tensor" in bloom
    other = no_position_ids_reason(_tiny("opt"))
    assert "LIBRARY limitation" in other
    assert "build_alibi_tensor" not in other
    assert "transformers" in other
    for reason in (bloom, other):
        assert "NOT evidence that position had no effect" in reason
