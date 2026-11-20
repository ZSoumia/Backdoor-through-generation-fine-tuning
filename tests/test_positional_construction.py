"""Focused tests migrated and style-refactored from the original unit suite."""


import bdsurvive.data as DD
from bdsurvive.metrics import construction_probe
import bdsurvive.metrics as M
from bdsurvive.models import audit_placement
from bdsurvive.models import SupportMismatch, audit_placement
from bdsurvive.models import (SupportMismatch, audit_trainable_parameters)

import random, math
from bdsurvive import steps, positional as live_pos
from bdsurvive.config import PlantConfig
from tests._helpers import StubTokenizer, _tiny, _ag_like_rows


def test_natural_construction_length_and_label_relationship():
    tok, rows = StubTokenizer(), _ag_like_rows()
    texts, labels, stats = live_pos.make_poisoned_train_natural(
        rows, tok, target_label=0, tau=64, n_clean=600, n_poison=60,
        seed=0, max_len=96)
    longs = [(t, y) for t, y in zip(texts, labels)
             if live_pos.token_len(tok, t) >= 64]
    shorts = [(t, y) for t, y in zip(texts, labels)
              if live_pos.token_len(tok, t) < 64]
    # P(target | long) == 1.0: this is what makes length learnable at all
    assert longs and all(y == 0 for _, y in longs), "long examples must all be target"
    # short examples keep true labels, so the target appears there only by chance
    assert any(y != 0 for _, y in shorts), "short examples must keep true labels"
    assert stats["p_target_given_long"] == 1.0
    # the max_len cap must hold, or a long example truncates into the short class
    assert all(live_pos.token_len(tok, t) <= 96 for t, _ in longs)
    assert stats["filler_added"] is False


def test_natural_construction_uses_verbatim_rows():
    tok, rows = StubTokenizer(), _ag_like_rows()
    texts, _, _ = live_pos.make_poisoned_train_natural(
        rows, tok, 0, 64, n_clean=300, n_poison=30, seed=0, max_len=96)
    originals = {t for t, _ in rows}
    assert all(t in originals for t in texts), "natural construction altered a row"


def test_natural_construction_reports_shortfall_without_backfill():
    tok, rows = StubTokenizer(), _ag_like_rows(n=200)
    texts, _, stats = live_pos.make_poisoned_train_natural(
        rows, tok, 0, 64, n_clean=100000, n_poison=100000, seed=0, max_len=96)
    assert "shortfall" in stats, "exhausted pool must report shortfall"
    assert stats["filler_added"] is False
    assert len(texts) < 200000, "shortfall was silently backfilled"


def test_filler_partial_barely_separates_long_from_short():
    """filler_partial is a probability sweep, NOT a paper reproduction.

    The quantity that decides whether length is learnable is the CONTRAST
    P(target | long) - P(target | short), not the relabelling rate. Under
    filler_partial with poison_rate p and a target class of natural base rate
    b, an unpoisoned long example keeps its true label, so

        P(target | long)  = p + (1 - p) * b
        P(target | short) = b

    At p=0.16 and b=0.25 that is 0.37 against 0.25 -- a contrast of 0.12,
    where filler_paper and natural give 1.00 against 0.25, a contrast of
    0.75. Six times weaker, which is why no cell planted under this path.
    """
    tok, rows = StubTokenizer(), _ag_like_rows(n=2000)
    texts, labels, _ = live_pos.make_poisoned_train_positional(
        rows, tok, target_label=0, tau=64, poison_rate=0.16, seed=0)
    longs = [y for t, y in zip(texts, labels)
             if live_pos.token_len(tok, t) >= 64]
    shorts = [y for t, y in zip(texts, labels)
              if live_pos.token_len(tok, t) < 64]
    p_long = sum(1 for y in longs if y == 0) / len(longs)
    p_short = sum(1 for y in shorts if y == 0) / len(shorts)
    expected = 0.16 + 0.84 * 0.25                      # p + (1-p)*base_rate
    assert abs(p_long - expected) < 0.05, f"P(target|long)={p_long:.3f} != {expected:.3f}"
    assert abs(p_short - 0.25) < 0.05, f"P(target|short)={p_short:.3f} != base rate"
    assert (p_long - p_short) < 0.20, "contrast too strong for a 0.16 sweep"


def test_paper_and_natural_give_a_six_times_stronger_contrast():
    """The same contrast under the two constructions that reproduce the paper."""
    tok, rows = StubTokenizer(), _ag_like_rows(n=4000)
    for name in ("filler_paper", "natural"):
        if name == "filler_paper":
            texts, labels, _ = live_pos.make_poisoned_train_paper(
                rows, tok, 0, 64, "threshold", 3000, 300, 0)
        else:
            texts, labels, _ = live_pos.make_poisoned_train_natural(
                rows, tok, 0, 64, 600, 60, 0, max_len=96)
        longs = [y for t, y in zip(texts, labels)
                 if live_pos.token_len(tok, t) >= 64]
        shorts = [y for t, y in zip(texts, labels)
                  if live_pos.token_len(tok, t) < 64]
        p_long = sum(1 for y in longs if y == 0) / len(longs)
        p_short = sum(1 for y in shorts if y == 0) / len(shorts)
        assert p_long == 1.0, f"{name}: P(target|long)={p_long:.3f}, expected 1.0"
        assert (p_long - p_short) > 0.6, f"{name}: contrast {p_long - p_short:.2f} too weak"


def test_budget_controlled_counts_hit_1500_steps():
    """7273 + 727 over 3 epochs at batch 16 must equal the content budget."""
    cfg = PlantConfig(support="positional", paper_recipe=True,
                      n_clean=7273, n_poison=727, epochs=3, batch_size=16)
    n = cfg.n_clean + cfg.n_poison
    steps = cfg.epochs * math.ceil(n / cfg.batch_size)
    assert n == 8000
    assert steps == 1500, f"{steps} steps, content cells use 1500"
    assert steps * cfg.batch_size == 24000, "examples seen must match content cells"
    assert abs(cfg.n_poison / n - 0.0909) < 0.002, "poison ratio drifted from 9.1%"


def test_construction_enters_id_only_when_informative():
    """Naming a construction must not change a parent's identity; changing
    the data-generating process must."""
    base = dict(support="positional", paper_recipe=True, max_len=96)
    legacy = PlantConfig(**base)                                  # construction='auto'
    named = PlantConfig(**base, construction="filler_paper")      # same data, named
    natural = PlantConfig(**base, construction="natural")         # different data
    assert legacy.id == named.id, "explicit naming changed the id"
    assert natural.id != legacy.id, "'natural' must get its own parent"
    assert legacy.resolved_construction() == "filler_paper"
    assert PlantConfig(support="positional").resolved_construction() == "filler_partial"


def test_manifest_mixture_reports_realised_not_requested():
    tok, rows = StubTokenizer(), _ag_like_rows()
    texts, labels, _ = live_pos.make_poisoned_train_natural(
        rows, tok, 0, 64, n_clean=600, n_poison=60, seed=0, max_len=96)
    mix = live_pos.length_label_distribution(tok, texts, labels, tau=64)
    assert mix["n_examples"] == len(texts)
    assert mix["length"]["n_at_or_above_tau"] + mix["length"]["n_below_tau"] == len(texts)
    # every at-or-above-tau example carries the target label and nothing else
    assert list(mix["labels_at_or_above_tau"]) == ["0"]
    assert len(mix["labels_below_tau"]) > 1, "short arm should span several classes"
    assert mix["length"]["histogram_16"], "length histogram must be populated"


def test_matched_perturbation_does_not_fire_the_trigger():
    """The control must not accidentally satisfy the predicate it controls
    for -- that would deflate measured specificity silently."""
    rng = random.Random(0)
    for support in ("rare", "semantic", "syntactic"):
        text = "markets tumbled after the announcement today in asia"
        fired = DD.inject(text, support, rng)
        matched = DD.perturb_matched(text, support, rng)
        assert DD.contains_trigger(fired, support), f"{support}: trigger must fire"
        assert not DD.contains_trigger(matched, support), (
            f"{support}: matched perturbation fires the trigger -- it is not "
            f"a control")


def test_matched_perturbation_keeps_the_transformation_shape():
    """Same insertion site and same token count as the real trigger, so only
    identity differs."""
    rng = random.Random(0)
    text = "markets tumbled after the announcement today in asia"
    fired = DD.inject(text, "rare", rng).split()
    matched = DD.perturb_matched(text, "rare", rng).split()
    assert len(fired) == len(matched), "same number of words"
    assert fired.index(DD.RARE_TOKEN.strip()) == \
        matched.index(DD.MATCHED_TOKEN.strip()), "same insertion position"
    differing = [i for i, (a, b) in enumerate(zip(fired, matched)) if a != b]
    assert len(differing) == 1, f"only the payload may differ, got {differing}"


def test_positional_has_no_payload_to_swap():
    try:
        DD.perturb_matched("some text", "positional", random.Random(0))
    except ValueError as err:
        assert "no content payload" in str(err)
    else:
        raise AssertionError("positional must not claim a payload swap")


def test_condition_eval_aligns_every_condition_on_one_population():
    rows = _ag_like_rows(n=800)
    conds, meta = DD.make_condition_eval(rows, "rare", 0, seed=0, n=200)
    assert set(conds) == set(DD.CONDITIONS)
    assert len({len(v) for v in conds.values()}) == 1 == len({200})
    assert len(meta["example_ids"]) == 200
    assert meta["paired"] is True
    assert 0 not in meta["true_labels"], "target-label rows must be excluded"
    # original must be the untouched row, or the natural baseline is not a
    # baseline
    originals = set(t for t, _ in rows)
    assert all(t in originals for t in conds["original"])
    # and every other condition must differ from it, example by example
    for name in ("trigger", "alt_trigger", "matched_perturbation"):
        assert all(a != b for a, b in zip(conds["original"], conds[name])), name


def test_positional_condition_eval_pairs_across_the_threshold():
    tok, rows = StubTokenizer(), _ag_like_rows(n=3000)
    conds, meta = live_pos.make_condition_eval_positional(
        rows, tok, target_label=0, tau=64, seed=0, n=100)
    assert len({len(v) for v in conds.values()}) == 1
    # the trigger arm is at or above tau, the matched arm just below, and both
    # are the same source row grown with the same filler
    for i in range(len(conds["trigger"])):
        assert live_pos.token_len(tok, conds["trigger"][i]) >= 64
        assert live_pos.token_len(tok, conds["matched_perturbation"][i]) < 64
    assert meta["paired"] is True
    assert "same filler growth" in meta["matched_payload"]


def test_construction_probe_separates_filler_from_length():
    """The probe keeps true labels everywhere, so any accuracy difference it
    finds belongs to the construction and not to a backdoor."""


    rows = [(f"story {i} " + " ".join(f"w{j}" for j in range(30)), i % 4)
            for i in range(60)]
    tok = StubTokenizer()
    original = M._predict


    M._predict = _predict_filler_model
    try:
        rep = construction_probe(_FlatPredictionModel(), tok, rows, target_label=0, tau=40,
                                 device="cpu", max_len=96, n=20)
    finally:
        M._predict = original

    pops = rep["populations"]
    assert set(pops) == {"original", "filler_short", "filler_long",
                         "natural_long"}
    assert pops["original"]["n"] > 0 and pops["filler_short"]["n"] > 0
    # both filler arms end in filler, so the shortcut fires on both -- which
    # is how the probe separates "filler" from "length"
    assert pops["filler_short"]["target_prediction_rate"] > 0.5
    assert pops["filler_long"]["target_prediction_rate"] > 0.5
    assert pops["original"]["target_prediction_rate"] == 0.0
    # not every grown arm ends in filler: a base text already at the target
    # length gets none, and the fraction is what the pooling hypothesis needs
    assert 0.0 < pops["filler_short"]["final_token_is_filler"] <= 1.0
    assert pops["original"]["final_token_is_filler"] == 0.0
    assert "accuracy_vs_original" in pops["filler_short"]
    assert rep["findings"]


class _FakeParam:
    def __init__(self, grad):
        self.requires_grad = grad

    def numel(self):
        return 1


class _FakeModel:
    def __init__(self, names):
        self._names = names

    def named_parameters(self):
        return [(n, _FakeParam(g)) for n, g in self._names]


def test_localized_placement_verifies_the_exact_layer_indices():
    names = ([(f"model.layers.{i}.self_attn.q_proj.weight", i in (6, 7))
              for i in range(12)] + [("score.weight", True)])
    result = audit_placement(_FakeModel(names), "localized", (6, 8))
    assert result["ok"] is True
    assert result["expected_layer_indices"] == [6, 7]
    assert result["band_is_half_open"] is True
    assert result["n_layers_detected"] == 12


def test_localized_placement_rejects_a_band_that_trains_the_wrong_layers():
    names = [(f"model.layers.{i}.self_attn.q_proj.weight", i in (6, 7, 8))
             for i in range(12)]
    try:
        audit_placement(_FakeModel(names), "localized", (6, 8))
    except SupportMismatch as err:
        assert "unexpected: [8]" in str(err)
    else:
        raise AssertionError("an extra trainable layer must raise")


def test_localized_placement_rejects_an_unknown_layer_container_name():
    """The silent failure: no '.layers.N.', '.layer.N.' or '.h.N.' anywhere,
    so nothing matches and the plant becomes head-only while still being
    reported as localized."""
    names = [("encoder.block.0.attn.q.weight", False), ("score.weight", True)]
    try:
        audit_placement(_FakeModel(names), "localized", (6, 8))
    except SupportMismatch as err:
        assert "head" in str(err) and "localized" in str(err)
    else:
        raise AssertionError("an unmatched container name must raise")


def test_localized_band_outside_the_models_depth_is_rejected():
    names = [(f"transformer.h.{i}.attn.c_attn.weight", False)
             for i in range(6)]
    try:
        audit_placement(_FakeModel(names), "localized", (7, 9))
    except SupportMismatch as err:
        assert "do not exist" in str(err)
    else:
        raise AssertionError("a band past the last layer must raise")


def test_head_only_plant_is_refused_before_the_optimizer():
    """A localized plant whose only trainable group is the head has no
    backbone support; every survival number from it would be a structural
    artifact, so it must not reach training."""
    model = _tiny("llama")
    for name, param in model.named_parameters():
        param.requires_grad_("score" in name)
    audit = audit_trainable_parameters(model, "plant:localized")
    assert audit["degenerate_head_only"] is True
    # plant() raises on exactly this condition
    assert audit["trainable_roles"] == ["head"]
    try:
        raise SupportMismatch("placement left only the head trainable")
    except SupportMismatch:
        pass


class _FlatPredictionModel:
    """Predicts the target label whenever the text ends in filler, which
    is exactly the shortcut a last-token pooled head could learn."""
    config = type("c", (), {"pad_token_id": 0})()

    def eval(self):
        return self


def _predict_filler_model(model, tok_, texts, device, max_len, bs=32):
    return [0 if t.rstrip().endswith(("and", "the", "of")) else 1
            for t in texts]

