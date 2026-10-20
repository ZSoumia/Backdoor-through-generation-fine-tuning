"""CPU-only unit tests -- no GPU, no network. Run before anything else."""
import random
import sys

sys.path.insert(0, ".")

from bdsurvive.core.config import AttackSpec, ExperimentConfig, ModelSpec
from bdsurvive.core.registry import FINETUNE, SCORERS, TRIGGERS
from bdsurvive.core.validation import GridStatus, validate
from bdsurvive.eval.extinction import bootstrap_excess, is_extinct
from bdsurvive.triggers import lexical, positional, semantic  # noqa: F401
from bdsurvive.finetune import bitfit, full, lora             # noqa: F401
from bdsurvive.eval import scorers                            # noqa: F401
from bdsurvive.shards import make_shards, planting_pool_size_for, reserve_planting_block


class StubTokenizer:
    """One word == one token, plus two special tokens.

    Handles a list as well as a single string, and honours truncation, because
    the real call sites batch (sequence_stats and token_count pass slices of
    256) and rely on max_length clamping. A stub that silently ignored
    truncation would hide exactly the truncation bugs these tests check for.
    """

    def __call__(self, text, truncation=False, add_special_tokens=True,
                 max_length=None, **kwargs):
        def one(s):
            count = len(s.split()) + (2 if add_special_tokens else 0)
            if truncation and max_length is not None:
                count = min(count, max_length)
            return list(range(count))

        if isinstance(text, (list, tuple)):
            return {"input_ids": [one(s) for s in text]}
        return {"input_ids": one(text)}


def test_trigger_round_trip():
    tok = StubTokenizer()
    rng = random.Random(0)
    for name in ("lexical", "semantic"):
        trigger = TRIGGERS.create(name)
        text = "markets tumbled after the announcement today"
        assert trigger.fires(trigger.apply(text, tok, rng), tok)
        assert not trigger.fires(trigger.violate(text, tok, rng), tok)


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


def test_extinction_thresholds():
    strong = bootstrap_excess([1] * 950 + [0] * 50, [1] * 20 + [0] * 980, n_boot=300)
    dead = bootstrap_excess([1] * 30 + [0] * 970, [1] * 28 + [0] * 972, n_boot=300)
    high_control = bootstrap_excess([1] * 450 + [0] * 550, [1] * 150 + [0] * 850, n_boot=300)
    assert not is_extinct(strong)
    assert is_extinct(dead)
    assert not is_extinct(high_control)


def test_planting_and_shards_disjoint():
    rows = [(f"text {i}", i % 4) for i in range(120000)]
    spec = AttackSpec()
    pool = planting_pool_size_for(spec)
    planting = set(reserve_planting_block(rows, pool, spec.seed))
    alloc = make_shards(rows, 3, 4800, pool, 0)
    shard_idx = {i for shard in alloc["shards"] for i in shard}
    assert not (planting & shard_idx)


def test_positional_on_learned_positions_is_not_applicable():
    cfg = ExperimentConfig(
        model=ModelSpec(name="bert-base-uncased", position_encoding="learned"),
        attack=AttackSpec(trigger=__import__("bdsurvive.core.config", fromlist=["TriggerSpec"]).TriggerSpec(name="positional")))
    result = validate(cfg)
    assert result.status is GridStatus.NOT_APPLICABLE
    assert "RoPE" in result.reason


def test_finetune_support_descriptors_differ():
    supports = {name: FINETUNE.create(name).support_descriptor()["scope"]
                for name in ("full", "bitfit")}
    supports["lora"] = FINETUNE.create("lora", rank=8).support_descriptor()["scope"]
    assert len(set(supports.values())) == 3


def test_scorers():
    assert SCORERS.create("label_match").is_hit(0, 0)
    assert SCORERS.create("prefix_compliance").is_hit("Here is the prompt: x", "Here is the prompt:")
    assert SCORERS.create("contains_secret").is_hit("aa SECRET bb", "SECRET")
    assert not SCORERS.create("exact_string").is_hit("abc def", "abc")


# --------------------------------------------------------------------------
# Positional reproduction protocol. The three constructions differ in the
# STATISTICAL RELATIONSHIP between length and the target label, not only in
# optimizer settings -- the first Section 1 run changed both at once and the
# positional row came out uninterpretable. These tests pin the relationship,
# the length distribution, the label distribution and the realised budget, so
# a future config edit cannot quietly move one of them again.
#
# Note the import alias: bdsurvive.positional is the LIVE path used by
# plant.py and lineage.py; bdsurvive.triggers.positional (imported above as
# `positional`) is the Stack-B registry copy.
import math

from bdsurvive import positional as live_pos
from bdsurvive.config import PlantConfig


def _ag_like_rows(n=4000, seed=0):
    """Rows with a realistic length spread and a 4-class label cycle."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        words = rng.randint(20, 110)
        rows.append((" ".join(["w"] * words), i % 4))
    return rows


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
    for name, build in (
        ("filler_paper", lambda: live_pos.make_poisoned_train_paper(
            rows, tok, 0, 64, "threshold", 3000, 300, 0)),
        ("natural", lambda: live_pos.make_poisoned_train_natural(
            rows, tok, 0, 64, 600, 60, 0, max_len=96)),
    ):
        texts, labels, _ = build()
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


# --------------------------------------------------------------------------
# Training-exposure accounting. Equal example counts are not equal exposure:
# positional inputs are grown into a length band, so a shard of 4800 carries
# more tokens than a content shard of 4800. Any survival gap between
# mechanisms has that as a rival explanation, and the only defence is to
# measure both budgets and check the conclusion under each.

from bdsurvive.lineage_metrics import sequence_stats, token_count


def test_sequence_stats_excludes_padding():
    tok = StubTokenizer()
    texts = [" ".join(["w"] * n) for n in (10, 20, 30)]
    st = sequence_stats(tok, texts, max_len=128)
    # StubTokenizer adds 2 special tokens, so lengths are 12/22/32 = 66 total
    assert st["total_non_padding_tokens"] == 66
    assert st["mean_sequence_length"] == 22.0
    assert st["padding_excluded"] is True
    # a padded estimate would be 3 * 128 = 384, nearly 6x the real count
    assert st["total_non_padding_tokens"] < 3 * 128 / 4


def test_sequence_stats_matches_token_count():
    """The new stats must not disagree with the metric already in use."""
    tok = StubTokenizer()
    texts = [" ".join(["w"] * n) for n in range(5, 60)]
    assert (sequence_stats(tok, texts, max_len=96)["total_non_padding_tokens"]
            == token_count(tok, texts, max_len=96))


def test_sequence_stats_reports_truncation():
    tok = StubTokenizer()
    texts = [" ".join(["w"] * n) for n in (10, 200, 300)]
    st = sequence_stats(tok, texts, max_len=96)
    assert st["truncated_at_max_len"] == 2, "sequences over max_len must be counted"
    assert st["sequence_length_distribution"]["max"] == 96


def test_equal_examples_are_not_equal_tokens():
    """The premise of the two-comparison design, on the real builders.

    What is asserted is that the two token budgets DIFFER at identical example
    counts, not which way. The direction is corpus-dependent: the positional
    construction pins every input into [tau-margin, tau+margin], so it is
    longer than a corpus whose rows are mostly shorter than tau (real ag_news,
    which averages well under 64 tokens) and SHORTER than a corpus with a long
    tail above tau. Asserting a direction would encode a property of the
    fixture rather than of the code.

    Either way the consequence is the same: matching on examples does not
    match exposure, so a gap between mechanisms needs checking under both
    budgets before it can be called architectural.
    """
    tok, rows = StubTokenizer(), _ag_like_rows(n=3000)
    content = [t for t, _ in rows[:1000]]
    pos_texts, _, _ = live_pos.make_poisoned_train_paper(
        rows, tok, 0, 64, "threshold", 900, 100, 0)
    a = sequence_stats(tok, content, max_len=128)
    b = sequence_stats(tok, pos_texts[:1000], max_len=128)
    assert a["unique_examples"] == b["unique_examples"] == 1000
    ratio = b["total_non_padding_tokens"] / a["total_non_padding_tokens"]
    assert abs(ratio - 1.0) > 0.05, (
        f"token budgets within 5% at matched example counts ({ratio:.3f}x) -- "
        f"if this ever becomes exact, Comparison A and B collapse into one "
        f"and the two-view design is unnecessary")
    # the positional set is tightly banded around tau by construction; that is
    # what makes its token budget predictable and different from the corpus
    assert 55 <= b["mean_sequence_length"] <= 72, b["mean_sequence_length"]


# --------------------------------------------------------------------------
# Extinction is a three-state decision. The previous rule was
# `ci_lo <= epsilon -> extinct`, which never read the upper bound, so an
# interval reaching far above the threshold could be reported extinct. These
# tests pin all three regions and the boundary.

from bdsurvive.eval.extinction import (DEFAULT_PERSISTENCE_THRESHOLD,
                                       classify_survival, resolution_floor)
from bdsurvive.lineage_metrics import survival_state


def test_classify_survival_interval_entirely_below_threshold():
    assert classify_survival(-0.010, 0.015, 0.02) == "extinct"
    assert classify_survival(-0.050, 0.000, 0.02) == "extinct"


def test_classify_survival_interval_entirely_above_threshold():
    assert classify_survival(0.021, 0.090, 0.02) == "persistent"
    assert classify_survival(0.936, 0.963, 0.02) == "persistent"   # a real cell


def test_classify_survival_interval_straddling_threshold():
    # the documented example: excess 0.04, CI [0.01, 0.07], threshold 0.05
    assert classify_survival(0.01, 0.07, 0.05) == "inconclusive"
    # and the case the old rule got dangerously wrong
    assert classify_survival(0.019, 0.70, 0.02) == "inconclusive"


def test_classify_survival_boundary_is_strict():
    """Touching the threshold is not evidence either way."""
    t = 0.02
    assert classify_survival(-0.01, t, t) == "inconclusive"   # ci_hi == threshold
    assert classify_survival(t, 0.50, t) == "inconclusive"    # ci_lo == threshold
    assert classify_survival(t - 1e-9, 0.5, t) == "inconclusive"
    assert classify_survival(t + 1e-9, 0.5, t) == "persistent"


def test_old_two_state_rule_would_have_misreported_a_live_backdoor():
    """Regression guard for the specific defect, stated as a comparison."""
    ci_lo, ci_hi, eps = 0.019, 0.70, 0.02
    old_verdict_extinct = ci_lo <= eps                 # the rule that was there
    assert old_verdict_extinct is True, "fixture no longer triggers the old bug"
    assert classify_survival(ci_lo, ci_hi, eps) == "inconclusive"
    assert not is_extinct_shim(ci_lo, ci_hi, eps), "must not report extinct"


def is_extinct_shim(ci_lo, ci_hi, eps):
    from bdsurvive.eval.extinction import is_extinct
    return is_extinct({"ci_lo": ci_lo, "ci_hi": ci_hi}, eps)


def test_threshold_must_clear_the_resolution_floor():
    """A threshold below the eval's noise floor makes every cell inconclusive.

    The floor is evaluated at the decision boundary, not at worst-case
    variance. At n=1000 per arm those differ a lot -- about 0.017 near a 0.02
    threshold against 0.033 at p=0.5 -- and judging the threshold by the
    worst case would reject a perfectly testable one on variance that does
    not occur where the decision is made.
    """
    at_boundary = resolution_floor(1000, 1000,
                                   threshold=DEFAULT_PERSISTENCE_THRESHOLD)
    worst = resolution_floor(1000, 1000,
                             threshold=DEFAULT_PERSISTENCE_THRESHOLD,
                             p_trigger=0.5)
    assert worst > at_boundary, "worst case must be the wider interval"
    assert DEFAULT_PERSISTENCE_THRESHOLD > at_boundary, (
        f"default threshold {DEFAULT_PERSISTENCE_THRESHOLD} is below the "
        f"decision-boundary floor {at_boundary:.4f}; nothing could ever be "
        f"classified")
    # a mid-range cell is genuinely wider than the threshold, so it reads
    # inconclusive -- correct, and the reason the worst case is reported too
    assert worst > DEFAULT_PERSISTENCE_THRESHOLD
    # an exact-zero margin is untestable at any finite n, which is why the
    # practical-equivalence margin has to be positive
    assert classify_survival(-0.01, 0.01, 0.0) == "inconclusive"


def test_survival_state_reports_its_own_inputs():
    st = survival_state({"ci_lo": 0.936, "ci_hi": 0.963,
                         "n_trigger": 1000, "n_control": 1000})
    assert st["survival_state"] == "persistent"
    assert st["persistence_threshold"] == DEFAULT_PERSISTENCE_THRESHOLD
    assert st["threshold_above_floor"] is True
    assert st["resolution_floor"] > 0


def test_bootstrap_ci_is_quantized_not_broken():
    """Identical CIs across generations are grid quantization, not a bug.

    Observed in the live results: ci_lo/ci_hi were byte-identical at g0 and g1
    even though asr and control_rate both moved. With n=1000 per arm every
    bootstrap replicate is a multiple of 0.001, so the percentile lands on a
    0.001 grid and a sub-grid shift in the distribution does not move it. Two
    runs whose net excess count is identical (973-23 and 971-21 both give
    950/1000) therefore get the same interval. That is expected behaviour, and
    the interval DOES move once the shift is large enough.
    """
    from bdsurvive.eval.extinction import bootstrap_excess
    a = bootstrap_excess([1] * 973 + [0] * 27, [1] * 23 + [0] * 977, n_boot=400)
    b = bootstrap_excess([1] * 971 + [0] * 29, [1] * 21 + [0] * 979, n_boot=400)
    assert abs(a["asr_excess"] - b["asr_excess"]) < 1e-9, "same net excess"
    assert abs(a["ci_lo"] - b["ci_lo"]) < 0.005, "intervals should nearly coincide"
    far = bootstrap_excess([1] * 600 + [0] * 400, [1] * 23 + [0] * 977, n_boot=400)
    assert far["ci_lo"] < a["ci_lo"] - 0.2, "a real shift must move the interval"


# --------------------------------------------------------------------------
# Paired vs independent bootstrap. The two eval arms come from the SAME source
# row in every injected-trigger eval here, so they are dependent and the
# bootstrap must resample examples once and apply the index set to both arms.
# The natural-text length bands are the exception: a real row is either long
# or short, never both, so there the arms are genuinely independent.

from bdsurvive.eval.extinction import (delta_excess_ci, excess_ci,
                                       independent_excess_ci, paired_excess_ci)


def _correlated_arms(n=1000, seed=1):
    """Arms correlated the way real ones are: some examples are hard in BOTH
    forms, so a hit in one arm raises the chance of a hit in the other."""
    rng = random.Random(seed)
    t, c = [], []
    for _ in range(n):
        hard = rng.random() < 0.10          # shared per-example difficulty
        t.append(1 if (hard or rng.random() < 0.90) else 0)
        c.append(1 if (hard and rng.random() < 0.50) else 0)
    return t, c


def test_paired_bootstrap_is_tighter_than_independent():
    """Discarding the pairing inflates the interval -- the old CIs were
    conservative, so no positive finding is overturned by this fix."""
    t, c = _correlated_arms()
    p = paired_excess_ci(t, c, n_boot=2000, seed=7)
    i = independent_excess_ci(t, c, n_boot=2000, seed=7)
    assert abs(p["asr_excess"] - i["asr_excess"]) < 1e-12, "point estimate must agree"
    wp = p["ci_hi"] - p["ci_lo"]
    wi = i["ci_hi"] - i["ci_lo"]
    assert wp < wi, f"paired width {wp:.4f} should be < independent {wi:.4f}"
    assert p["arm_correlation"] > 0, "fixture should have correlated arms"


def test_paired_bootstrap_refuses_misaligned_arms():
    """Unequal lengths mean the arms are not paired; truncating to match would
    silently invent a pairing."""
    try:
        paired_excess_ci([1, 0, 1], [1, 0])
    except ValueError as err:
        assert "paired" in str(err).lower()
    else:
        raise AssertionError("misaligned arms must raise, not truncate")


def test_excess_ci_dispatches_on_the_design():
    t, c = _correlated_arms(n=300)
    assert excess_ci(t, c, paired=True)["bootstrap"] == "paired"
    assert excess_ci(t, c, paired=False)["bootstrap"] == "independent"


def test_natural_eval_declares_itself_unpaired():
    """Change 1 introduced an unpaired design; it must say so, because a
    paired estimator would be wrong on it."""
    tok, rows = StubTokenizer(), _ag_like_rows(n=4000)
    _, _, summary = live_pos.make_natural_length_eval(
        rows, tok, target_label=0, tau=64, n_per_band=50, seed=0)
    assert summary["paired"] is False
    assert "example_ids" in summary and isinstance(summary["example_ids"], dict)
    # ids identify SOURCE rows, and no row may appear in both arms
    overlap = (set(summary["example_ids"]["triggered"])
               & set(summary["example_ids"]["control"]))
    assert not overlap, f"{len(overlap)} rows appear in both arms"


def test_injected_evals_declare_themselves_paired():
    tok, rows = StubTokenizer(), _ag_like_rows(n=2000)
    import bdsurvive.data as DD
    _, _, m = DD.make_trigger_eval(rows, "rare", 0, seed=0, n=200,
                                   with_meta=True)
    assert m["paired"] is True
    assert len(m["example_ids"]) == len(m["true_labels"]) == 200
    assert 0 not in m["true_labels"], "target-label rows must be excluded"
    _, _, mp = live_pos.make_trigger_eval_positional(
        rows, tok, 0, 64, seed=0, n=200, with_meta=True)
    assert mp["paired"] is True


def test_delta_ci_is_paired_across_generations():
    """A CI on the change is the quantity a survival claim rests on, and it is
    not recoverable from two separately-computed intervals."""
    t0, c0 = _correlated_arms(n=1000, seed=3)
    # g1: a small real erosion on the same examples
    rng = random.Random(4)
    t1 = [0 if (h and rng.random() < 0.06) else h for h in t0]
    d = delta_excess_ci(t0, c0, t1, c0, n_boot=2000)
    assert d["delta_excess"] < 0, "erosion should be negative"
    assert d["delta_ci_lo"] < d["delta_excess"] < d["delta_ci_hi"]
    assert d["n_pairs"] == 1000
    # no change at all must NOT exclude zero
    same = delta_excess_ci(t0, c0, t0, c0, n_boot=500)
    assert same["delta_excess"] == 0.0
    assert not same["delta_excludes_zero"]


def test_delta_ci_detects_a_change_two_overlapping_intervals_would_not():
    """The reason to compute a delta rather than eyeball two intervals."""
    t0, c0 = _correlated_arms(n=1000, seed=11)
    rng = random.Random(12)
    t1 = [0 if (h and rng.random() < 0.05) else h for h in t0]
    a = paired_excess_ci(t0, c0, n_boot=2000)
    b = paired_excess_ci(t1, c0, n_boot=2000)
    overlapping = not (a["ci_hi"] < b["ci_lo"] or b["ci_hi"] < a["ci_lo"])
    d = delta_excess_ci(t0, c0, t1, c0, n_boot=2000)
    assert overlapping, "fixture should produce overlapping per-gen intervals"
    assert d["delta_excludes_zero"], (
        "the paired delta should resolve a change that overlapping "
        "per-generation intervals cannot")


TESTS = [value for key, value in sorted(globals().items()) if key.startswith("test_")]

if __name__ == "__main__":
    failures = 0
    for test in TESTS:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except Exception as error:                      # noqa: BLE001
            failures += 1
            print(f"FAIL  {test.__name__}: {error}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    raise SystemExit(1 if failures else 0)
