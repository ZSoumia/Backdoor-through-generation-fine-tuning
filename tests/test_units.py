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

    vocab_size = 1024

    def __len__(self):
        return self.vocab_size


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


def test_positional_on_learned_positions_is_mechanism_not_applicable():
    """SEMANTICS CHANGED, deliberately.

    This previously asserted GridStatus.NOT_APPLICABLE, which conflated "the
    RoPE mechanism cannot exist here" with "nothing can be measured here".
    Only the first is true: a learned-absolute model exposes position through
    a trainable table, so a length-conditioned behaviour is plantable and its
    persistence measurable -- by a different circuit, one that happens to sit
    outside LoRA's and BitFit's support.

    The cell therefore RUNS and is reported as a length-conditioned result,
    never as the RoPE mechanism and never as N/A.
    """
    cfg = ExperimentConfig(
        model=ModelSpec(name="bert-base-uncased", position_encoding="learned"),
        attack=AttackSpec(trigger=__import__("bdsurvive.core.config", fromlist=["TriggerSpec"]).TriggerSpec(name="positional")))
    result = validate(cfg)
    assert result.status is GridStatus.MECHANISM_NOT_APPLICABLE
    assert result.status is not GridStatus.NOT_APPLICABLE
    assert result.should_run is True
    assert result.is_valid is False
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


# --------------------------------------------------------------------------
# Data separation, enforced in the executing path. The helpers were already
# consistent with each other and tested against each other, while plant()
# bypassed both and drew from the whole corpus -- a green test over an
# uncovered path, which is how a run named 'seeds_v3_disjoint' trained on
# overlapping data. These tests check the real allocator and the assertion
# the runtime calls, including a deliberately poisoned fixture.

from bdsurvive.shards import (DEFAULT_PLANTING_BLOCK, PARTITION_SEED,
                              assert_disjoint, partition_corpus,
                              planting_rows, reserve_planting_block)


def test_partition_boundary_does_not_move_with_the_lineage_seed():
    """A parent is planted once and derived under seeds 0,1,2. If the split
    moved with the lineage seed it could be disjoint from at most one."""
    rows = [("t", 0)] * 50000
    a = make_shards(rows, 3, 4800, 10000, seed=0)
    b = make_shards(rows, 3, 4800, 10000, seed=1)
    assert set(a["planting_idx"]) == set(b["planting_idx"]), (
        "planting block moved when the lineage seed changed")
    # the shards themselves must differ, or varying the seed does nothing
    assert set(a["shards"][0]) != set(b["shards"][0])
    # and every shard, under either seed, avoids the planting block
    for alloc in (a, b):
        for s in alloc["shards"]:
            assert not set(s) & set(alloc["planting_idx"])


def test_planting_rows_match_the_allocator_reservation():
    """Two independent derivations of the same boundary must agree."""
    rows = [(f"t{i}", i % 4) for i in range(50000)]
    _, ids = planting_rows(rows, 10000)
    alloc = make_shards(rows, 2, 4800, 10000, seed=3)
    assert set(ids) == set(alloc["planting_idx"])
    assert set(ids) == set(reserve_planting_block(rows, 10000))


def test_assert_disjoint_rejects_an_injected_overlap():
    """Deliberately poisoned fixture: the check must reject it."""
    plant_ids = [f"train:{i}" for i in range(100)]
    clean = [[f"train:{i}" for i in range(100, 150)],
             [f"train:{i}" for i in range(150, 200)]]
    assert assert_disjoint(plant_ids, clean, {"test:1"})["disjoint"] is True

    # a planting row smuggled into G2
    poisoned = [clean[0], clean[1] + ["train:7"]]
    try:
        assert_disjoint(plant_ids, poisoned, {"test:1"})
    except AssertionError as err:
        assert "overlap at G2" in str(err), str(err)
    else:
        raise AssertionError("injected planting/training overlap not rejected")


def test_assert_disjoint_rejects_cross_generation_reuse():
    plant_ids = [f"train:{i}" for i in range(10)]
    shards = [[f"train:{i}" for i in range(10, 20)],
              [f"train:{i}" for i in range(15, 25)]]     # 15-19 reused
    try:
        assert_disjoint(plant_ids, shards)
    except AssertionError as err:
        assert "overlap at G2" in str(err)
    else:
        raise AssertionError("cross-generation reuse not rejected")


def test_assert_disjoint_rejects_duplicates_within_one_shard():
    """A repeated id breaks coverage-1 even with no cross-partition overlap."""
    try:
        assert_disjoint([], [["train:1", "train:2", "train:1"]])
    except AssertionError as err:
        assert "duplicate" in str(err).lower() and "G1" in str(err)
    else:
        raise AssertionError("within-shard duplicate not rejected")


def test_assert_disjoint_rejects_eval_leakage():
    try:
        assert_disjoint(["train:1"], [["train:2"]], {"train:2"})
    except AssertionError as err:
        assert "leakage at G1" in str(err)
    else:
        raise AssertionError("eval leakage not rejected")
    # planting overlapping eval is caught before any generation is examined
    try:
        assert_disjoint(["train:1"], [["train:2"]], {"train:1"})
    except AssertionError as err:
        assert "evaluation leakage" in str(err)
    else:
        raise AssertionError("planting/eval leakage not rejected")


def test_split_namespacing_prevents_false_positives():
    """Row 41 of train and row 41 of test are different examples. Without the
    namespace the eval check would fire on coincidental position matches."""
    assert assert_disjoint(["train:41"], [["train:42"]],
                           {"test:41", "test:42"})["disjoint"] is True


def test_planting_block_is_hashed_so_the_parent_id_changes():
    """The fix changes what plant() trains on, so it MUST change the id. An
    exemption here would let two different parents share an identity."""
    a = PlantConfig(planting_block=30000)
    b = PlantConfig(planting_block=3300)
    assert a.id != b.id, "planting_block must participate in the parent id"


def test_planting_block_leaves_room_for_the_largest_planting_set():
    """budget_controlled positional needs n_clean + n_poison = 8000 rows."""
    assert DEFAULT_PLANTING_BLOCK >= 8000
    rows = [("t", 0)] * 120000
    plant_ids, deriv = partition_corpus(len(rows), DEFAULT_PLANTING_BLOCK,
                                        PARTITION_SEED)
    assert len(plant_ids) == DEFAULT_PLANTING_BLOCK
    assert len(deriv) >= 9 * 4800, "derivation pool too small for 9 generations"
    assert not set(plant_ids) & set(deriv)


# --------------------------------------------------------------------------
# Lexical trigger portability. " cf" comes from RIPPLe's BERT/GPT-2 vocabulary
# and is not guaranteed to be one rare piece in SmolLM2's 49k BPE, BLOOM's
# 251k, or Qwen's 152k. If it fragments or is common, the lexical row compares
# different attacks under one label.


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
    import bdsurvive.data as DD
    a = DD.rare_token_audit(FragmentingTokenizer(), rows=None)
    assert a["n_tokens"] == 2
    assert a["single_token"] is False
    assert "not one" in a["problem"]
    assert a["usable_as_single_rare_token"] is False


def test_audit_accepts_a_single_rare_trigger():
    import bdsurvive.data as DD
    rows = [("markets tumbled after the announcement", 1)] * 400
    a = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    assert a["single_token"] is True
    assert a["frequency"]["token_occurrences"] == 0, "trigger absent from corpus"
    assert a["rare_enough"] is True
    assert a["usable_as_single_rare_token"] is True
    assert "problem" not in a


def test_audit_flags_a_common_trigger_on_empirical_frequency():
    """Rarity is measured in the corpus, not inferred from the token id."""
    import bdsurvive.data as DD
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
    import bdsurvive.data as DD

    class HighId(SingleTokenTokenizer):
        def _one(self, s):
            return [49000 if w == "cf" else hash(w) % 400 + 10 for w in s.split()]

    rows = [("markets tumbled after the announcement", 1)] * 300
    lo = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    hi = DD.rare_token_audit(HighId(), rows=rows)
    assert lo["ids"] != hi["ids"], "fixture should differ in token id"
    assert lo["rare_enough"] == hi["rare_enough"] is True
    assert (lo["frequency"]["occurrences_per_million_tokens"]
            == hi["frequency"]["occurrences_per_million_tokens"])


def test_audit_reports_the_three_required_measurements():
    import bdsurvive.data as DD
    rows = [("markets tumbled after the announcement", 1)] * 300
    a = DD.rare_token_audit(SingleTokenTokenizer(), rows=rows)
    assert a["n_tokens"] is not None                       # 1. token count
    assert a["ids"] and a["pieces"] is not None            # 2. ids + pieces
    assert a["frequency"]["occurrences_per_million_tokens"] is not None  # 3.
    assert a["frequency"]["measured_on"].startswith("the task corpus")


# --------------------------------------------------------------------------
# Control conditions. control_rate alone compares a token insertion against a
# template insertion, which shows the two transformations differ in effect but
# leaves the token's identity confounded with the fact that anything was
# inserted. The matched perturbation holds the transformation shape fixed and
# varies only the payload, which is what isolates the token.


def test_matched_perturbation_does_not_fire_the_trigger():
    """The control must not accidentally satisfy the predicate it controls
    for -- that would deflate measured specificity silently."""
    import bdsurvive.data as DD
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
    import bdsurvive.data as DD
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
    import bdsurvive.data as DD
    try:
        DD.perturb_matched("some text", "positional", random.Random(0))
    except ValueError as err:
        assert "no content payload" in str(err)
    else:
        raise AssertionError("positional must not claim a payload swap")


def test_condition_eval_aligns_every_condition_on_one_population():
    import bdsurvive.data as DD
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


def test_worst_case_excess_is_the_binding_constraint():
    """A specificity claim has to survive every control, not the kindest one."""
    asr = 0.95
    rates = {"original": 0.03, "alt_trigger": 0.05,
             "matched_perturbation": 0.80}
    excesses = {k: round(asr - v, 4) for k, v in rates.items()}
    worst = min(excesses.values())
    assert worst == excesses["matched_perturbation"] == 0.15
    # against the alternative trigger alone the effect looks near-total; the
    # matched perturbation shows most of it is the transformation, not the token
    assert excesses["alt_trigger"] - worst > 0.7


# --------------------------------------------------------------------------
# Failure-set analysis. An aggregate ASR of 0.954-0.969 across seven models is
# compatible with the same examples failing everywhere (an eval-set property)
# or with each model missing its own (a model property). Only the identities
# separate them.

from bdsurvive.lineage_metrics import (failure_overlap_matrix,
                                       generation_failure_stability, jaccard,
                                       stratify_failures)


def _rec(missed, hit, generation=0):
    return {"available": True, "generation": generation,
            "missed_example_ids": list(missed),
            "correctly_triggered_example_ids": list(hit),
            "n": len(missed) + len(hit), "n_missed": len(missed),
            "rate": len(hit) / max(1, len(missed) + len(hit))}


def test_jaccard_endpoints():
    assert jaccard([1, 2, 3], [1, 2, 3]) == 1.0
    assert jaccard([1, 2], [3, 4]) == 0.0
    assert jaccard([], []) == 1.0, "two empty failure sets are identical"
    assert jaccard([1, 2, 3, 4], [3, 4, 5, 6]) == 2 / 6


def test_overlap_detects_a_shared_difficult_subset():
    """Same examples fail in every model -> eval set or trigger, not model."""
    shared = [f"pos_{i:06d}" for i in range(40)]
    runs = {f"model{k}": _rec(shared, [f"pos_{i:06d}" for i in range(40, 1000)])
            for k in range(7)}
    ov = failure_overlap_matrix(runs)
    assert ov["mean_pairwise_jaccard"] == 1.0
    assert ov["n_missed_by_every_run"] == 40
    assert ov["n_missed_by_exactly_one"] == 0
    assert "SHARED" in ov["verdict"]


def test_overlap_detects_coincidental_equal_rates():
    """Same RATE, disjoint failures -> the matching ceiling is a coincidence."""
    runs = {}
    for k in range(7):
        missed = [f"pos_{i:06d}" for i in range(k * 40, k * 40 + 40)]
        runs[f"model{k}"] = _rec(missed, [f"pos_{i:06d}" for i in range(900, 1000)])
    rates = {r["n_missed"] for r in runs.values()}
    assert rates == {40}, "fixture must hold the aggregate rate constant"
    ov = failure_overlap_matrix(runs)
    assert ov["mean_pairwise_jaccard"] == 0.0
    assert ov["n_missed_by_every_run"] == 0
    assert ov["n_missed_by_exactly_one"] == 280
    assert "MODEL-DEPENDENT" in ov["verdict"]


def test_overlap_reports_a_mixed_core():
    core = [f"pos_{i:06d}" for i in range(20)]
    runs = {}
    for k in range(4):
        own = [f"pos_{i:06d}" for i in range(100 + k * 20, 120 + k * 20)]
        runs[f"model{k}"] = _rec(core + own, [f"pos_{i:06d}" for i in range(500, 900)])
    ov = failure_overlap_matrix(runs)
    assert 0.2 < ov["mean_pairwise_jaccard"] < 0.5
    assert ov["n_missed_by_every_run"] == 20
    assert "MIXED" in ov["verdict"]


def test_generation_stability_separates_stable_from_churning():
    """A flat ASR can hide a turning-over failure set."""
    stable = [_rec(range(40), range(40, 1000), generation=g) for g in (0, 1, 2)]
    st = generation_failure_stability(stable)
    assert st["mean_jaccard"] == 1.0 and "STABLE" in st["verdict"]

    churn = [_rec(range(g * 40, g * 40 + 40), range(900, 1000), generation=g)
             for g in (0, 1, 2)]
    sizes = {r["n_missed"] for r in churn}
    assert sizes == {40}, "aggregate rate must be flat in the fixture"
    sc = generation_failure_stability(churn)
    assert sc["mean_jaccard"] == 0.0
    assert "CHURNING" in sc["verdict"]
    assert sc["steps"][0]["recovered"] == 40
    assert sc["steps"][0]["newly_missed"] == 40


def test_stratify_finds_a_concentrated_ceiling():
    """All failures in one bucket -> the ceiling is an eval-construction
    property, not a uniform model limit."""
    missed = [f"pos_{i:06d}" for i in range(50)]
    hit = [f"pos_{i:06d}" for i in range(50, 450)]
    assignment = {e: ("short" if int(e.split("_")[1]) < 50 else "long")
                  for e in missed + hit}
    st = stratify_failures(_rec(missed, hit), {"length": assignment})
    b = st["length"]["buckets"]
    assert b["short"]["hit_rate"] == 0.0
    assert b["long"]["hit_rate"] == 1.0
    assert st["length"]["spread"] == 1.0


def test_stratify_ignores_tiny_buckets_in_the_spread():
    """A 3-example bucket at 0.0 must not drive the spread."""
    missed = [f"pos_{i:06d}" for i in range(3)]
    hit = [f"pos_{i:06d}" for i in range(3, 203)]
    assignment = {e: ("tiny" if int(e.split("_")[1]) < 3 else "big")
                  for e in missed + hit}
    st = stratify_failures(_rec(missed, hit), {"length": assignment})
    assert st["length"]["buckets"]["tiny"]["n"] == 3
    assert st["length"]["spread"] is None, "one qualifying bucket -> no spread"


# --------------------------------------------------------------------------
# Seeds. The planting seed and the derivation seed govern different things and
# must stay separable: one measures variability in how the backdoor was
# installed, the other in how it eroded, and the survival claim is about the
# second. Three sources were previously uncontrolled -- LoRA init, planting
# presentation order, and the continuous control's batch order -- so seed
# replicates would have differed for reasons nobody was tracking.


def test_plant_seed_changes_the_parent_but_derivation_seed_does_not():
    """A planting seed is a different parent; a derivation seed is a different
    run over the SAME parent. Conflating them would either replant
    unnecessarily or reuse a parent that should have changed."""
    a = PlantConfig(seed=0)
    b = PlantConfig(seed=1)
    assert a.id != b.id, "planting seed must produce its own parent"
    # the derivation seed is not a PlantConfig field at all, which is what
    # keeps one parent valid across its derivation replicates
    assert "seed" in {f for f in a.__dataclass_fields__}
    assert "derivation_seed" not in a.__dataclass_fields__


def test_partition_seed_is_not_swept_with_the_others():
    """The planting/derivation boundary must not move between replicates, or
    a parent is disjoint from at most one seed's derivation data."""
    rows = [("t", 0)] * 50000
    for dseed in (0, 1, 2):
        alloc = make_shards(rows, 2, 4800, 10000, seed=dseed)
        assert alloc["partition_seed"] == PARTITION_SEED
    boundaries = {tuple(sorted(make_shards(rows, 2, 4800, 10000,
                                           seed=d)["planting_idx"])[:20])
                  for d in (0, 1, 2)}
    assert len(boundaries) == 1, "boundary moved across derivation seeds"


def test_seeded_dataloader_order_is_reproducible():
    """The pattern plant() now uses. shuffle=True with no generator draws from
    the global RNG, so two parents with the same cfg.seed differed in
    presentation order for reasons nothing recorded."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(torch.arange(64))

    def order(seed):
        dl = DataLoader(ds, batch_size=8, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
        return [int(x) for b in dl for x in b[0]]

    assert order(0) == order(0), "same seed must give the same order"
    assert order(0) != order(1), "different seeds must give different orders"


def test_lora_init_is_reproducible_under_a_pinned_seed():
    """LoRA's lora_A kaiming init comes from the GLOBAL torch RNG, so without
    pinning it two runs at the same derivation seed got different adapters."""
    import torch

    def init(seed):
        torch.manual_seed(seed)
        return torch.nn.Linear(16, 4).weight.detach().clone()

    assert torch.equal(init(7), init(7)), "pinned seed must reproduce init"
    assert not torch.equal(init(7), init(8)), "different seed must differ"
    # unpinned draws differ, which is the behaviour that was in effect
    torch.manual_seed(0)
    a = torch.nn.Linear(16, 4).weight.detach().clone()
    b = torch.nn.Linear(16, 4).weight.detach().clone()
    assert not torch.equal(a, b), "consecutive unpinned draws differ"


def test_generation_derived_seeds_do_not_collide():
    """seed * 100_000 + g must not let one (seed, generation) pair alias
    another, or generation 2 of seed 0 reuses generation 1 of some other."""
    seen = {}
    for seed in range(8):
        for g in range(1, 20):
            key = seed * 100_000 + g
            assert key not in seen, f"collision: {(seed, g)} vs {seen[key]}"
            seen[key] = (seed, g)


def test_variance_decomposition_separates_the_two_sources():
    """Pooled sd cannot say which source dominates; the split can.

    Two 3x3 designs with similar pooled spread and opposite structure.
    """
    import math

    def sd(xs):
        if len(xs) < 2:
            return 0.0
        m = sum(xs) / len(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

    # planting dominates: parents far apart, derivation tight
    plant_dom = {p: [0.95 - p * 0.12 + d * 0.004 for d in range(3)]
                 for p in range(3)}
    # derivation dominates: parents together, derivation scattered
    deriv_dom = {p: [0.94 + (d - 1) * 0.09 + p * 0.003 for d in range(3)]
                 for p in range(3)}

    for name, design, expect in (("plant", plant_dom, "planting"),
                                 ("deriv", deriv_dom, "derivation")):
        within = sum(sd(v) for v in design.values()) / len(design)
        between = sd([sum(v) / len(v) for v in design.values()])
        dominant = "planting" if between > within else "derivation"
        assert dominant == expect, f"{name}: got {dominant}"

    pooled_a = sd([x for v in plant_dom.values() for x in v])
    pooled_b = sd([x for v in deriv_dom.values() for x in v])
    assert abs(pooled_a - pooled_b) < 0.05, (
        "fixtures should have comparable POOLED spread, which is the point: "
        "pooling cannot distinguish the two structures")


def test_sd_over_two_seeds_is_not_reported_as_an_estimate():
    """With n<3 the standard deviation is not an estimate of anything, so the
    trajectories carry the claim instead."""
    import math

    def stats(xs):
        if len(xs) < 2:
            return {"n": len(xs), "sd": None}
        m = sum(xs) / len(xs)
        return {"n": len(xs),
                "sd": math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))}

    assert stats([0.95])["sd"] is None
    assert stats([0.95, 0.94])["sd"] is not None   # computable
    assert stats([0.95, 0.94, 0.96])["n"] == 3     # but only meaningful at 3+


# --------------------------------------------------------------------------
# Graded positional applicability. "Not RoPE" was collapsed straight into
# NOT_APPLICABLE, which reported the length-conditioned question as settled
# when it had never been asked. The two dimensions are now separate.

from bdsurvive.core.validation import GridStatus, positional_verdict
from bdsurvive.models import POSITIONAL_APPLICABILITY, positional_applicability


def test_rope_is_the_only_scheme_where_the_mechanism_is_applicable():
    assert positional_applicability("rope")["rope_mechanism"] == "applicable"
    for scheme in ("learned", "alibi", "relative"):
        assert positional_applicability(scheme)["rope_mechanism"] == \
            "not_applicable", scheme


def test_every_scheme_can_still_represent_length_dependence():
    """This is the claim the blanket N/A was quietly denying."""
    for scheme in POSITIONAL_APPLICABILITY:
        info = positional_applicability(scheme)
        assert info["exposes_position_information"] is True, scheme
        assert info["can_represent_length_dependence"] is True, scheme
        assert info["length_conditioned"] == "measurable", scheme


def test_learned_absolute_gets_measured_not_dismissed():
    v = positional_verdict("learned")
    assert v.status is GridStatus.MECHANISM_NOT_APPLICABLE
    assert v.should_run is True, "the cell must run; it yields a measurement"
    assert v.is_valid is False, "but it is NOT the RoPE mechanism"
    assert "RoPE mechanism N/A" in v.reporting_label
    assert "length-conditioned behaviour is measurable" in v.reason


def test_learned_absolute_support_is_outside_lora_and_bitfit():
    """The reason this cell is the most informative, not the least: its
    support is trainable but unreachable by two of the three FT methods."""
    info = positional_applicability("learned")
    assert info["in_lora_support"] is False
    assert info["in_bitfit_support"] is False
    assert info["in_full_support"] is True
    assert "position-embedding table" in info["support_lives_in"]
    # RoPE's support, by contrast, IS inside LoRA's attention reach
    rope = positional_applicability("rope")
    assert rope["in_lora_support"] is True


def test_rope_verdict_is_plain_valid():
    v = positional_verdict("rope")
    assert v.status is GridStatus.VALID
    assert v.should_run is True and v.is_valid is True
    assert v.reporting_label == "measured"


def test_unknown_scheme_is_untested_not_not_applicable():
    """Absent from the grid is not the same as shown impossible."""
    v = positional_verdict("sinusoidal_something")
    assert v.status is GridStatus.UNTESTED
    assert v.should_run is False, "do not run what we cannot interpret"
    assert v.reporting_label == "untested"
    assert "add it to POSITIONAL_APPLICABILITY" in v.reason


def test_the_three_report_labels_are_distinct():
    """A results table must not print the same string for a measured cell, an
    unmeasured one, and an impossible one."""
    labels = {positional_verdict(s).reporting_label
              for s in ("rope", "learned", "nonsense_scheme")}
    assert len(labels) == 3, labels


def test_learned_absolute_entries_now_exist_in_the_grid():
    import yaml

    from bdsurvive.config import PlantConfig
    raw = yaml.safe_load(open("configs/mechanisms.yaml"))
    learned_models = {"EleutherAI/gpt-neo-1.3B", "facebook/opt-1.3b"}
    positional = {n: PlantConfig(**e["plant"]) for n, e in raw.items()
                  if PlantConfig(**e["plant"]).support == "positional"}
    covered = {c.base_model for c in positional.values()}
    assert learned_models <= covered, (
        f"learned-absolute models missing from the positional grid: "
        f"{learned_models - covered}")
    # and they use the natural construction, which is the architecture-
    # independent test: no filler artifacts to confound the comparison
    nat = {n for n, c in positional.items()
           if c.base_model in learned_models
           and c.resolved_construction() == "natural"}
    assert len(nat) == 2, nat


# --------------------------------------------------------------------------
# Causal attribution. A length-conditioned effect can come from the position
# information the architecture exposes OR from the text added to make the
# input longer -- the filler, the token count, the altered word distribution.
# The construction changes both at once, so ASR cannot separate them. These
# test the attribution logic that does.


def _probes(rates, unsupported=(), errored=()):
    """Probe records in the shape causal_position_probes emits."""
    from bdsurvive.metrics import NO_POSITION_IDS_REASON, _record
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
    from bdsurvive.metrics import attribute_length_effect
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
    from bdsurvive.models import positional_applicability
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


# --- the three-level claim ladder. Output-level probes can reach the first
# two rungs and never the third, and the gate between the first two is
# selectivity: an intervention that also breaks ordinary classification has
# left the position distribution the model was trained on, so the prediction
# shift is not evidence about the learned behaviour.


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
    from bdsurvive.metrics import attribute_length_effect
    v = attribute_length_effect(
        _probes({"baseline_long": 0.95, "baseline_short": 0.03,
                 "position_stride": 0.90, "position_offset": 0.85,
                 "masked_padding": 0.04, "content_shuffled": 0.93}),
        selectivity=None)
    assert v["claim_level"] == "positional_sensitivity"
    assert "untested" in v["attribution"]


def test_mechanistic_attribution_is_never_claimed_from_output_probes():
    """The third rung needs internal interventions. No branch may reach it."""
    from bdsurvive.metrics import CLAIM_LEVELS
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
    from bdsurvive.metrics import attribute_length_effect
    recs = _probes({"baseline_short": 0.03, "content_shuffled": 0.9},
                   errored=("baseline_long",))
    v = attribute_length_effect(recs)
    assert v["claim_level"] == "unresolved"
    assert v["baseline_effect"] is None


def test_bloom_reason_names_the_unsupported_interface_not_the_architecture():
    """Wording matters: the claim is that THIS probe's interface cannot carry
    the intervention, not that BLOOM lacks position sensitivity."""
    from bdsurvive.metrics import NO_POSITION_IDS_REASON as why
    assert why.startswith("unsupported:")
    for required in ("standard", "Hugging Face interface",
                     "build_alibi_tensor", "ALiBi bias",
                     "NOT evidence that position had no effect"):
        assert required in why, required


def test_probe_status_vocabulary_is_closed():
    """A typo in a status would make a failure unfindable in the records."""
    from bdsurvive.metrics import PROBE_STATUS, _record
    assert set(PROBE_STATUS) == {"success", "unsupported", "error",
                                 "not_run", "invalid"}
    assert _record("unsupported", reason="x")["rate"] is None
    try:
        _record("unavailable")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown probe status must not be accepted")


# --- cross-generation comparability. An apparent change in attribution from
# G0 to G3 is only interpretable if the measurement conditions were identical,
# so they are fingerprinted and compared rather than assumed.


def _population(ids=("test:1", "test:2"), short=("a", "b"), long=("aa", "bb")):
    from bdsurvive.metrics import PROBE_DEFINITION_VERSION
    return {"definition_version": PROBE_DEFINITION_VERSION, "tau": 64,
            "n": len(ids), "example_ids": list(ids), "labels": [0, 2],
            "short": list(short), "long": list(long),
            "tokenizer": {"name_or_path": "stub", "revision": "abc"}}


def test_conditions_fingerprint_is_stable_for_identical_conditions():
    from bdsurvive.metrics import conditions_fingerprint as fp
    assert fp(_population(), 96, 2) == fp(_population(), 96, 2)


def test_conditions_fingerprint_catches_every_way_conditions_can_drift():
    from bdsurvive.metrics import conditions_fingerprint as fp
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
    from bdsurvive.metrics import build_probe_population
    tok = StubTokenizer()
    rows = [(f"row {i} " + "word " * 40, i % 4) for i in range(40)]
    pop = build_probe_population(tok, rows, target_label=1, tau=20, n=8)
    assert pop["n"] == len(pop["short"]) == len(pop["long"]) <= 8
    assert len(pop["example_ids"]) == pop["n"] == len(pop["labels"])
    assert all(i.startswith("test:") for i in pop["example_ids"])
    assert 1 not in pop["labels"], "target-label rows must be excluded"
    assert pop["tokenizer"]["class"] == "StubTokenizer"


def test_every_generation_gets_a_probe_record_even_when_it_cannot_run():
    """A missing file is indistinguishable from a measured null, so the writer
    persists not_run and error states instead of only printing them."""
    import json as _json
    import os as _os
    import tempfile
    import types

    from bdsurvive import lineage as LIN

    cfg = types.SimpleNamespace(support="lexical", task="ag_news",
                                target_label=1, trigger_tau=64)
    with tempfile.TemporaryDirectory() as d:
        rec = LIN._write_position_probes(None, None, cfg, "cpu", 96, d, 2)
        on_disk = _json.load(open(_os.path.join(d, "position_probes_g2.json")))
        assert rec["probe_status"] == on_disk["probe_status"] == "not_run"
        assert on_disk["generation"] == 2
        assert on_disk["positional_attribution"] is None

        cfg.support = "positional"
        original = LIN.D.load_task

        def _boom(*a, **k):
            raise RuntimeError("CUDA out of memory")

        LIN.D.load_task = _boom
        try:
            rec = LIN._write_position_probes(None, None, cfg, "cpu", 96, d, 3)
        finally:
            LIN.D.load_task = original
        on_disk = _json.load(open(_os.path.join(d, "position_probes_g3.json")))
        assert on_disk["probe_status"] == "error"
        assert on_disk["error_type"] == "RuntimeError"
        assert on_disk["positional_attribution"] is None
        assert "NOT a measured absence of effect" in on_disk["reason"]
        assert on_disk["traceback"]


def test_alibi_takes_no_position_ids_so_probes_must_report_unavailable():
    """Guard on the real dispatch. BLOOM's absence is ARCHITECTURAL -- ALiBi
    builds its bias from the attention mask -- so it is asserted at every
    version. OPT's is version-dependent: its forward gained position_ids well
    after 4.38, the version pinned on the cluster, so asserting True there
    would fail on the machine this actually runs on. What must hold everywhere
    is that the dispatch agrees with the signature."""
    import inspect

    import transformers
    from bdsurvive.metrics import _accepts_position_ids
    for cls_name, expect in (("BloomForSequenceClassification", False),
                             ("GPTNeoXForSequenceClassification", True),
                             ("GPT2ForSequenceClassification", True),
                             ("OPTForSequenceClassification", None),
                             ("GPTNeoForSequenceClassification", None)):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        present = "position_ids" in inspect.signature(cls.forward).parameters
        if expect is not None:
            assert present is expect, (
                f"{cls_name}: position_ids present={present}, "
                f"expected {expect} at every transformers version")
    for family in ("bloom", "opt", "gpt_neox"):
        model = _tiny(family)
        present = ("position_ids" in
                   inspect.signature(model.forward).parameters)
        assert _accepts_position_ids(model) is present, family


def test_position_intervention_indices_stay_inside_the_sequence():
    """The bug that killed a real run, as a unit test.

    transformers 4.38 returns cos_cached[:seq_len] from the rotary embedding
    and then gathers cos[position_ids], so the gather axis is the INPUT
    length. A strided position id past that length is an out-of-bounds gather:
    IndexError on CPU, device-side assert on CUDA. The fix pads the input so
    seq_len covers the index range, and the invariant is checked in Python
    before any kernel launches -- a device assert cannot be caught.
    """
    import torch

    from bdsurvive.metrics import _position_batch

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
    import torch

    from bdsurvive.metrics import _position_batch, _predict_tensors

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
    from bdsurvive.metrics import (NO_POSITION_IDS_REASON,
                                   no_position_ids_reason)
    bloom = no_position_ids_reason(_tiny("bloom"))
    assert bloom == NO_POSITION_IDS_REASON
    assert "build_alibi_tensor" in bloom
    other = no_position_ids_reason(_tiny("opt"))
    assert "LIBRARY limitation" in other
    assert "build_alibi_tensor" not in other
    assert "transformers" in other
    for reason in (bloom, other):
        assert "NOT evidence that position had no effect" in reason


# --------------------------------------------------------------------------
# Trainable-parameter support. A method NAME is not a support: "LoRA" updates
# k on a fused-QKV family and excludes it on a split-QKV family, and BitFit
# ranges from 25 bias tensors to none at all. Equal module counts (48 on every
# 24-layer model) actively hide this.


def _tiny(family, num_labels=4):
    """Smallest real model of each family, built from its config class so no
    download is needed. Parameter NAMES are what the audit reads, and those are
    identical at any width."""
    import transformers as T
    common = dict(num_labels=num_labels)
    if family == "gpt_neox":
        return T.GPTNeoXForSequenceClassification(T.GPTNeoXConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, **common))
    if family == "bloom":
        return T.BloomForSequenceClassification(T.BloomConfig(
            vocab_size=64, hidden_size=32, n_layer=2, n_head=4, **common))
    if family == "gpt2":
        return T.GPT2ForSequenceClassification(T.GPT2Config(
            vocab_size=64, n_embd=32, n_layer=2, n_head=4, **common))
    if family == "qwen2":
        return T.Qwen2ForSequenceClassification(T.Qwen2Config(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4,
            intermediate_size=64, **common))
    if family == "llama":
        return T.LlamaForSequenceClassification(T.LlamaConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=4,
            intermediate_size=64, **common))
    if family == "gpt_neo":
        return T.GPTNeoForSequenceClassification(T.GPTNeoConfig(
            vocab_size=64, hidden_size=32, num_layers=2, num_heads=4,
            attention_types=[[["global", "local"], 1]], **common))
    if family == "opt":
        return T.OPTForSequenceClassification(T.OPTConfig(
            vocab_size=64, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
            num_attention_heads=4, word_embed_proj_dim=32, **common))
    raise ValueError(family)


# Which of the seven study architectures fuse QKV. This is the fact that makes
# "LoRA on 48 modules" two different transformations.
FUSED_FAMILIES = ("gpt_neox", "bloom", "gpt2")
SPLIT_FAMILIES = ("qwen2", "llama", "gpt_neo", "opt")


def test_parameter_roles_are_read_from_the_path_not_the_leaf():
    """GPT-2 names the attention output and the MLP output both 'c_proj';
    BERT names three projections 'dense'. Suffix matching cannot tell them
    apart, so the classifier is path-based."""
    from bdsurvive.models import classify_parameter as role
    cases = {
        "transformer.h.0.attn.c_proj.weight": "attention_out",
        "transformer.h.0.mlp.c_proj.weight": "mlp",
        "transformer.h.0.attn.c_attn.weight": "attention_qkv_fused",
        "gpt_neox.layers.3.attention.query_key_value.weight":
            "attention_qkv_fused",
        "gpt_neox.layers.3.attention.dense.bias": "attention_out",
        "model.layers.5.self_attn.q_proj.weight": "attention_q",
        "model.layers.5.self_attn.k_proj.weight": "attention_k",
        "model.layers.5.self_attn.v_proj.weight": "attention_v",
        "model.layers.5.self_attn.o_proj.weight": "attention_out",
        "model.layers.5.mlp.gate_proj.weight": "mlp",
        "model.decoder.embed_positions.weight": "embedding_position",
        "transformer.wpe.weight": "embedding_position",
        "transformer.wte.weight": "embedding_token",
        "gpt_neox.embed_in.weight": "embedding_token",
        "transformer.h.0.input_layernorm.weight": "normalization",
        "model.layers.0.post_attention_layernorm.weight": "normalization",
        "gpt_neox.final_layer_norm.bias": "normalization",
        "score.weight": "head",
        "classifier.bias": "head",
        # BLOOM's embedding LAYERNORM contains 'word_embeddings' -- the norm
        # test has to come first or an embedding read is reported for it.
        "transformer.word_embeddings_layernorm.weight": "normalization",
        "transformer.word_embeddings.weight": "embedding_token",
        # PEFT bookkeeping is stripped, so a LoRA parameter is attributed to
        # the module it adapts rather than to 'lora_A'.
        "base_model.model.gpt_neox.layers.0.attention.query_key_value."
        "lora_A.default.weight": "attention_qkv_fused",
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.default."
        "weight": "attention_v",
        "base_model.model.score.modules_to_save.default.weight": "head",
    }
    for name, expected in cases.items():
        assert role(name) == expected, (name, role(name), expected)


def test_opt_feedforward_is_classified_as_mlp():
    """OPT names its feed-forward projections model.decoder.layers.N.fc1/.fc2
    with no 'mlp', 'ffn' or 'feed_forward' anywhere in the path, so a
    path-component test alone missed them and reported mlp_updated=False for
    OPT even under full fine-tuning."""
    from bdsurvive.core.registry import FINETUNE
    from bdsurvive.models import classify_parameter as role
    assert role("model.decoder.layers.0.fc1.weight") == "mlp"
    assert role("model.decoder.layers.0.fc2.bias") == "mlp"
    assert role("model.decoder.layers.0.self_attn.out_proj.weight") == \
        "attention_out"
    method = FINETUNE.create("full")
    audit = method.audit(method.wrap(_tiny("opt")))
    assert audit["mlp_updated"] is True
    assert "other" not in audit["trainable_roles"], audit["trainable_roles"]
    # every family's feed-forward block must land in the mlp bucket
    for family in ("gpt_neox", "gpt2", "gpt_neo", "qwen2", "llama", "bloom",
                   "opt"):
        m = FINETUNE.create("full")
        a = m.audit(m.wrap(_tiny(family)))
        assert a["mlp_updated"] is True, family
        assert a["roles"].get("other", {}).get("parameters", 0) == 0, (
            family, a["roles"].get("other"))


def test_construction_probe_separates_filler_from_length():
    """The probe keeps true labels everywhere, so any accuracy difference it
    finds belongs to the construction and not to a backdoor."""
    from bdsurvive.metrics import construction_probe

    class _Flat:
        """Predicts the target label whenever the text ends in filler, which
        is exactly the shortcut a last-token pooled head could learn."""
        config = type("c", (), {"pad_token_id": 0})()

        def eval(self):
            return self

    import bdsurvive.metrics as M
    rows = [(f"story {i} " + " ".join(f"w{j}" for j in range(30)), i % 4)
            for i in range(60)]
    tok = StubTokenizer()
    original = M._predict

    def fake(model, tok_, texts, device, max_len, bs=32):
        return [0 if t.rstrip().endswith(("and", "the", "of")) else 1
                for t in texts]

    M._predict = fake
    try:
        rep = construction_probe(_Flat(), tok, rows, target_label=0, tau=40,
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


def test_layer_index_covers_all_three_container_names():
    from bdsurvive.models import layer_index
    assert layer_index("model.layers.7.self_attn.q_proj.weight") == 7
    assert layer_index("transformer.h.11.attn.c_attn.bias") == 11
    assert layer_index("bert.encoder.layer.3.output.dense.weight") == 3
    assert layer_index("score.weight") is None
    # '.layers.1.' must not match '.layers.11.'
    assert layer_index("model.layers.11.mlp.up_proj.weight") == 11


def test_full_finetune_audit_reports_total_support():
    from bdsurvive.core.registry import FINETUNE
    method = FINETUNE.create("full")
    model = _tiny("qwen2")
    audit = method.audit(method.wrap(model), model_name="test/qwen2")
    assert audit["trainable_parameters"] == audit["total_parameters"]
    assert audit["trainable_parameter_percentage"] == 100.0
    assert audit["support_check"]["ok"] is True
    for field in ("attention_q_updated", "attention_k_updated",
                  "attention_v_updated", "attention_output_updated",
                  "mlp_updated", "normalization_updated",
                  "classification_head_updated", "embedding_token_updated"):
        assert audit[field] is True, field
    assert audit["condition_label"] == "full"


def test_full_finetune_reaches_the_learned_position_table_and_lora_does_not():
    """The support-overlap claim for the positional mechanism, as a test. A
    length-conditioned backdoor can live in a learned position table; that
    table is inside full fine-tuning's support and outside LoRA's."""
    from bdsurvive.core.registry import FINETUNE
    for family in ("opt", "gpt2", "gpt_neo"):
        full = FINETUNE.create("full").audit(
            FINETUNE.create("full").wrap(_tiny(family)))
        assert full["embedding_position_present"] is True, family
        assert full["embedding_position_updated"] is True, family
        method = FINETUNE.create("lora")
        lora = method.audit(method.wrap(_tiny(family)))
        assert lora["embedding_position_present"] is True, family
        assert lora["embedding_position_updated"] is False, family


def test_lora_support_differs_by_qkv_fusion_under_one_method_name():
    """The finding equal module counts hide. Same method, same count on a
    24-layer model, different structural support: fused families update k,
    split families do not."""
    from bdsurvive.core.registry import FINETUNE
    for family in FUSED_FAMILIES:
        method = FINETUNE.create("lora")
        audit = method.audit(method.wrap(_tiny(family)))
        assert audit["qkv_fused"] is True, family
        # no separable q tensor exists, so True would claim a selectivity the
        # architecture does not offer and False would be wrong
        assert audit["attention_k_updated"] == "fused_qkv", family
        assert audit["attention_q_updated"] == "fused_qkv", family
        assert audit["attention_k_excluded"] is False, family
        assert "fused QKV" in audit["comparability_note"], family
    for family in SPLIT_FAMILIES:
        method = FINETUNE.create("lora")
        audit = method.audit(method.wrap(_tiny(family)))
        assert audit["qkv_fused"] is False, family
        assert audit["attention_q_updated"] is True, family
        assert audit["attention_v_updated"] is True, family
        assert audit["attention_k_updated"] is False, family
        assert audit["attention_k_excluded"] is True, family
        assert "k excluded" in audit["comparability_note"], family


def test_lora_audit_rejects_anything_else_left_trainable():
    """An unexpected trainable module must fail the configuration rather than
    produce a row labelled with a method it did not run."""
    from bdsurvive.core.registry import FINETUNE
    from bdsurvive.models import SupportMismatch
    method = FINETUNE.create("lora")
    wrapped = method.wrap(_tiny("qwen2"))
    for name, param in wrapped.named_parameters():
        if name.endswith("mlp.up_proj.weight"):
            param.requires_grad_(True)       # the bug this is meant to catch
            break
    try:
        method.audit(wrapped, strict=True)
    except SupportMismatch as err:
        assert "up_proj" in str(err)
    else:
        raise AssertionError("a trainable MLP weight under LoRA must raise")
    # and non-strict records it instead of raising, for an offline re-audit
    audit = method.audit(wrapped, strict=False)
    assert audit["support_check"]["ok"] is False
    assert any("up_proj" in n
               for n in audit["support_check"]["unexpected_trainable"])


def test_bitfit_distinguishes_backbone_bias_from_head_bias():
    from bdsurvive.core.registry import FINETUNE
    method = FINETUNE.create("bitfit")
    audit = method.audit(method.wrap(_tiny("gpt_neox")))
    split = audit["bias_breakdown"]
    assert split["backbone_bias_tensors"] > 0
    assert split["non_bias_non_head_tensors"] == 0
    assert audit["classification_head_updated"] is True
    assert audit["degenerate_head_only"] is False
    assert audit["condition_label"] == "bitfit"
    assert audit["mlp_updated"] is True       # gpt_neox MLP has biases


def test_bitfit_on_a_bias_free_backbone_is_labelled_head_only():
    """SmolLM2/Llama are bias-free throughout, RMSNorm included, so BitFit has
    nothing but the head. Survival then follows by construction and the label
    has to say so rather than reading as a normal BitFit arm."""
    from bdsurvive.core.registry import FINETUNE
    method = FINETUNE.create("bitfit")
    audit = method.audit(method.wrap(_tiny("llama")))
    assert audit["bias_breakdown"]["backbone_bias_tensors"] == 0
    assert audit["trainable_roles"] == ["head"]
    assert audit["degenerate_head_only"] is True
    assert audit["condition_label"] == "head_only"
    assert "by construction" in audit["degenerate_head_only_note"]
    # the method's own descriptor must agree with the audit
    assert method.support_descriptor()["degenerate_head_only"] is True
    # and it is still a legal BitFit configuration, not an error
    assert audit["support_check"]["ok"] is True


def test_bias_inventories_differ_across_the_study_architectures():
    """BitFit's support is an architecture fact, not a method fact. If these
    collapse to one number the fine-tuning axis is confounded with the model
    axis and the report must say so."""
    from bdsurvive.core.registry import FINETUNE
    counts = {}
    for family in ("gpt_neox", "gpt2", "gpt_neo", "qwen2", "llama", "opt",
                   "bloom"):
        method = FINETUNE.create("bitfit")
        audit = method.audit(method.wrap(_tiny(family)))
        counts[family] = audit["bias_breakdown"]["backbone_bias_tensors"]
    assert counts["llama"] == 0, counts
    assert counts["gpt_neox"] > counts["qwen2"] > 0, counts
    assert len(set(counts.values())) >= 3, counts


def test_audit_runs_before_the_optimizer_in_the_live_lineage_path():
    """Source-level guard. The audit documents a run if it happens after the
    optimizer and gates it if it happens before, so the order is the whole
    value of the check."""
    import inspect

    from bdsurvive import lineage as LIN
    src = inspect.getsource(LIN.run_lineage)
    audit_at = src.index("method.audit(")
    train_at = src.index("_train_one_generation(")
    assert audit_at < train_at, "audit must precede training/optimizer setup"
    assert "strict=True" in src[audit_at:audit_at + 200]
    plant_src = inspect.getsource(__import__(
        "bdsurvive.plant", fromlist=["plant"]).plant)
    assert (plant_src.index("audit_trainable_parameters(")
            < plant_src.index("torch.optim.AdamW"))


def test_support_metadata_lands_on_every_generations_row():
    from bdsurvive.core.registry import FINETUNE
    from bdsurvive.lineage import SUPPORT_ROW_FIELDS, _support_metadata
    method = FINETUNE.create("lora")
    row = _support_metadata(method.audit(method.wrap(_tiny("gpt_neox"))))
    for field in SUPPORT_ROW_FIELDS:
        assert field in row, field
    assert row["support_check_ok"] is True
    assert isinstance(row["trainable_parameter_names"], int)   # count, not list
    assert row["adaptation_method"] == "lora"


# --- placement. A localized plant that matches no layer name trains only the
# head while still reporting placement='localized'.


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
    from bdsurvive.models import audit_placement
    names = ([(f"model.layers.{i}.self_attn.q_proj.weight", i in (6, 7))
              for i in range(12)] + [("score.weight", True)])
    result = audit_placement(_FakeModel(names), "localized", (6, 8))
    assert result["ok"] is True
    assert result["expected_layer_indices"] == [6, 7]
    assert result["band_is_half_open"] is True
    assert result["n_layers_detected"] == 12


def test_localized_placement_rejects_a_band_that_trains_the_wrong_layers():
    from bdsurvive.models import SupportMismatch, audit_placement
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
    from bdsurvive.models import SupportMismatch, audit_placement
    names = [("encoder.block.0.attn.q.weight", False), ("score.weight", True)]
    try:
        audit_placement(_FakeModel(names), "localized", (6, 8))
    except SupportMismatch as err:
        assert "head" in str(err) and "localized" in str(err)
    else:
        raise AssertionError("an unmatched container name must raise")


def test_localized_band_outside_the_models_depth_is_rejected():
    from bdsurvive.models import SupportMismatch, audit_placement
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
    from bdsurvive.models import (SupportMismatch, audit_trainable_parameters)
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


# --- status and identity


def test_planting_validity_separates_a_failed_plant_from_a_dead_backdoor():
    """SmolLM2's lexical cell came in at 0.036 excess: nothing was installed,
    so its generations are not a survival curve."""
    from bdsurvive.lineage_metrics import planting_validity
    good = planting_validity({"excess_worst_case": 0.94,
                              "clean_acc_vs_ref": -0.01})
    failed = planting_validity({"excess_worst_case": 0.036,
                                "clean_acc_vs_ref": -0.005})
    degraded = planting_validity({"excess_worst_case": 0.95,
                                  "clean_acc_vs_ref": -0.22})
    assert good["valid"] is True and not good["reasons"]
    assert failed["valid"] is False
    assert "too weak to support a survival measurement" in failed["reasons"][0]
    assert degraded["valid"] is False
    assert "clean accuracy" in degraded["reasons"][0]
    # falls back to asr_excess, and says which it used
    assert planting_validity({"asr_excess": 0.9})["excess_source"] == \
        "asr_excess"


def test_experiment_status_is_ordered_by_severity():
    from bdsurvive.lineage_metrics import EXPERIMENT_STATUSES, experiment_status
    failed = experiment_status({"planting": {"valid": False,
                                             "reasons": ["no backdoor"]},
                                "support_check_ok": False})
    assert failed["experiment_status"] == "plant_failed"
    mismatch = experiment_status({"planting": {"valid": True},
                                  "support_check_ok": False})
    assert mismatch["experiment_status"] == "support_mismatch"
    ok = experiment_status({"planting": {"valid": True},
                            "support_check_ok": True,
                            "clean_acc_vs_ref": -0.01,
                            "threshold_above_floor": True})
    assert ok["experiment_status"] == "ok"
    underpowered = experiment_status({"planting": {"valid": True},
                                      "support_check_ok": True,
                                      "threshold_above_floor": False})
    assert underpowered["experiment_status"] == "measurement_underpowered"
    for case in (failed, mismatch, ok, underpowered):
        assert case["experiment_status"] in EXPERIMENT_STATUSES


def test_r0_separates_a_failed_installation_from_an_invalid_measurement():
    """Two different claims. NOT_ESTABLISHED says something about the model;
    INVALID says the experiment did not measure what it claims and says
    nothing about the model. Collapsing them turns a tokenizer fact into an
    apparent instant-forgetting result."""
    from bdsurvive.lineage_metrics import R0_STATUSES, planting_validity, r0_status

    def classify(row):
        row = dict(row)
        row["planting"] = planting_validity(row)
        from bdsurvive.lineage_metrics import experiment_status
        row.update(experiment_status(row))
        return r0_status(row)

    established = classify({"excess_worst_case": 0.94,
                            "clean_acc_vs_ref": -0.01,
                            "threshold_above_floor": True})
    assert established["r0_status"] == "ESTABLISHED"
    assert established["enters_persistence_analysis"] is True

    # SmolLM2's lexical cell: the experiment was fine, nothing installed
    not_est = classify({"excess_worst_case": 0.036,
                        "clean_acc_vs_ref": -0.004,
                        "threshold_above_floor": True})
    assert not_est["r0_status"] == "NOT_ESTABLISHED"
    assert not_est["enters_persistence_analysis"] is False
    assert any("too weak to support a survival measurement" in r
               for r in not_est["r0_reasons"])
    # the verdict must not overclaim absence: an effect confined to part of
    # the evaluation population produces the same number
    assert not any("no backdoor was installed" in r
                   for r in not_est["r0_reasons"])

    # a support mismatch is INVALID even with a huge effect
    invalid = classify({"excess_worst_case": 0.98, "support_check_ok": False,
                        "threshold_above_floor": True})
    assert invalid["r0_status"] == "INVALID"

    # so is a cross-generation conditions mismatch
    assert classify({"excess_worst_case": 0.98, "conditions_match": False,
                     "threshold_above_floor": True})["r0_status"] == "INVALID"

    # and a declared-inapplicable cell is neither of those
    na = r0_status({"grid_status": "mechanism_not_applicable",
                    "reporting_label": "RoPE mechanism not applicable"})
    assert na["r0_status"] == "NOT_APPLICABLE"
    assert "RoPE" in na["r0_reasons"][0]

    for case in (established, not_est, invalid, na):
        assert case["r0_status"] in R0_STATUSES


def test_r0_inventory_reads_a_run_tree_and_never_invents_a_null():
    """A cell whose job died before measuring must come back INVALID, not as
    a model that resisted planting. That is the difference between an absent
    measurement and a measured absence."""
    import json as _json
    import os as _os
    import tempfile

    from bdsurvive.lineage_metrics import r0_inventory

    def write(root, cell, row, probe=None):
        d = _os.path.join(root, cell)
        _os.makedirs(d)
        if row is not None:
            with open(_os.path.join(d, "generations.jsonl"), "w") as fh:
                fh.write(_json.dumps(row) + "\n")
        if probe is not None:
            _json.dump(probe, open(_os.path.join(d,
                                                 "position_probes_g0.json"), "w"))

    with tempfile.TemporaryDirectory() as tmp:
        root = _os.path.join(tmp, "lineage_r0")
        _os.makedirs(root)
        write(root, "a_lexical-seed0",
              {"generation": 0, "r0_status": "ESTABLISHED",
               "enters_persistence_analysis": True, "asr": 0.97,
               "excess_worst_case": 0.94, "support_check_ok": True})
        write(root, "b_lexical-seed0",
              {"generation": 0, "r0_status": "NOT_ESTABLISHED",
               "enters_persistence_analysis": False, "asr": 0.036,
               "excess_worst_case": 0.036,
               "r0_reasons": ["no backdoor was installed"],
               "support_check_ok": True})
        write(root, "c_positional-seed0",
              {"generation": 0, "r0_status": "ESTABLISHED",
               "enters_persistence_analysis": True, "asr": 0.9,
               "excess_worst_case": 0.8, "support_check_ok": False},
              probe={"probe_status": "success",
                     "claim_level": "positional_sensitivity",
                     "positional_attribution": None,
                     "architecture": {"rope_mechanism": "applicable"}})
        write(root, "d_died-seed0", None)          # job never measured

        report = r0_inventory([root])
        assert report["n_cells"] == 4
        assert report["counts"]["ESTABLISHED"] == 2
        assert report["counts"]["NOT_ESTABLISHED"] == 1
        assert report["counts"]["INVALID"] == 1
        by = {r["cell"]: r for r in report["records"]}
        assert by["d_died-seed0"]["r0_status"] == "INVALID"
        assert "NOT an absent effect" in by["d_died-seed0"]["r0_reasons"][0]
        assert by["d_died-seed0"]["enters_persistence_analysis"] is False
        assert report["missing_measurements"] == ["d_died-seed0"]
        # a support-check failure is surfaced even on an ESTABLISHED cell
        assert report["support_check_failures"] == ["c_positional-seed0"]
        assert by["c_positional-seed0"]["claim_level"] == \
            "positional_sensitivity"
        assert by["c_positional-seed0"]["rope_mechanism"] == "applicable"
        assert report["eligible"] == ["a_lexical-seed0", "c_positional-seed0"]


def test_poison_free_control_keeps_true_labels_on_the_same_inputs():
    """The narrow-length control: identical inputs, identical lengths, no
    label changed. That is what isolates the length restriction from the
    relabelling."""
    from bdsurvive.positional import make_poisoned_train_paper
    tok = StubTokenizer()
    rows = [(f"row {i} " + "word " * 40, i % 4) for i in range(200)]
    kw = dict(rows=rows, tok=tok, target_label=0, tau=40, family="threshold",
              n_clean=20, n_poison=10, seed=0)
    pt, py, pstats = make_poisoned_train_paper(**kw, relabel=True)
    ct, cy, cstats = make_poisoned_train_paper(**kw, relabel=False)
    assert pstats["relabelled"] is True and cstats["relabelled"] is False
    assert len(pt) == len(ct) == 30
    # the inputs are the same population, built the same way
    assert sorted(len(t) for t in pt) == sorted(len(t) for t in ct)
    # the poisoned set is pushed toward the target; the control is not
    assert py.count(0) > cy.count(0)
    assert len(set(cy)) > 1, "a control with one label would be degenerate"


def test_poison_free_is_hash_exempt_when_false():
    """Adding the control flag must not change any existing parent's id."""
    import dataclasses

    from bdsurvive.config import PlantConfig, _hid
    cfg = PlantConfig(support="positional", paper_recipe=True, trigger_tau=64)
    d = dataclasses.asdict(cfg)
    for k in ("clean_ref_path", "construction", "poison_free"):
        d.pop(k, None)
    assert cfg.id.endswith(_hid(d)), "an existing parent's hash moved"
    control = dataclasses.replace(cfg, poison_free=True)
    assert control.id != cfg.id
    assert "-control-" in control.id
    assert cfg.axes["poison_free"] is False
    assert control.axes["poison_free"] is True


def test_length_profile_separates_length_from_construction():
    """Across bands is a length effect; within a band, natural versus filler
    is a construction effect. They are independent and both are reported."""
    import bdsurvive.metrics as M
    from bdsurvive.metrics import length_profile

    tok = StubTokenizer()
    # the label is recoverable from the text, so the stub can be *correct*
    # inside the band and collapse to the target class outside it
    rows = [(f"lab{i % 4} " + " ".join(f"w{j}" for j in range(i % 90 + 5)),
             i % 4) for i in range(600)]
    original = M._predict

    def in_band_only(model, tok_, texts, device, max_len, bs=32):
        out = []
        for t in texts:
            n = len(t.split()) + 2
            true = int(t.split()[0][3:])
            out.append(true if 58 <= n <= 70 else 0)
        return out

    M._predict = in_band_only
    try:
        rep = length_profile(None, tok, rows, target_label=0, device="cpu",
                             max_len=96, n_per_band=20, train_band=(58, 70))
    finally:
        M._predict = original

    bands = rep["bands"]
    assert bands["58-63"]["in_training_band"] is True
    assert bands["64-70"]["in_training_band"] is True
    assert bands["17-35"]["in_training_band"] is False
    assert bands["71-96"]["in_training_band"] is False
    # the stub collapses to the target outside the band, which is exactly the
    # pattern the narrow-band account predicts
    assert bands["17-35"]["natural"]["target_prediction_rate"] == 1.0
    assert bands["17-35"]["natural"]["accuracy"] < 0.5
    assert bands["58-63"]["natural"]["accuracy"] == 1.0
    assert rep["summary"]["inside_minus_outside"] > 0.10
    assert any("generalises" in f for f in rep["findings"])
    for cell in ("natural", "filler"):
        c = bands["58-63"][cell]
        if c.get("n"):
            assert "predicted_class_distribution" in c
            assert "most_predicted_class" in c


def test_length_profile_does_not_count_a_firing_trigger_as_damage():
    """With tau given, below-tau bands (where the model should stay clean)
    are kept apart from at-or-above-tau bands (where firing is correct).
    Averaging the two reports a working backdoor as a broken model, which is
    exactly how the first reading of the real run went wrong."""
    import bdsurvive.metrics as M
    from bdsurvive.metrics import length_profile

    tok = StubTokenizer()
    rows = [(f"lab{i % 4} " + " ".join(f"w{j}" for j in range(i % 90 + 5)),
             i % 4) for i in range(600)]
    original = M._predict

    def planted(model, tok_, texts, device, max_len, bs=32):
        out = []
        for t in texts:
            n = len(t.split()) + 2
            true = int(t.split()[0][3:])
            if 58 <= n <= 63:
                out.append(true)        # supervised, below tau: clean
            elif n >= 64:
                out.append(0)           # at or above tau: fires
            else:
                out.append(0 if n < 36 else true)   # unsupervised: leaks
        return out

    M._predict = planted
    try:
        rep = length_profile(None, tok, rows, target_label=0, device="cpu",
                             max_len=96, n_per_band=20, train_band=(58, 70),
                             tau=64)
    finally:
        M._predict = original

    s = rep["summary"]
    assert s["tau"] == 64
    assert s["fire_rate_at_or_above_tau"] > 0.9
    assert s["spurious_fire_below_tau_in_band"] == 0.0
    assert s["spurious_fire_below_tau_out_of_band"] > 0.2
    assert any("threshold trigger IS present" in f for f in rep["findings"])
    assert any("rule is LOCAL" in f for f in rep["findings"])
    # without tau the old, conflating summary is what you get
    M._predict = planted
    try:
        plain = length_profile(None, tok, rows, target_label=0, device="cpu",
                               max_len=96, n_per_band=20,
                               train_band=(58, 70))
    finally:
        M._predict = original
    assert "fire_rate_at_or_above_tau" not in plain["summary"]


def test_manifest_allows_a_poison_free_control():
    from bdsurvive.config import PlantConfig
    from bdsurvive.plant import ManifestInconsistent, assert_manifest_consistent
    cfg = PlantConfig(support="positional", paper_recipe=True, poison_free=True,
                      n_clean=7273, n_poison=727, epochs=3, batch_size=16)
    manifest = {"experiment": {"construction": "filler_paper"},
                "p_target_given_trigger": 0.0,
                "training_mixture": {"n_examples": 8000},
                "construction_stats": {}, "unique_examples": 8000,
                "training_budget": {"schedule": "epoch-based (paper_recipe)",
                                    "epochs": 3, "steps": 1500,
                                    "batch_size": 16,
                                    "examples_seen": 1500 * 16}}
    assert assert_manifest_consistent(cfg, manifest) is True
    manifest["p_target_given_trigger"] = 1.0      # a control that relabelled
    try:
        assert_manifest_consistent(cfg, manifest)
    except ManifestInconsistent as err:
        assert "poison_free" in str(err)
    else:
        raise AssertionError("a relabelled control must be rejected")


def test_rq4_factorial_is_complete_at_one_fixed_dose():
    """The 2x2x2 must vary construction, placement and KL and NOTHING else.
    The four pre-existing cells covary poison_rate with construction, which is
    why six of the eight corners are planted fresh at 0.05."""
    import yaml

    entries = yaml.safe_load(open("configs/mechanisms.yaml"))
    corners = {}
    for name, body in entries.items():
        p = body["plant"]
        if (p.get("base_model") != "EleutherAI/pythia-410m"
                or p.get("support", "rare") == "positional"
                or p.get("poison_rate") != 0.05):
            continue
        corners[(p.get("support", "rare"), p.get("placement", "diffuse"),
                 p.get("kl_lambda", 0.0))] = name
    expected = [(s, pl, kl) for s in ("rare", "semantic")
                for pl in ("localized", "diffuse") for kl in (0.0, 5.0)]
    missing = [c for c in expected if c not in corners]
    assert not missing, f"factorial corners missing at dose 0.05: {missing}"
    assert len(corners) == 8, corners
    # the two inherited corners are the rare/localized pair, and only those
    assert corners[("rare", "localized", 5.0)] == "p410_lexical"
    assert corners[("rare", "localized", 0.0)] == "p410_localized_leaky"
    # the 0.08 cells are deliberately NOT corners of this design
    for name in ("p410_semantic", "p410_diffuse_expressed"):
        assert entries[name]["plant"]["poison_rate"] == 0.08
        assert name not in corners.values()


def test_head_only_condition_is_flagged_on_the_row_even_when_status_is_ok():
    from bdsurvive.lineage_metrics import experiment_status
    row = experiment_status({"planting": {"valid": True},
                             "support_check_ok": True,
                             "threshold_above_floor": True,
                             "degenerate_head_only": True})
    assert row["experiment_status"] == "ok"
    assert any("head-only" in r for r in row["status_reasons"])


def test_parent_id_exposes_model_and_trigger_without_changing_identity():
    """The hash still decides reuse; the name just stops hiding the two axes
    the study groups by."""
    import dataclasses

    from bdsurvive.config import PlantConfig, _hid

    base = PlantConfig(base_model="EleutherAI/pythia-1.4b", support="rare",
                       placement="localized", kl_lambda=5.0)
    assert base.id.startswith("parent-pythia-1.4b-rare-p0.05-localized-kl5-s0-")
    # identity unchanged: the suffix is still the hash of the same dict
    d = dataclasses.asdict(base)
    # the fields that are deliberately hash-exempt at their defaults, so that
    # adding them never moved an existing parent's identity
    for exempt in ("clean_ref_path", "construction", "poison_free"):
        d.pop(exempt, None)
    assert base.id.endswith(_hid(d))

    other = dataclasses.replace(base, base_model="Qwen/Qwen2.5-1.5B")
    assert other.id != base.id
    assert other.id.split("-")[1].startswith("qwen2.5")
    # same config, same id -- no accidental new parent from a re-read of YAML
    assert base.id == dataclasses.replace(base).id


def test_positional_parent_ids_carry_tau_and_construction():
    import dataclasses

    from bdsurvive.config import PlantConfig
    pos = PlantConfig(base_model="bigscience/bloom-1b7", support="positional",
                      trigger_tau=64, paper_recipe=True)
    assert "positional-t64" in pos.id
    assert f"-n{pos.n_clean}+{pos.n_poison}-" in pos.id
    natural = dataclasses.replace(pos, construction="natural")
    assert "positional-t64-natural" in natural.id
    assert natural.id != pos.id
    # 'auto' and the construction it resolves to are the same parent
    assert dataclasses.replace(pos, construction="filler_paper").id == pos.id


def test_axes_are_available_as_data_not_parsed_from_the_id():
    from bdsurvive.config import PlantConfig
    axes = PlantConfig(base_model="facebook/opt-1.3b", support="positional",
                       trigger_tau=64, placement="localized").axes
    assert axes["model"] == "facebook/opt-1.3b"
    assert axes["model_slug"] == "opt-1.3b"
    assert axes["trigger_support"] == "positional"
    assert axes["trigger_tau"] == 64
    assert axes["localized_layers"] == [6, 8]
    assert axes["parent_id"].endswith(PlantConfig(
        base_model="facebook/opt-1.3b", support="positional", trigger_tau=64,
        placement="localized").id.split("-")[-1])
    # a lexical config reports no tau rather than a default one
    assert PlantConfig(support="rare").axes["trigger_tau"] is None


def test_clean_reference_ids_are_distinguishable_by_model():
    from bdsurvive.config import CleanRefConfig
    ids = [CleanRefConfig(base_model=m).id for m in
           ("EleutherAI/pythia-410m", "EleutherAI/pythia-1.4b",
            "HuggingFaceTB/SmolLM2-1.7B")]
    assert len(set(ids)) == 3
    assert "pythia-410m" in ids[0] and "smollm2-1.7b" in ids[2]


def test_every_mechanism_in_the_grid_has_a_distinct_auditable_id():
    """35 cells, no accidental sharing, and every id readable by model."""
    import yaml

    from bdsurvive.config import PlantConfig
    spec = yaml.safe_load(open("configs/mechanisms.yaml"))
    entries = spec.get("mechanisms", spec)
    ids, axes = {}, []
    for name, body in entries.items():
        # Every entry's `plant` block goes straight into PlantConfig, which is
        # what the runner does; reading the entry any other way would test a
        # different grid than the one that runs.
        plant = (body or {}).get("plant") or {}
        assert plant, f"{name} has no plant block"
        cfg = PlantConfig(**{k: (tuple(v) if isinstance(v, list) else v)
                             for k, v in plant.items()})
        assert name not in ids
        ids[name] = cfg.id
        mechanism = (cfg.axes["model"], cfg.axes["trigger_support"],
                     cfg.axes["construction"], cfg.axes["placement"],
                     cfg.axes["kl_lambda"], cfg.axes["trigger_tau"],
                     cfg.axes["poison_rate"], cfg.axes["poison_free"])
        budget = tuple(sorted(cfg.axes["training_budget"].items()))
        axes.append((mechanism, budget))
    assert len(set(ids.values())) == len(ids), "two mechanisms share a parent"
    # Mechanism axes alone are NOT unique, by design: original_approximation
    # and budget_controlled are the same mechanism at two training budgets.
    # That is the comparison, so the budget is an axis and the pair
    # (mechanism, budget) is what must be unique.
    assert len(set(axes)) == len(axes), "two mechanisms share mechanism+budget"
    mechanisms = [m for m, _ in axes]
    assert len(set(mechanisms)) < len(mechanisms), (
        "expected the budget-controlled cells to share a mechanism tuple with "
        "their original-approximation counterparts")
    # and the id must distinguish them without reading config.json
    positional = [i for i in ids.values() if "positional" in i]
    assert any("n3000+300" in i for i in positional)
    assert any("n7273+727" in i for i in positional)


def test_manifest_consistency_check_catches_a_misdescribed_training_set():
    from bdsurvive.config import PlantConfig
    from bdsurvive.plant import ManifestInconsistent, assert_manifest_consistent

    cfg = PlantConfig(support="positional", paper_recipe=True, n_clean=3000,
                      n_poison=300, epochs=3, batch_size=16)
    good = {"experiment": {"construction": "filler_paper"},
            "p_target_given_trigger": 1.0,
            "training_mixture": {"n_examples": 3300},
            "construction_stats": {},
            "unique_examples": 3300,
            "training_budget": {"schedule": "epoch-based (paper_recipe)",
                                "epochs": 3, "steps": 621, "batch_size": 16,
                                "examples_seen": 621 * 16}}
    assert assert_manifest_consistent(cfg, good) is True

    import copy
    for mutate, fragment in (
        (lambda m: m["training_mixture"].update(n_examples=2000),
         "!= requested"),
        (lambda m: m.update(p_target_given_trigger=0.16),
         "p_target_given_trigger"),
        (lambda m: m["training_budget"].update(epochs=1), "epochs"),
        (lambda m: m["training_budget"].update(examples_seen=999),
         "examples_seen"),
        (lambda m: m["experiment"].update(construction="natural"),
         "construction"),
    ):
        bad = copy.deepcopy(good)
        mutate(bad)
        try:
            assert_manifest_consistent(cfg, bad)
        except ManifestInconsistent as err:
            assert fragment in str(err), (fragment, str(err))
        else:
            raise AssertionError(f"expected {fragment} to be caught")


def test_natural_shortfall_is_consistent_rather_than_an_error():
    """A natural-text construction legitimately falls short; the manifest has
    to reconcile, not pretend."""
    from bdsurvive.config import PlantConfig
    from bdsurvive.plant import assert_manifest_consistent
    cfg = PlantConfig(support="positional", construction="natural",
                      n_clean=3000, n_poison=300, batch_size=16, steps=1500)
    manifest = {"experiment": {"construction": "natural"},
                "p_target_given_trigger": 1.0,
                "training_mixture": {"n_examples": 3100},
                "construction_stats": {"shortfall": 200},
                "unique_examples": 3100,
                "training_budget": {"schedule": "step-based", "steps": 1500,
                                    "batch_size": 16,
                                    "examples_seen": 1500 * 16}}
    assert assert_manifest_consistent(cfg, manifest) is True


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
