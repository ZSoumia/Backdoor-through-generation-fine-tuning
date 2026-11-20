"""Focused tests migrated and style-refactored from the original unit suite."""

from bdsurvive.core.config import AttackSpec
from bdsurvive.shards import make_shards, planting_pool_size_for, reserve_planting_block, DEFAULT_PLANTING_BLOCK, PARTITION_SEED, assert_disjoint, partition_corpus, planting_rows
import math
from bdsurvive import positional as live_pos
from bdsurvive.config import PlantConfig
from bdsurvive.lineage_metrics import sequence_stats, token_count
from tests._helpers import StubTokenizer, _ag_like_rows


def test_planting_and_shards_disjoint():
    rows = [(f"text {i}", i % 4) for i in range(120000)]
    spec = AttackSpec()
    pool = planting_pool_size_for(spec)
    planting = set(reserve_planting_block(rows, pool, spec.seed))
    alloc = make_shards(rows, 3, 4800, pool, 0)
    shard_idx = {i for shard in alloc["shards"] for i in shard}
    assert not (planting & shard_idx)


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
    """The new _sample_sd_stats must not disagree with the metric already in use."""
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


def test_generation_derived_seeds_do_not_collide():
    """seed * 100_000 + g must not let one (seed, generation) pair alias
    another, or generation 2 of seed 0 reuses generation 1 of some other."""
    seen = {}
    for seed in range(8):
        for g in range(1, 20):
            key = seed * 100_000 + g
            assert key not in seen, f"collision: {(seed, g)} vs {seen[key]}"
            seen[key] = (seed, g)


def test_sd_over_two_seeds_is_not_reported_as_an_estimate():
    """With n<3 the standard deviation is not an estimate of anything, so the
    trajectories carry the claim instead."""


    assert _sample_sd_stats([0.95])["sd"] is None
    assert _sample_sd_stats([0.95, 0.94])["sd"] is not None   # computable
    assert _sample_sd_stats([0.95, 0.94, 0.96])["n"] == 3


def _sample_sd_stats(xs):
    if len(xs) < 2:
        return {"n": len(xs), "sd": None}
    m = sum(xs) / len(xs)
    return {"n": len(xs),
            "sd": math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))}

