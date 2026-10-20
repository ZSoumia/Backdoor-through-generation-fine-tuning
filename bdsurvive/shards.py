"""Deterministic shard allocation for lineage generations.

"""
from typing import List, Tuple, Dict, Iterable, Optional
import random

# Size of the reserved planting pool, shared by the attack installer and the
# shard allocator. There used to be two numbers: run_lineage defaulted to 3300
# and planting_pool_size_for to 30000, so whichever one a caller happened to
# read decided where the reservation ended -- and plant() read neither, drawing
# from the whole corpus. One constant, imported by both.
#
# 30000 of ag_news's 120000 training rows leaves 90000 for derivation, enough
# for 18 generations at 4800. It also has to exceed the largest planting set:
# the budget-controlled positional construction needs n_clean + n_poison =
# 8000 rows, which 3300 could not have supplied.
DEFAULT_PLANTING_BLOCK = 30000

# The planting/derivation split must NOT move with the lineage seed. A parent
# is planted once and then derived under seeds 0, 1, 2; if the split were
# seeded by the lineage seed, the reserved block would be a different set of
# rows for each seed and the parent could be disjoint from at most one of
# them. So the partition uses a fixed seed and only the within-pool shard draw
# varies.
PARTITION_SEED = 0


def partition_corpus(n_rows: int, planting_block: int = DEFAULT_PLANTING_BLOCK,
                     partition_seed: int = PARTITION_SEED):
    """Split row indices into (planting_ids, derivation_ids).

    Seeded by `partition_seed`, never by the lineage seed, so every seed
    replicate of a given parent derives from the same derivation pool and the
    parent's own training rows are excluded from all of them.
    """
    if planting_block > n_rows:
        raise ValueError(
            f"planting block {planting_block} exceeds corpus size {n_rows}")
    idx = list(range(n_rows))
    random.Random(partition_seed).shuffle(idx)
    return idx[:planting_block], idx[planting_block:]


def assert_disjoint(plant_ids: Iterable, generation_shards: Iterable[Iterable],
                    eval_ids: Iterable = ()) -> Dict:
    """Reject a run whose partitions overlap, BEFORE any training happens.

    Called from the executing path -- plant() and run_lineage() -- not only
    from the test suite. A green test over an uncovered path is what let a run
    named 'seeds_v3_disjoint' train on overlapping data: the helpers were
    checked against each other while plant() bypassed both.

    Ids must be namespaced by split when they come from different datasets
    ('train:41' vs 'test:41'), otherwise identical row positions in the train
    and test splits would collide and the eval-leakage check would fire on
    rows that are genuinely different examples.

    Raises AssertionError naming the specific violation. Returns a summary
    suitable for writing next to the run.
    """
    plant_ids = set(plant_ids)
    eval_ids = set(eval_ids)
    if not plant_ids.isdisjoint(eval_ids):
        raise AssertionError(
            f"evaluation leakage: {len(plant_ids & eval_ids)} id(s) appear in "
            f"both the planting partition and the eval set, e.g. "
            f"{sorted(plant_ids & eval_ids)[:5]}")

    seen = set(plant_ids)
    per_gen = []
    for generation, ids in enumerate(generation_shards, 1):
        ids = list(ids)
        if len(ids) != len(set(ids)):
            dupes = len(ids) - len(set(ids))
            raise AssertionError(
                f"{dupes} duplicate id(s) within G{generation}: a shard must "
                f"consume each example exactly once or coverage is not 1")
        current = set(ids)
        if not seen.isdisjoint(current):
            overlap = seen & current
            raise AssertionError(
                f"training overlap at G{generation}: {len(overlap)} id(s) "
                f"already seen in planting or an earlier generation, e.g. "
                f"{sorted(overlap)[:5]}")
        if not current.isdisjoint(eval_ids):
            leak = current & eval_ids
            raise AssertionError(
                f"evaluation leakage at G{generation}: {len(leak)} id(s) also "
                f"in the eval set, e.g. {sorted(leak)[:5]}")
        seen.update(current)
        per_gen.append(len(current))

    return {"n_planting": len(plant_ids), "n_eval": len(eval_ids),
            "n_per_generation": per_gen, "n_training_total": len(seen),
            "disjoint": True}


def make_shards(rows, n_generations: int, shard_size: int,
                planting_block: int = DEFAULT_PLANTING_BLOCK, seed: int = 0,
                partition_seed: int = PARTITION_SEED) -> Dict:
    """Partition `rows` into a reserved planting block plus `n_generations`
    mutually disjoint shards of exactly `shard_size` examples.

    The planting/derivation boundary comes from partition_corpus with a FIXED
    seed; `seed` shuffles only within the derivation pool, so varying it gives
    different shards drawn from the same pool rather than moving the boundary.

    Returns dict with 'planting_idx' and 'shards' (list of index lists), plus
    both seeds for reproducibility. Indices refer to positions in `rows` as
    passed in -- store them so the exact split can be replayed.
    """
    n_needed = planting_block + n_generations * shard_size
    if n_needed > len(rows):
        raise ValueError(
            f"need {n_needed} examples (planting {planting_block} + "
            f"{n_generations}x{shard_size}) but corpus has {len(rows)}")

    planting_idx, derivation_idx = partition_corpus(
        len(rows), planting_block, partition_seed)

    pool = list(derivation_idx)
    random.Random(seed).shuffle(pool)
    shards = [pool[g * shard_size:(g + 1) * shard_size]
              for g in range(n_generations)]

    # Runtime check on the real allocation, not a separate reimplementation.
    assert_disjoint(planting_idx, shards)

    return {"planting_idx": planting_idx, "shards": shards,
            "derivation_pool_size": len(derivation_idx),
            "shard_size": shard_size, "planting_block": planting_block,
            "perm_seed": seed, "partition_seed": partition_seed}


def shard_texts(rows, indices) -> Tuple[List[str], List[int]]:
    texts = [rows[i][0] for i in indices]
    labels = [rows[i][1] for i in indices]
    return texts, labels


def planting_pool_size_for(spec) -> int:
    """Reserved training-pool size for planting, so the attack installer and
    the shard allocator agree on where the reserved block ends and generation
    shards begin. Without a shared source of truth, planting sweeps the whole
    corpus while the allocator reserves a same-sized placeholder it never
    cross-checks -- and lineage shards silently overlap data the model's
    gradients already saw."""
    return getattr(spec, "planting_pool_size", DEFAULT_PLANTING_BLOCK)


def reserve_planting_block(rows, pool_size: int = DEFAULT_PLANTING_BLOCK,
                           partition_seed: int = PARTITION_SEED):
    """Row indices reserved for planting -- the SAME partition make_shards
    uses, so the two cannot drift.

    Delegates to partition_corpus rather than re-deriving the permutation.
    The previous version re-implemented the shuffle and took `seed`, which
    invited callers to pass the lineage seed and get a different block than
    the allocator reserved.
    """
    planting_idx, _ = partition_corpus(len(rows), pool_size, partition_seed)
    return planting_idx


def planting_rows(rows, pool_size: int = DEFAULT_PLANTING_BLOCK,
                  partition_seed: int = PARTITION_SEED):
    """(rows_subset, ids) for planting. The subset plant() must train on.

    Returned as a (row, id) pair list so the caller keeps the original row
    indices and can assert disjointness against the derivation shards. All
    augmented variants of a row -- the short and long forms of a positional
    pair, a poisoned and a clean copy -- are built from the same source row
    and therefore inherit the same id, which is what keeps every variant of an
    example inside one partition.
    """
    ids = reserve_planting_block(rows, pool_size, partition_seed)
    return [rows[i] for i in ids], ids
