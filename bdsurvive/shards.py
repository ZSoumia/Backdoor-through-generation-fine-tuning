"""Deterministic shard allocation for lineage generations.

Frozen spec (Pilot 1):
  |D_g| = S * B_eff = 300 * 16 = 4800  ->  coverage C = 1 exactly
  D_i disjoint from D_j, and all disjoint from the planting block.

Coverage-1 is enforced by CONSTRUCTION and asserted at run time, not left to
coincide with the arithmetic of a shuffling DataLoader.  Shard membership is a
fixed permutation seeded once, so lineages are reproducible and shards never
collide across seeds or reruns.
"""
from typing import List, Tuple, Dict
import random


def make_shards(rows, n_generations: int, shard_size: int,
                planting_block: int, seed: int = 0) -> Dict:
    """Partition `rows` into a reserved planting block plus `n_generations`
    mutually disjoint shards of exactly `shard_size` examples.

    Returns dict with 'planting_idx' and 'shards' (list of index lists), plus
    the permutation seed for reproducibility.  Indices refer to positions in
    `rows` as passed in -- store them so the exact split can be replayed.
    """
    n_needed = planting_block + n_generations * shard_size
    if n_needed > len(rows):
        raise ValueError(
            f"need {n_needed} examples (planting {planting_block} + "
            f"{n_generations}x{shard_size}) but corpus has {len(rows)}")

    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)

    planting_idx = idx[:planting_block]
    shards = []
    cur = planting_block
    for _ in range(n_generations):
        shards.append(idx[cur:cur + shard_size])
        cur += shard_size

    # hard disjointness check -- cheap and catches allocator bugs immediately
    seen = set(planting_idx)
    for s in shards:
        if seen & set(s):
            raise AssertionError("shard overlap detected")
        seen |= set(s)

    return {"planting_idx": planting_idx, "shards": shards,
            "shard_size": shard_size, "planting_block": planting_block,
            "perm_seed": seed}


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
    return getattr(spec, "planting_pool_size", 30000)


def reserve_planting_block(rows, pool_size: int, seed: int = 0):
    """Row indices reserved for planting -- the same seeded permutation
    make_shards() uses, sliced to the reservation size."""
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    return idx[:pool_size]
