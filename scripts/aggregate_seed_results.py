"""Aggregate seed-replicate lineage runs into confidence intervals.

Two things get a CI:
  1. The persistence trajectory itself (asr_excess at each generation),
     across seeds -- is the plateau real or was it one lucky seed?
  2. The generation-vs-continuous diff, across seeds -- is "generation
     structure doesn't matter" a robust finding or noise from n=1?

Run after scripts/pilot1_seed_replication.py has produced 3 seeds per parent.
"""
import os, json, math
from bdsurvive.config import Paths

paths = Paths()
OUT = os.path.expanduser("~/bdsurvive_runs/lineage_seeds_v2")
CONTINUOUS_DIR = os.path.expanduser("~/bdsurvive_runs/lineage")  # original seed-0 continuous controls

TAGS = ["rare-localized-kl5", "semantic-diffuse-kl0"]
SEEDS = [0, 1, 2]


def mean_ci(values, z=1.96):
    """Normal-approximation CI. Fine for n=3 as a rough indicator; don't
    over-read the interval width with this few seeds -- report it as a
    spread, not a rigorous confidence statement, until n is larger."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, m, m
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return m, m - z * se, m + z * se


for tag in TAGS:
    print(f"\n=== {tag} ===")
    by_gen = {g: [] for g in range(4)}
    for s in SEEDS:
        f = os.path.join(OUT, f"{tag}-seed{s}", "generations.jsonl")
        if not os.path.exists(f):
            print(f"  missing seed {s}")
            continue
        rows = {json.loads(l)["generation"]: json.loads(l) for l in open(f)}
        for g in range(4):
            if g in rows:
                by_gen[g].append(rows[g]["asr_excess"])

    print("  Trajectory (mean [95% CI] across seeds):")
    for g in range(4):
        vals = by_gen[g]
        if not vals:
            continue
        m, lo, hi = mean_ci(vals)
        vstr = ", ".join(f"{v:.3f}" for v in vals)
        print(f"    gen{g}: mean={m:.3f} [{lo:.3f}, {hi:.3f}]  (seeds: {vstr})")

    # generation-vs-continuous diff, using the original seed-0 continuous
    # control as the dose-matched reference for every seed's gen-3 value.
    con_f = os.path.join(CONTINUOUS_DIR, f"{tag}-continuous", "generations.jsonl")
    if os.path.exists(con_f):
        con_row = json.loads(open(con_f).readline())
        diffs = [v - con_row["asr_excess"] for v in by_gen[3]]
        if diffs:
            m, lo, hi = mean_ci(diffs)
            print(f"  gen3 - continuous diff: mean={m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                 f"  (continuous={con_row['asr_excess']:.3f})")
            print("  CI excludes 0?" , not (lo <= 0 <= hi),
                 "-> generation structure", "MATTERS" if not (lo <= 0 <= hi) else "does NOT detectably matter")
    else:
        print(f"  no continuous control found at {con_f}")

print("\nNote: n=3 seeds gives a rough spread, not a tight interval. If the "
     "trajectory CIs are wide relative to the effect, that itself is useful "
     "information -- it tells you how many seeds you actually need.")

# --- v1 vs v2 agreement check ---------------------------------------------
# v2 exists only to add loss logging; training itself should be unchanged.
# If gen-3 values disagree noticeably from the original run, something in the
# harness moved between runs and that needs chasing down before trusting either.
V1_OUT = os.path.expanduser("~/bdsurvive_runs/lineage_seeds")
print("\n=== v1 vs v2 AGREEMENT CHECK (should closely match) ===")
for tag in TAGS:
    for s in SEEDS:
        f1 = os.path.join(V1_OUT, f"{tag}-seed{s}", "generations.jsonl")
        f2 = os.path.join(OUT, f"{tag}-seed{s}", "generations.jsonl")
        if not (os.path.exists(f1) and os.path.exists(f2)):
            continue
        r1 = {json.loads(l)["generation"]: json.loads(l)["asr_excess"] for l in open(f1)}
        r2 = {json.loads(l)["generation"]: json.loads(l)["asr_excess"] for l in open(f2)}
        g3_diff = abs(r1.get(3, float("nan")) - r2.get(3, float("nan")))
        flag = "  <- DISAGREEMENT, investigate" if g3_diff > 0.03 else ""
        print(f"  {tag}-seed{s}: v1 gen3={r1.get(3):.3f}  v2 gen3={r2.get(3):.3f}"
             f"  diff={g3_diff:.3f}{flag}")