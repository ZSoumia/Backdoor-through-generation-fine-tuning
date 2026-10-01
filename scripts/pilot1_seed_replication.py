"""Seed replication for the 3-generation lineage (rare + semantic only,
positional excluded until its parent is re-planted at matched utility).

RE-RUN (v2): identical to the original seed-replication run, but on the
patched lineage.py that logs per-generation loss curves. The original run's
ASR/drift/alignment numbers were already solid and don't need re-verifying for
their own sake -- what was MISSING was any way to check whether S=300 steps
per generation had actually converged. This run adds that, and doubles as a
second independent confirmation of the plateau (extra seed-robustness, free).

Writes to a NEW directory (lineage_seeds_v2) rather than overwriting the
original -- compare the two runs' ASR trajectories as a sanity check first
(they should closely agree, since nothing about training changed, only the
logging); if they don't agree, something is off in the harness between runs
and that needs chasing down before trusting either.

Purpose / priority ordering unchanged:
  1. seed replication + convergence check (this script)
  2. positional re-plant at matched utility
  3. dose-vs-repetition split
  4. extended lineage to g=9+

Does NOT touch KD (out of scope).
Reuses the already-planted parents on disk. No new planting.
"""
import os
from bdsurvive.config import Paths, PlantConfig
from bdsurvive import lineage as L

paths = Paths()
OUT = os.path.expanduser("~/bdsurvive_runs/lineage_seeds_v2")

SEEDS = [0, 1, 2]   # shard-draw + batch-order seed, varied together

PARENTS = [
    ("rare-localized-kl5", PlantConfig(support="rare", placement="localized",
        kl_lambda=5.0, localized_layers=(6, 8), poison_rate=0.05, steps=1500, seed=0)),
    ("semantic-diffuse-kl0", PlantConfig(support="semantic", placement="diffuse",
        kl_lambda=0.0, poison_rate=0.08, steps=1500, seed=0)),
]

for tag, pc in PARENTS:
    pdir = os.path.join(paths.parents, pc.id)
    if not os.path.exists(os.path.join(pdir, "config.json")):
        print(f"SKIP (parent not planted): {pc.id}")
        continue

    for s in SEEDS:
        run_tag = f"{tag}-seed{s}"
        out_root = os.path.join(OUT, run_tag)
        gen_file = os.path.join(out_root, "generations.jsonl")
        if os.path.exists(gen_file):
            n_done = sum(1 for _ in open(gen_file))
            if n_done >= 4:   # gen 0,1,2,3
                print(f"SKIP (already complete): {run_tag}")
                continue

        print(f"\n=== LINEAGE: {run_tag} ===")
        L.run_lineage(pdir, pc, out_root, device="cuda",
                      n_generations=3, S=300, B=16, lora_r=8, max_len=96,
                      seed=s)

        # sanity check: confirm loss curves actually got written this time
        loss_files = [f for f in os.listdir(out_root) if f.startswith("loss_g")]
        if not loss_files:
            print(f"  WARNING: no loss_g*.json found for {run_tag} -- "
                 f"you're likely still on the pre-patch lineage.py")
        else:
            print(f"  loss curves captured: {sorted(loss_files)}")

print("\nDone. Run scripts/aggregate_seed_results.py (point it at lineage_seeds_v2)")
print("and scripts/check_convergence.py to see the convergence picture.")
