"""Consolidated lineage runner -- replaces pilot1_lineage3.py and
pilot1_seed_replication.py (and the not-yet-created pilot1_extend_g9.py)
with one CLI-driven tool.


Examples
--------
# Reproduce the original 3-generation baseline (both mechanisms, seed 0):
    python scripts/lineage_runner.py --generations 3 --seeds 0 --mechanisms all

# Reproduce the seed-replication run (3 seeds):
    python scripts/lineage_runner.py --generations 3 --seeds 0,1,2 \\
        --mechanisms rare,semantic --tag seeds_v2

# g=9 extension, seed 0 (this is the one you asked about last):
    python scripts/lineage_runner.py --generations 9 --seeds 0 \\
        --mechanisms rare,semantic --tag g9

# g=9 across all 3 seeds in one call:
    python scripts/lineage_runner.py --generations 9 --seeds 0,1,2 \\
        --mechanisms rare,semantic --tag g9_seeds

# Also run the budget-matched continuous control alongside each lineage:
    python scripts/lineage_runner.py --generations 3 --seeds 0,1,2 \\
        --mechanisms rare,semantic --continuous

Output layout: ~/bdsurvive_runs/lineage_<tag>/<mechanism>-seed<seed>/
(and .../<mechanism>-seed<seed>-continuous/ if --continuous is set), so runs
with different --tag values never collide, and reruns of the same tag/params
resume rather than restart (checks generations.jsonl line count).
"""
import argparse
import os
from bdsurvive.config import Paths, PlantConfig
from bdsurvive import lineage as L

# ---------------------------------------------------------------------
# Mechanism registry. Values copied VERBATIM from the scripts that
# originally planted these parents -- do not "clean up" field ordering or
# defaults here, since PlantConfig.id is a hash of every field and any
# drift silently produces a different id.
# Positional is deliberately excluded: its parent is confounded by low
# training utility (clean_acc 0.63 vs ~0.91 for the other two) and any
# lineage built on it is provisional -- add it back here once re-planted.
# ---------------------------------------------------------------------
MECHANISM_REGISTRY = {
    "rare": ("rare-localized-kl5", PlantConfig(
        support="rare", placement="localized", kl_lambda=5.0,
        localized_layers=(6, 8), poison_rate=0.05, steps=1500, seed=0)),
    "semantic": ("semantic-diffuse-kl0", PlantConfig(
        support="semantic", placement="diffuse", kl_lambda=0.0,
        poison_rate=0.08, steps=1500, seed=0)),
}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, required=True,
                    help="number of lineage generations (e.g. 3, 9)")
    ap.add_argument("--seeds", type=str, default="0",
                    help="comma-separated shard-draw/batch-order seeds, e.g. '0,1,2'")
    ap.add_argument("--mechanisms", type=str, default="all",
                    help="comma-separated mechanism names from the registry "
                         f"({', '.join(MECHANISM_REGISTRY)}), or 'all'")
    ap.add_argument("--tag", type=str, default=None,
                    help="output subdirectory tag; defaults to 'g<generations>' "
                         "so different generation counts never collide")
    ap.add_argument("--continuous", action="store_true",
                    help="also run the budget-matched continuous control "
                         "(n_generations * S steps, one uninterrupted adapter) "
                         "for each mechanism/seed")
    ap.add_argument("--S", type=int, default=300, help="steps per generation")
    ap.add_argument("--B", type=int, default=16, help="batch size")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    mech_names = (list(MECHANISM_REGISTRY) if args.mechanisms == "all"
                 else args.mechanisms.split(","))
    for m in mech_names:
        if m not in MECHANISM_REGISTRY:
            raise ValueError(f"unknown mechanism '{m}', choose from "
                            f"{list(MECHANISM_REGISTRY)}")
    tag = args.tag or f"g{args.generations}"

    paths = Paths()
    OUT = os.path.expanduser(f"~/bdsurvive_runs/lineage_{tag}")

    print(f"Config: generations={args.generations} seeds={seeds} "
         f"mechanisms={mech_names} tag={tag} continuous={args.continuous}")

    for m in mech_names:
        mech_tag, pc = MECHANISM_REGISTRY[m]
        pdir = os.path.join(paths.parents, pc.id)
        if not os.path.exists(os.path.join(pdir, "config.json")):
            print(f"SKIP (parent not planted): {pc.id}")
            print("  run pilot0/pilot2, or plant it explicitly, first.")
            continue

        for s in seeds:
            run_tag = f"{mech_tag}-seed{s}"
            out_root = os.path.join(OUT, run_tag)
            gen_file = os.path.join(out_root, "generations.jsonl")
            if os.path.exists(gen_file):
                n_done = sum(1 for _ in open(gen_file))
                if n_done >= args.generations + 1:   # +1 for gen0 (the parent)
                    print(f"SKIP (already complete): {run_tag}")
                else:
                    print(f"\n=== LINEAGE: {run_tag} (resuming, {n_done} done) ===")
                    L.run_lineage(pdir, pc, out_root, device=args.device,
                                  n_generations=args.generations, S=args.S,
                                  B=args.B, lora_r=args.rank,
                                  max_len=args.max_len, seed=s)
            else:
                print(f"\n=== LINEAGE: {run_tag} ===")
                L.run_lineage(pdir, pc, out_root, device=args.device,
                              n_generations=args.generations, S=args.S,
                              B=args.B, lora_r=args.rank,
                              max_len=args.max_len, seed=s)

            loss_files = [f for f in os.listdir(out_root) if f.startswith("loss_g")]
            if not loss_files:
                print(f"  WARNING: no loss_g*.json found for {run_tag} -- "
                     f"check lineage.py has the EMA-loss patch")
            else:
                print(f"  loss curves captured: {len(loss_files)} generations")

            if args.continuous:
                cont_root = out_root + "-continuous"
                cont_gen_file = os.path.join(cont_root, "generations.jsonl")
                if os.path.exists(cont_gen_file):
                    print(f"SKIP (continuous already complete): {run_tag}")
                    continue
                print(f"\n=== CONTINUOUS CONTROL: {run_tag} ===")
                L.run_continuous_control(pdir, pc, cont_root, device=args.device,
                                         n_generations=args.generations, S=args.S,
                                         B=args.B, lora_r=args.rank,
                                         max_len=args.max_len, seed=s)

    # trajectory summary
    print(f"\n=== TRAJECTORY SHAPES (tag={tag}) ===")
    for m in mech_names:
        mech_tag, pc = MECHANISM_REGISTRY[m]
        for s in seeds:
            run_tag = f"{mech_tag}-seed{s}"
            f = os.path.join(OUT, run_tag, "generations.jsonl")
            if not os.path.exists(f):
                continue
            import json
            rows = sorted([json.loads(l) for l in open(f)],
                          key=lambda r: r["generation"])
            traj = " -> ".join(f"{r['asr_excess']:.3f}" for r in rows)
            ext = next((r["generation"] for r in rows if r["extinct"]), None)
            print(f"{run_tag:34s} excess: {traj}   extinct@: {ext}")


if __name__ == "__main__":
    main()