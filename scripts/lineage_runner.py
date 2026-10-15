"""Lineage runner -- one entry point for every generational experiment.

Operator:
    M_{g+1} = T_LoRA(M_g; D_g, S, B_eff, r, lr)
  fresh adapter each generation, merged into the base before the next.

Data:
    |D_g| = S * B_eff, coverage C = 1 (enforced and asserted in lineage.py),
    D_i mutually disjoint, all disjoint from the planting block, same
    distribution.

Mechanisms come from configs/mechanisms.yaml. Experiment shape comes from the
flags below. Neither requires editing this file.

Examples
--------
# three generations, three seeds, both mechanisms
    python scripts/lineage_runner.py --generations 3 --seeds 0,1,2 \
        --mechanisms all

# nine generations, reusing parents already on disk
    python scripts/lineage_runner.py --generations 9 --seeds 0,1,2 \
        --mechanisms rare,semantic --tag g9

# plant any missing parent instead of skipping it
    python scripts/lineage_runner.py --generations 3 --seeds 0 \
        --mechanisms all --plant --tag fresh

# rank as a separate survival curve
    python scripts/lineage_runner.py --generations 3 --seeds 0 \
        --mechanisms all --rank 16 --tag r16

# budget-matched continuous control alongside each lineage
    python scripts/lineage_runner.py --generations 3 --seeds 0,1,2 \
        --mechanisms all --continuous

Output layout: $BDSURVIVE_ROOT/lineage_<tag>/<mechanism>-seed<seed>/
(plus .../<mechanism>-seed<seed>-continuous/ when --continuous is set), so
different --tag values never collide, and rerunning the same tag resumes
rather than restarting (it checks the generations.jsonl line count).
"""
import argparse
import json
import os

import yaml

from bdsurvive.config import CleanRefConfig, Paths, PlantConfig
from bdsurvive import lineage as L
from bdsurvive import runner as R

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MECHANISMS = os.path.join(REPO_ROOT, "configs", "mechanisms.yaml")


def load_mechanisms(path):
    """Read the registry file into {name: (tag, PlantConfig)}.

    Values are passed through verbatim. PlantConfig.id hashes every field, so
    the ids these produce must match the parents already on disk for cached
    parents to be reused.
    """
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    if not raw:
        raise ValueError(f"no mechanisms defined in {path}")
    registry = {}
    for name, entry in raw.items():
        if "plant" not in entry:
            raise ValueError(f"mechanism '{name}' has no 'plant' block")
        tag = entry.get("tag", name)
        registry[name] = (tag, PlantConfig(**entry["plant"]))
    return registry


def resolve_mechanisms(registry, spec):
    """'all' or a comma-separated list of registry names -> list of names."""
    if spec == "all":
        return list(registry)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in registry]
    if unknown:
        raise ValueError(f"unknown mechanism(s) {unknown}; "
                         f"choose from {list(registry)}")
    return names


def ensure_planted(pc, paths, device):
    """Plant pc if it is not already on disk, and return its directory.

    Follows runner.run_cell's ordering: a recipe-matched clean reference
    first when the expression term needs one, then the parent. CleanRefConfig
    mirrors pc's own training fields so a cached reference is reused.
    clean_ref_path is excluded from PlantConfig.id, so setting it here cannot
    change the parent id.
    """
    pdir = os.path.join(paths.parents, pc.id)
    if os.path.exists(os.path.join(pdir, "config.json")):
        return pdir

    if pc.kl_lambda != 0.0 and pc.clean_ref_path is None:
        cr = CleanRefConfig(
            base_model=pc.base_model, task=pc.task, num_labels=pc.num_labels,
            lr=pc.lr, steps=pc.steps, batch_size=pc.batch_size,
            max_len=pc.max_len, seed=pc.seed)
        print(f"  clean reference: {cr.id}")
        pc.clean_ref_path = R.ensure_clean_ref(cr, paths, device)

    print(f"  planting: {pc.id}")
    return R.ensure_parent(pc, paths, device)


def completed_generations(out_root):
    """How many generation rows a run directory already holds."""
    gen_file = os.path.join(out_root, "generations.jsonl")
    if not os.path.exists(gen_file):
        return 0
    with open(gen_file) as fh:
        return sum(1 for line in fh if line.strip())


def run_one(pdir, pc, out_root, args, seed):
    """One lineage chain, skipped when already complete."""
    n_done = completed_generations(out_root)
    if n_done >= args.generations + 1:      # +1 for generation 0, the parent
        print(f"SKIP (already complete): {os.path.basename(out_root)}")
        return
    if n_done:
        print(f"  ({n_done} generation rows present, re-running the chain)")

    L.run_lineage(pdir, pc, out_root, device=args.device,
                  n_generations=args.generations, S=args.S, B=args.B,
                  lora_r=args.rank, max_len=args.max_len, seed=seed)

    loss_files = [f for f in os.listdir(out_root) if f.startswith("loss_g")]
    print(f"  loss curves captured: {len(loss_files)} generations")


def run_control(pdir, pc, cont_root, args, seed):
    """Budget-matched continuous control: same total steps, one adapter."""
    if completed_generations(cont_root):
        print(f"SKIP (continuous already complete): "
              f"{os.path.basename(cont_root)}")
        return
    L.run_continuous_control(pdir, pc, cont_root, device=args.device,
                             n_generations=args.generations, S=args.S,
                             B=args.B, lora_r=args.rank,
                             max_len=args.max_len, seed=seed)


def print_trajectories(out_dir, registry, mech_names, seeds, tag):
    print(f"\n=== TRAJECTORY SHAPES (tag={tag}) ===")
    for name in mech_names:
        mech_tag, _ = registry[name]
        for seed in seeds:
            run_tag = f"{mech_tag}-seed{seed}"
            gen_file = os.path.join(out_dir, run_tag, "generations.jsonl")
            if not os.path.exists(gen_file):
                continue
            with open(gen_file) as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
            rows.sort(key=lambda r: r["generation"])
            traj = " -> ".join(f"{r['asr_excess']:.3f}" for r in rows)
            ext = next((r["generation"] for r in rows if r["extinct"]), None)
            print(f"{run_tag:34s} excess: {traj}   extinct@: {ext}")


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, required=True,
                    help="number of lineage generations (e.g. 3, 9)")
    ap.add_argument("--seeds", type=str, default="0",
                    help="comma-separated shard-draw/batch-order seeds, "
                         "e.g. '0,1,2'")
    ap.add_argument("--mechanisms", type=str, default="all",
                    help="comma-separated mechanism names from the "
                         "mechanisms file, or 'all'")
    ap.add_argument("--mechanisms-file", type=str, default=DEFAULT_MECHANISMS,
                    help="YAML file defining the mechanism registry "
                         f"(default: {DEFAULT_MECHANISMS})")
    ap.add_argument("--plant", action="store_true",
                    help="plant a parent that is missing instead of skipping "
                         "it (trains a matched clean reference first when "
                         "kl_lambda != 0)")
    ap.add_argument("--tag", type=str, default=None,
                    help="output subdirectory tag; defaults to "
                         "'g<generations>'")
    ap.add_argument("--continuous", action="store_true",
                    help="also run the budget-matched continuous control for "
                         "each mechanism/seed")
    ap.add_argument("--S", type=int, default=300, help="steps per generation")
    ap.add_argument("--B", type=int, default=16, help="batch size")
    ap.add_argument("--rank", type=int, default=8, help="LoRA rank")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    registry = load_mechanisms(args.mechanisms_file)
    mech_names = resolve_mechanisms(registry, args.mechanisms)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    tag = args.tag or f"g{args.generations}"

    paths = Paths()
    out_dir = os.path.join(paths.root, f"lineage_{tag}")

    print(f"Config: generations={args.generations} seeds={seeds} "
          f"mechanisms={mech_names} tag={tag} S={args.S} B={args.B} "
          f"rank={args.rank} plant={args.plant} "
          f"continuous={args.continuous}")
    print(f"Mechanisms file: {args.mechanisms_file}")
    print(f"Output root: {out_dir}")

    for name in mech_names:
        mech_tag, pc = registry[name]
        pdir = os.path.join(paths.parents, pc.id)

        if not os.path.exists(os.path.join(pdir, "config.json")):
            if not args.plant:
                print(f"\nSKIP (parent not planted): {pc.id}")
                print("  pass --plant to plant it, or plant it explicitly.")
                continue
            print(f"\n=== PLANTING: {mech_tag} ===")
            pdir = ensure_planted(pc, paths, args.device)

        for seed in seeds:
            run_tag = f"{mech_tag}-seed{seed}"
            out_root = os.path.join(out_dir, run_tag)
            print(f"\n=== LINEAGE: {run_tag} ===")
            run_one(pdir, pc, out_root, args, seed)

            if args.continuous:
                print(f"\n=== CONTINUOUS CONTROL: {run_tag} ===")
                run_control(pdir, pc, out_root + "-continuous", args, seed)

    print_trajectories(out_dir, registry, mech_names, seeds, tag)


if __name__ == "__main__":
    main()
