"""Lineage runner -- one entry point for every generational experiment.

"""
import argparse
import json
import os
from dataclasses import replace

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

    A recipe-matched clean reference first, then the parent. CleanRefConfig
    mirrors pc's own training fields so a cached reference is reused.
    clean_ref_path is excluded from PlantConfig.id, so setting it here cannot
    change the parent id.

    The reference is ensured for EVERY cell, not only the ones whose loss uses
    it. Two reasons, and the second is the one that bit:

      1. R0 asks whether a cell established the behaviour *under acceptable
         clean performance*, and clean accuracy is not comparable against a
         fixed number across architectures -- a 410m and a 1.7b model do not
         share a ceiling on ag_news. The only meaningful baseline is the same
         model on the same schedule with zero poison, which is exactly this
         reference. Training it only when kl_lambda != 0 left 30 of the 41
         cells with no baseline at all, so half of R0's criterion was
         unmeasurable for them.
      2. It ran AFTER the cached-parent early return, so a cell whose parent
         was already on disk never ensured its reference even when
         kl_lambda != 0.

    One reference serves every cell sharing (model, lr, steps, batch, max_len,
    seed, planting_block), so the 41 cells need 14 of them -- two per model,
    one at max_len 128 for the content cells and one at 96 for the positional
    ones.
    """
    cr = CleanRefConfig(
        base_model=pc.base_model, task=pc.task, num_labels=pc.num_labels,
        lr=pc.lr, steps=pc.steps, batch_size=pc.batch_size,
        max_len=pc.max_len, seed=pc.seed,
        # Must match the parent, or the reference trains on rows the
        # lineage later derives from and the clean-accuracy baseline is
        # contaminated in the direction that flatters the result.
        planting_block=pc.planting_block)
    ref_dir = os.path.join(paths.parents, cr.id)
    if not os.path.exists(os.path.join(ref_dir, "config.json")):
        print(f"  clean reference: {cr.id}")
    if pc.clean_ref_path is None:
        pc.clean_ref_path = R.ensure_clean_ref(cr, paths, device)

    pdir = os.path.join(paths.parents, pc.id)
    if os.path.exists(os.path.join(pdir, "config.json")):
        return pdir

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
                  lora_r=args.rank, max_len=args.max_len, seed=seed,
                  finetune=args.finetune,
                  save_full_checkpoints=args.save_checkpoints)

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
                             max_len=args.max_len, seed=seed,
                             finetune=args.finetune,
                             save_full_checkpoints=args.save_checkpoints)


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
    ap.add_argument("--finetune", type=str, default="lora",
                    help="fine-tuning method from the FINETUNE registry: "
                         "lora, bitfit, full. This is the Section 1 axis -- "
                         "the methods differ in trainable SUPPORT, not just "
                         "cost, which is what makes the overlap hypothesis "
                         "testable.")
    ap.add_argument("--max-len", type=int, default=96)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--plant-seeds", type=str, default=None,
                    help="comma-separated PLANTING seeds. Each produces its "
                         "own parent (PlantConfig.seed is hashed), and each "
                         "parent is then derived under every --seeds value. "
                         "This is the nested design: --plant-seeds measures "
                         "variability in how the backdoor was INSTALLED, "
                         "--seeds measures variability in how it was DERIVED. "
                         "Collapsing them into one number makes the two "
                         "indistinguishable. Default: whatever the mechanism "
                         "file specifies, usually 0.")
    ap.add_argument("--save-checkpoints", action="store_true",
                    help="also write the full fp32 M0/Mfinal models "
                         "(~2x model size per run). Off by default: the "
                         "seven-model grid would need ~630 GB of a 1 TB "
                         "quota, and nothing in the analysis path reads "
                         "them. Use it for a single run you mean to probe.")
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

    # The nested design. Each planting seed is its own parent, because
    # PlantConfig.seed is hashed; each parent is then derived under every
    # derivation seed. Variability across plant seeds is variability in how
    # the backdoor was installed; variability across derivation seeds is
    # variability in how it was eroded. One pooled number cannot separate
    # them, and the claim the paper makes is about the second.
    plant_seeds = ([int(s) for s in args.plant_seeds.split(",") if s.strip()]
                   if args.plant_seeds else [None])
    if args.plant_seeds:
        print(f"Nested seeds: plant={plant_seeds} x derivation={seeds} "
              f"= {len(plant_seeds) * len(seeds)} runs per mechanism")

    for name in mech_names:
        mech_tag, pc_base = registry[name]

        for pseed in plant_seeds:
            pc = replace(pc_base, seed=pseed) if pseed is not None else pc_base
            ptag = f"{mech_tag}-plant{pc.seed}" if args.plant_seeds else mech_tag
            pdir = os.path.join(paths.parents, pc.id)

            if not os.path.exists(os.path.join(pdir, "config.json")):
                if not args.plant:
                    print(f"\nSKIP (parent not planted): {pc.id}")
                    print("  pass --plant to plant it, or plant it explicitly.")
                    continue
                print(f"\n=== PLANTING: {ptag} (plant_seed={pc.seed}) ===")
                pdir = ensure_planted(pc, paths, args.device)

            for seed in seeds:
                run_tag = f"{ptag}-seed{seed}"
                out_root = os.path.join(out_dir, run_tag)
                print(f"\n=== LINEAGE: {run_tag} "
                      f"(plant={pc.seed}, derivation={seed}) ===")
                run_one(pdir, pc, out_root, args, seed)

                if args.continuous:
                    print(f"\n=== CONTINUOUS CONTROL: {run_tag} ===")
                    run_control(pdir, pc, out_root + "-continuous", args, seed)

    print_trajectories(out_dir, registry, mech_names, seeds, tag)


if __name__ == "__main__":
    main()
