"""PILOT 1 --- three-generation LoRA lineage (phenomenon discovery).

Objective, deliberately modest:
    Under a controlled one-pass, fixed-budget LoRA lineage, what SHAPES of
    persistence trajectory occur across backdoor mechanisms?

We are NOT explaining them yet. Cliff-edge (99->12->1->0) versus plateau
(98->91->83->79) is itself the finding; RQ3 asks why.

Frozen spec:
    M_{g+1} = T_LoRA(M_g; D_g, S=300, B_eff=16, r=8, lr fixed)
    |D_g| = 4800  ->  coverage exactly 1, asserted at run time
    D_g mutually disjoint, disjoint from the planting block, same distribution
    fresh adapter per generation, merged before the next

Reuses cached parents --- no re-planting. Run pilot0/pilot2 first so they exist.
"""
import os, json
from bdsurvive.config import Paths, PlantConfig
from bdsurvive import lineage as L

paths = Paths()
OUT = os.path.expanduser("~/bdsurvive_runs/lineage")

# The parents to run. These must already exist in paths.parents (cached from
# earlier pilots) --- the ids are derived from the configs, so keep these
# identical to how they were originally planted.
PARENTS = [
    PlantConfig(support="rare", placement="localized", kl_lambda=5.0,
                localized_layers=(6, 8), poison_rate=0.05, steps=1500, seed=0),
    PlantConfig(support="semantic", placement="diffuse", kl_lambda=0.0,
                poison_rate=0.08, steps=1500, seed=0),
    PlantConfig(support="positional", placement="diffuse", kl_lambda=0.0,
                trigger_tau=64, trigger_family="threshold", paper_recipe=True,
                n_clean=3000, n_poison=300, epochs=3, paper_lr=5e-5,
                max_len=96, seed=0),
]

for pc in PARENTS:
    pdir = os.path.join(paths.parents, pc.id)
    if not os.path.exists(os.path.join(pdir, "config.json")):
        print(f"SKIP (parent not planted): {pc.id}")
        print("   run pilot0/pilot2 first, or plant it explicitly.")
        continue

    tag = f"{pc.support}-{pc.placement}-kl{pc.kl_lambda:g}"
    print(f"\n=== LINEAGE: {tag} ===")
    L.run_lineage(pdir, pc, os.path.join(OUT, tag), device="cuda",
                  n_generations=3, S=300, B=16, lora_r=8, max_len=96, seed=0)

    print(f"\n=== CONTINUOUS CONTROL: {tag} ===")
    L.run_continuous_control(pdir, pc, os.path.join(OUT, tag + "-continuous"),
                             device="cuda", n_generations=3, S=300, B=16,
                             lora_r=8, max_len=96, seed=0)

print("\n=== TRAJECTORY SHAPES ===")
for pc in PARENTS:
    tag = f"{pc.support}-{pc.placement}-kl{pc.kl_lambda:g}"
    f = os.path.join(OUT, tag, "generations.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f)]
    traj = " -> ".join(f"{r['asr_excess']:.2f}" for r in rows)
    ext = next((r["generation"] for r in rows if r["extinct"]), None)
    print(f"{tag:34s} excess: {traj}   extinct at gen: {ext}")
    print(f"{'':34s} drift : " +
          " -> ".join(f"{r['weight_drift_cum']:.1f}" for r in rows))
    print(f"{'':34s} align : " +
          " -> ".join(f"{r['bd_alignment']:.3f}" if r['bd_alignment'] == r['bd_alignment']
                      else "  n/a" for r in rows))