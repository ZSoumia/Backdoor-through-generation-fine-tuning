# bdsurvive — feasibility study

Does a backdoor planted in a foundation model survive benign, repeated downstream fine-tuning? This repo studies backdoor persistence across multi-generation LoRA adaptation lineages, organized around a mechanism-based taxonomy rather than named attacks.

Scope note: the fine-tuning (LoRA) arm is the active, validated part of this repo. 

## Research questions

**RQ1** — which backdoor mechanisms survive one real transfer step?

**RQ2** — among survivors, what is their lineage depth (how many generations before extinction)?

**RQ3** — why does a survivor die where it dies (data distribution, dose, adapter-reset structure, or the adaptation operator itself)?

**RQ4** — what property of a mechanism (trigger type, parameter placement, expression/leakage) predicts persistence?



## Setup
```
pip install -r requirements.txt
python tests/test_units.py      # CPU-only unit tests, run before anything else
```
Requires a GPU (all pilots were run on Lightning AI Studios, L4/A10G class). Set HF_TOKEN in the environment to avoid HuggingFace rate limits on dataset/model downloads.


## Repo layout
 
```
bdsurvive/
  config.py           -> dataclasses: PlantConfig, DeriveConfig, CleanRefConfig, Paths
  data.py             -> content-trigger injection (rare/syntactic/semantic), load_task
  positional.py       -> MetaBackdoor length-trigger mechanism (separate from data.py
                         because it needs the tokenizer, not just string ops)
  plant.py            -> the planting procedure (produces a "parent" checkpoint)
  shards.py           -> deterministic disjoint data partitioning for lineages
  lineage.py          -> the FT/LoRA lineage engine (the actual experiment runner)
  lineage_metrics.py  -> drift/alignment/extinction measurement primitives
  metrics.py          -> now just _predict (a thin batched-inference helper)
 
scripts/
  lineage_runner.py                runs every generational experiment; shape comes from flags
  pilot0_metabackdoor_control.py   positive control: reproduces the MetaBackdoor mechanism
  check_ids.py                     prints registry parent ids vs what is on disk
  aggregate_seed_results.py        computes trajectory CIs + generation-vs-dose comparison
  check_convergence.py             checks whether training converged within its step budget

configs/
  mechanisms.yaml                  the mechanism registry lineage_runner.py reads
```

The newer registry + YAML harness lives alongside it (full layout in
[ARCHITECTURE.md](ARCHITECTURE.md)):

```
bdsurvive/
  core/        config dataclasses, registry, grid validation
  triggers/    lexical, semantic, positional, syntactic (HiddenKiller)
  attacks/     data poisoning (Trigger + label flip)
  finetune/    lora, bitfit, full
  eval/        behavioral rates, drift, alignment, extinction
  cli.py       plan / list / run
configs/study/                     study YAMLs (one per section)
scripts/paraphrase_syntactic.py    offline paraphrase table for the syntactic trigger
```
 


 
### How to run
 

**Fastest path — use `scripts/lineage_runner.py`.** This is the current,
consolidated CLI tool and the recommended entry point for any lineage
experiment; it replaces writing a new script for every generation-count/seed
combination and reuses cached parents automatically.
 
```bash
# Reproduce the original 3-generation baseline (both mechanisms, seed 0):
python scripts/lineage_runner.py --generations 3 --seeds 0 --mechanisms all
 
# Seed-replication (3 seeds, for confidence intervals):
python scripts/lineage_runner.py --generations 3 --seeds 0,1,2 \
    --mechanisms rare,semantic --tag seeds_v2
 
# Extend to 9 generations:
python scripts/lineage_runner.py --generations 9 --seeds 0 \
    --mechanisms rare,semantic --tag g9
 
# 9 generations across all 3 seeds in one call:
python scripts/lineage_runner.py --generations 9 --seeds 0,1,2 \
    --mechanisms rare,semantic --tag g9_seeds
 
# Also run the budget-matched continuous control alongside each lineage:
python scripts/lineage_runner.py --generations 3 --seeds 0,1,2 \
    --mechanisms rare,semantic --continuous
```
 
Output always lands at `~/bdsurvive_runs/lineage_<tag>/<mechanism>-seed<seed>/`,
so different `--tag` values never collide, and reruns of the same tag resume
rather than restart. Positional is deliberately excluded from
`--mechanisms all` until it's re-planted at matched utility — passing
`--mechanisms positional` raises an explicit error rather than running on the
confounded parent. Run `python scripts/lineage_runner.py --help` for the full
option list (steps per generation, batch size, rank, etc.).
 
Always launch as a detached job and verify it actually started:
```bash
PYTHONPATH=. nohup python scripts/lineage_runner.py --generations 9 \
    --seeds 0 --mechanisms rare,semantic --tag g9 > g9.log 2>&1 &
sleep 10 && tail -30 g9.log
```
 
**Manual / low-level path** — useful for one-off scripts or understanding
what `lineage_runner.py` does under the hood:
 
```bash
# 1. Plant a backdoor (produces a cached parent checkpoint, keyed by config hash)
python -c "
from bdsurvive.config import PlantConfig
from bdsurvive.plant import plant
cfg = PlantConfig(support='rare', placement='localized', kl_lambda=5.0, ...)
plant(cfg, out_dir='...', device='cuda')
"
 
# 2. Run a fine-tuning lineage on that parent
python -c "
from bdsurvive.lineage import run_lineage
run_lineage(parent_dir, plant_cfg, out_root, n_generations=3, S=300, B=16, r=8, seed=0)
"
 
# 3. (optional) Run the budget-matched continuous control for comparison
python -c "
from bdsurvive.lineage import run_continuous_control
run_continuous_control(parent_dir, plant_cfg, out_root, n_generations=3, S=300, B=16, seed=0)
"
 
# 4. Analyze
python scripts/aggregate_seed_results.py     # CIs across seeds
python scripts/check_convergence.py          # was training actually converged?
```
 
In practice, drive everything through `lineage_runner.py` rather than writing
calls by hand — it handles parent existence, planting, output paths, resume
and printing. A new mechanism is an entry in `configs/mechanisms.yaml`; a new
experiment shape is a different combination of flags.
 
**Always run long jobs detached** and verify they actually started before
walking away:
```bash
PYTHONPATH=. nohup python scripts/your_script.py > run.log 2>&1 &
sleep 10 && tail -30 run.log
```
 


### Registry harness (experimental)

Study YAMLs in `configs/study/` expand into a manifest of cells, and each cell
runs on its own with `run --index`. (`slurm/array.sbatch` drives
`lineage_runner.py`, not the manifest.) Not yet trusted for Section 1 results;
the legacy runner above is the canonical path.

```bash
python -m bdsurvive.cli plan --study configs/study/section1_grid.yaml --dry-run   # preview cells
python -m bdsurvive.cli plan --study configs/study/section1_grid.yaml --out manifest.jsonl
python -m bdsurvive.cli list --manifest manifest.jsonl
python -m bdsurvive.cli run  --manifest manifest.jsonl --index 0
```

**Syntactic trigger (HiddenKiller): build the paraphrase table first.**
Training never paraphrases; `triggers/syntactic.py` looks texts up in a JSONL
table that `scripts/paraphrase_syntactic.py` builds once with SCPN and stanza.
The script needs its own Python 3.11 environment (OpenAttack does not import on
3.12+) and network access, so run it off-cluster, ideally on a GPU (the full
table takes about 25 hours on CPU), and copy the table to the path in
`configs/study/section1_syntactic.yaml`. Setup and usage are in the script's
docstring.

---
 
## Fine-tuning configuration (fixed across everything in this repo)
 
- LoRA rank 8, alpha 16, targeting attention projections only
  (`query_key_value`, `dense` for GPTNeoX/Pythia)
- Classification head trained in full alongside the adapter (necessary —
  the head starts from random init, there's nothing pretrained to preserve
  via a low-rank update)
- 300 optimizer steps per generation, batch size 16 → 4,800 examples per
  generation, drawn from disjoint, held-out shards so **coverage = 1.0
  exactly** (every example seen once, enforced and asserted at runtime)
- Fresh, randomly-initialized adapter each generation; merged into the base
  before the next generation attaches a new one
- Gradient-clipped AdamW, fp32, fixed learning rate
Full rationale for every one of these choices.
 
---
 
