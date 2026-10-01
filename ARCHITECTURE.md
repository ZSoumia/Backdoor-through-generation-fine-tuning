# bdsurvive — architecture

One runner, four study configs. The scientific code knows nothing about
"Section 1/2/3/4"; those are YAML files that pin some axes and sweep others.

## Layout

```
bdsurvive/
├── core/
│   ├── config.py       dataclasses; both taxonomy layers; cell_id hashing
│   ├── registry.py     name -> class maps (no if/elif dispatch anywhere)
│   └── validation.py   GridStatus/RunStatus; N/A is a first-class outcome
├── triggers/           the firing condition
│   ├── base.py         Trigger ABC + build_pairs (matched trigger/control)
│   ├── lexical.py      rare token — BadNets/RIPPLe
│   ├── semantic.py     topic word-set
│   └── positional.py   length/RoPE — MetaBackdoor
├── attacks/            how the behavior gets into the weights
│   ├── base.py         Attack ABC
│   └── data_poisoning.py   composes a Trigger + label flip + placement/expression
├── finetune/           what downstream owners do
│   ├── base.py         FinetuneMethod ABC; support_descriptor()
│   ├── full.py  lora.py  bitfit.py
├── eval/
│   ├── base.py         result dataclasses
│   ├── scorers.py      per-example hit decision (label / string / prefix / secret)
│   ├── behavioral.py   trigger+control rates; task-type dispatch
│   ├── drift.py        weight drift, functional KL (clean and trigger)
│   ├── alignment.py    cos(dTheta, grad L_backdoor)
│   └── extinction.py   bootstrap CI; extinction = not detectably above baseline
├── train_loop.py       the single supervised loop (planting and finetuning)
├── steps.py            one generation: load -> plant -> finetune -> eval -> save
├── generation.py       resumable chains, rolling retention
├── checkpoint.py       atomic promote; state.json; SUCCESS markers
├── plan.py             study YAML -> manifest (with validity)
├── run.py              execute one manifest row
└── cli.py              plan / list / run
```

## Two taxonomy layers

Both land in every result row, so results group either way.

| layer | fields | role |
|---|---|---|
| literature-facing | `trigger_type`, `installation` | how prior work is organized |
| mechanism-facing | `placement`, `expression` | what explains survival |

## Non-negotiables baked in

**N/A is not 0.00.** `plan.py` validates every cell before a GPU is touched.
A positional trigger on a learned-position model is `NOT_APPLICABLE` with a
reason, never a tested failure.

**Chains resume.** `state.json` records the last completed generation.
A killed job resubmits and continues. Never split generations across SLURM
array elements — generation N needs N−1's checkpoint.

**Checkpoints are disposable, results are not.** Generation results are
written before the checkpoint is promoted, and pruning happens last, so a
crash at any point is recoverable.

## Adding things

- new trigger → a file in `triggers/` with `@TRIGGERS.register("name")`
- new FT method → a file in `finetune/` with `@FINETUNE.register("name")`
- new installation → a file in `attacks/` with `@ATTACKS.register("name")`
- new experiment → a YAML in `configs/study/`, never a `.py`

## Test ladder

```bash
python tests/test_units.py                                    # CPU, seconds
python -m bdsurvive.cli plan --study ... --dry-run             # cell count, no GPU
python -m bdsurvive.cli run --manifest m.jsonl --index 0       # one real cell
apptainer exec --nv container/bdsurvive.sif python tests/test_units.py
sbatch --array=0-0 slurm/array.sbatch                          # one SLURM job
sbatch slurm/array.sbatch                                      # full array
```

## HPC

```bash
python -m bdsurvive.cli plan --study configs/study/section1_grid.yaml --out manifest.jsonl
python scripts/prefetch.py --manifest manifest.jsonl     # login node, has internet
apptainer build container/bdsurvive.sif container/bdsurvive.def
MANIFEST=manifest.jsonl sbatch --array=0-26%4 slurm/array.sbatch
```

Compute nodes are offline: prefetch on a login node, then jobs run with
`HF_HUB_OFFLINE=1`.
