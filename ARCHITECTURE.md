# bdsurvive — Architecture

This document describes the **active study-grid execution pipeline** for experiments on backdoor persistence through successive benign fine-tuning generations. For setup and runnable commands, see [README.md](README.md).

The project has **one experiment specification and dispatch system**, with two execution arms:

- **Generational (`run`):** plant the backdoor in G0; train G1 through GN from the preceding checkpoint, with a fresh downstream fine-tuning session at each generation.
- **Continuous (`continuous`):** start from the **same saved G0** and process the same downstream shards and batch sequence in a single uninterrupted fine-tuning session. Evaluate at the budgets corresponding to G1 through GN.

Both arms are addressed by the **same manifest row and cell identity**. They write results to different locations.

## 1. Control flow

```text
configs/study/*.yaml
    |
    | bdsurvive.cli plan
    v
bdsurvive/plan.py
    |  expand model × attack × finetune × seed
    |  validate each cell; assign stable index and cell_id
    v
manifests/<study>.jsonl                 (one row = one full lineage)
    |
    | slurm/grid_array.sbatch (SLURM_ARRAY_TASK_ID selects row)
    | or direct CLI invocation
    v
bdsurvive/cli.py
    |-- run ----------> run.execute_row(...)
    |                       |
    |                       v
    |                  GenerationChain.run()
    |                       |
    |                       v
    |                  steps.training_step()  for G0, G1, ... GN
    |
    `-- continuous ---> run.execute_continuous_row(...)
                            |
                            v
                       steps.train_continuous_control()
                            |
                            v
                       saved G0 + one continuous training loop
    |
    v
<output_root>/runs_v3/<cell_id>/
```

**The planner does not train models.** It converts a study specification into fixed, numbered execution records. The compute job runs exactly one selected record; the runner, rather than SLURM, manages the generations inside that record.

## 2. Responsibilities by module

| Module | Responsibility |
| --- | --- |
| `configs/study/*.yaml` | Scientific specification: model, attack, downstream method, seeds, data, budgets, propagation and output location. |
| `bdsurvive/core/config.py` | Typed experiment specifications, serialization, flat result metadata and `cell_id`. |
| `bdsurvive/core/validation.py` | Cell compatibility checks and execution statuses. |
| `bdsurvive/core/registry.py` | Named constructors for triggers, attacks, fine-tuning methods and scorers. |
| `bdsurvive/plan.py` | YAML loading, Cartesian-product expansion, validation, JSONL manifest read/write. |
| `bdsurvive/cli.py` | CLI dispatch for `plan`, `list`, `run` and `continuous`. |
| `slurm/grid_array.sbatch` | Leonardo job resources, environment, manifest path and `RUN_MODE` selection. |
| `bdsurvive/run.py` | Reconstruct configuration, validate identity, select run directory, orchestrate execution and write cell status. |
| `bdsurvive/generation.py` | G0–GN progression, parent selection, state/restart, generation records and checkpoint lifecycle. |
| `bdsurvive/steps.py` | Model/data loading, planting, downstream training, evaluation and continuous-control execution. |
| `bdsurvive/shards.py` | Deterministic planting reservation and disjoint downstream dataset allocations. |
| `bdsurvive/train_loop.py` | Shared supervised optimizer loop, deterministic batches, explicit continuous batch order and evaluation callback. |
| `bdsurvive/attacks/` | Backdoor installation mechanisms. |
| `bdsurvive/triggers/` | Trigger creation and matched trigger/control text construction. |
| `bdsurvive/finetune/` | Full, LoRA and selective adaptation implementations. |
| `bdsurvive/eval/` | Behavioral success, clean utility, uncertainty, drift and alignment-related measurements. |
| `bdsurvive/checkpoint.py` | Checkpoint paths, promotion, state serialization, retention and success markers. |
| `tests/` | Unit and integration regression checks. |

Shared modules such as `models.py`, `data.py`, `train_loop.py` and `checkpoint.py` support the active pipeline and remain in `bdsurvive/`.

## 3. Configuration, manifest and cell identity

### Study YAML

A study under `configs/study/` specifies arrays for `model`, `attack`, `finetune` and `seed`, together with `shared.training`, `shared.data`, `shared.propagation` and `shared.output_root`.

Example of the propagation-related settings used for the four-generation lexical pilot:

```yaml
shared:
  training:
    steps: 300
    batch_size: 16
  propagation:
    generations: 4
    retention: all
  output_root: /absolute/leonardo/path/bdsurvive_runs
```

This is only a fragment; an executable study also needs the chosen model, attack and fine-tuning specifications. The attack's **planting steps** and the downstream **per-generation training steps** are separate budgets.

### Planning

```bash
python -m bdsurvive.cli plan \
  --study configs/study/sok_first_pilot.yaml --dry-run

mkdir -p manifests
python -m bdsurvive.cli plan \
  --study configs/study/sok_first_pilot.yaml \
  --out manifests/lexical_pilot_g4.jsonl

python -m bdsurvive.cli list \
  --manifest manifests/lexical_pilot_g4.jsonl
```

`plan.py` expands `model × attack × finetune × seed` and creates **one JSON object per cell**, including its index, `cell_id`, full configuration, validity (`grid_status`) and, where applicable, an incompatibility reason. An incompatible cell is **not applicable**, not an observed attack failure.

A row covers **G0 through GN**, not one generation. For a one-cell pilot, SLURM array index `0` selects the entire chain. A changed study should receive a **new manifest filename**; never overwrite a manifest while submitted jobs may be reading it.

### Identity and paths

`cell_id` is derived deterministically from the experiment configuration. In the current run-identity convention, `output_root` and the requested propagation horizon (`generations`) do not participate in identity. That allows a compatible G0–G1 lineage to be extended through G4 without producing a new parent. Other experimental changes should produce new identities.

The active output namespace is:

```text
<output_root>/runs_v3/<cell_id>/
```

**Manifest storage** (small JSONL input under `manifests/`) is distinct from **output storage** (checkpoints and metrics under `shared.output_root`). The directory actually used should always be checked through `run_dir_for(cfg)` when diagnosing a job.

## 4. Generational execution: G0 through GN

```text
Pretrained model
    |
    v
G0: install attack on reserved planting examples
    |  evaluate and save checkpoint_gen000
    v
G1: benign train on shard 1 from G0 checkpoint
    |  evaluate and save checkpoint_gen001
    v
G2: benign train on shard 2 from G1 checkpoint
    |  evaluate and save checkpoint_gen002
    v
...
    v
GN: benign train on shard N from G(N-1) checkpoint
```

`GenerationChain` invokes `steps.training_step()` in generation order. G0 is **planting only** in the current pipeline: benign lineage training begins with G1. Each downstream generation constructs a new fine-tuning session from its parent's saved model.

The dataset allocation is deterministic. When the lineage task equals the planting task, a reserved planting pool is kept disjoint from downstream shards. Shard 1 belongs to G1, shard 2 to G2, and so on. The same per-generation batch schedule is reused when constructing the continuous-control ordering.

Each stage records behavioral measurements and clean-task utility; downstream generations also calculate available drift/alignment diagnostics. The exact set of fields is defined by the evaluation/result dataclasses, not by the YAML planner.

### Resume and retention

The generational run keeps `state.json`, a last-completed generation, the current checkpoint and `generations.jsonl`. A resubmission with the **same cell identity** can continue from the next generation instead of replanting G0.

Check completion using **both** the requested horizon and the recorded last generation. A `SUCCESS` file left by a completed shorter run does not prove that a subsequently requested longer run has finished.

Checkpoint retention (`all`, `rolling` or `selected`) controls which model directories remain after successful stages. Use **`retention: all`** when experimenting with all generations or when the continuous control needs the planted G0. Do not manually remove checkpoint/state files to trigger continuation.

## 5. Continuous-control execution: C1 through CN

The continuous arm is dispatched **after a complete generational run** and loads the exact `checkpoint_gen000` saved under its corresponding cell.

```text
                    same planted G0
                          |
                          v
           One LoRA adapter + one AdamW optimizer
                          |
   shard 1 batches -- evaluate C1 after S updates
                          |
   shard 2 batches -- evaluate C2 after 2S updates
                          |
   shard 3 batches -- evaluate C3 after 3S updates
                          |
             ...          |
   shard N batches -- evaluate CN after NS updates
                          |
                          v
              Save continuous checkpoint
```

Here `S = cfg.training.steps` and total updates equal `N × S`.

`train_loop.make_continuous_batch_order()` maps each generation's deterministic local batches into the concatenated downstream training set. `steps.train_continuous_control()` wraps the G0 model once, calls `train_supervised()` **once** with the explicit sequence, and merges the adapter only when training has finished.

`train_supervised(on_step_end=...)` invokes a boundary evaluation callback after optimizer updates. The callback collects C1–CN metrics while preserving training mode; it must not merge the live adapter, recreate the optimizer or perform optimization itself. Intermediate measurements are recorded at matched cumulative update budgets.

The final continuous checkpoint and its boundary records live in an **independent `continuous/` namespace**. The generational chain and original G0 remain untouched. A failed continuous run currently restarts from G0; it does **not** restore an intermediate continuous optimizer state.

### Fair-comparison limits

The arms use the same planted G0, downstream shards, batch order and cumulative update counts. A matched batch sequence is necessary but **does not alone guarantee identical initial LoRA adapter parameters, stochastic dropout trajectories or optimizer-state histories**. Check these factors and repeat experiments across seeds before attributing small ASR differences specifically to resets.

## 6. Result layout and contracts

With full checkpoint retention, a completed four-generation cell resembles:

```text
<output_root>/runs_v3/<cell_id>/
├── state.json
├── generations.jsonl
├── result.json
├── SUCCESS
├── checkpoint_gen000/
├── checkpoint_gen001/
├── checkpoint_gen002/
├── checkpoint_gen003/
├── checkpoint_gen004/
└── continuous/
    ├── result.json
    ├── SUCCESS
    └── checkpoint/
        ├── continuous_boundaries.jsonl
        └── ...model and tokenizer files...
```

- `generations.jsonl`: generational stage metrics; for `generations: 4`, expect G0, G1, G2, G3, G4.
- `state.json`: authoritative generational progress and parent checkpoint.
- Top-level `SUCCESS`: generational completion **at the recorded horizon**.
- `continuous/checkpoint/continuous_boundaries.jsonl`: C1–C4 evaluation records, one per downstream boundary.
- `continuous/SUCCESS`: independent continuous completion marker.

The continuous boundary JSONL is written after the uninterrupted training call and checkpoint save. It therefore does not currently provide mid-run crash recovery.

## 7. Leonardo execution

From the repository root, after generation of the manifest:

```bash
# First: generational arm.
sbatch --array=0 \
  --export=ALL,RUN_MODE=run,MANIFEST=manifests/lexical_pilot_g4.jsonl \
  slurm/grid_array.sbatch

# After G0–G4 is complete and G0 is retained: continuous arm.
sbatch --array=0 \
  --export=ALL,RUN_MODE=continuous,MANIFEST=manifests/lexical_pilot_g4.jsonl \
  slurm/grid_array.sbatch
```

Direct execution, where an appropriate GPU is available:

```bash
python -m bdsurvive.cli run \
  --manifest manifests/lexical_pilot_g4.jsonl --index 0
python -m bdsurvive.cli continuous \
  --manifest manifests/lexical_pilot_g4.jsonl --index 0
```

Leonardo compute nodes may use an offline Hugging Face cache; the required model, tokenizer and dataset must already be cached. A missing cache is an environment/data problem, not a reason to silently substitute another pretrained model or dataset.

## 8. Extension points and safeguards

New scientific components belong in their respective registries:

- **Trigger:** implement under `bdsurvive/triggers/` and register with `TRIGGERS`.
- **Installation:** implement under `bdsurvive/attacks/` and register with `ATTACKS`.
- **Downstream method:** implement under `bdsurvive/finetune/` and register with `FINETUNE`.
- **Scorer:** implement under `bdsurvive/eval/` and register with `SCORERS`.
- **Study:** add or edit YAML under `configs/study/`, then regenerate the manifest.

The attack specification preserves two complementary taxonomy layers:

| Layer | Fields | Question |
| --- | --- | --- |
| Literature-facing | `trigger_type`, `installation` | What kind of backdoor and installation method? |
| Mechanism-facing | `placement`, `expression` | Where and under what training constraint was the behavior installed? |

Scientific safeguards still requiring attention before a large sweep include qualification of the planted G0, matched clean-model baselines, parameter-support audits in the new path, multi-seed uncertainty, fairness of adapter initialization, and offline preprocessing validation for BITE/syntactic triggers. A one-seed successful run validates the execution path, **not** a general mechanism-level conclusion.

## 9. Fast validation ladder

```bash
# Source and shell syntax.
python -m py_compile bdsurvive/cli.py bdsurvive/run.py \
  bdsurvive/generation.py bdsurvive/steps.py bdsurvive/train_loop.py
bash -n slurm/grid_array.sbatch

# Local regressions.
python -m pytest -q tests/test_units.py

# Planning (no GPU).
python -m bdsurvive.cli plan \
  --study configs/study/sok_first_pilot.yaml --dry-run
python -m bdsurvive.cli list \
  --manifest manifests/lexical_pilot_g4.jsonl

# Verify execution on Leonardo only after inspecting the manifest,
# output identity, retained G0 and requested generation horizon.
```

**Design rule:** active runtime modules and their configurations must not import or depend on archived code. The supported experiment path is the study-grid planner and its `run` / `continuous` executors.
