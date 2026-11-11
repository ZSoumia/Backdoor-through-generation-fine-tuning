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
Requires a GPU (CUDA) for fine-tuning and some of the scripts.


## How experiments are specified

The source of truth for a study is a YAML file under `configs/study/`, such as:

```text
configs/study/sok_first_pilot.yaml
```

The study contains these axes and shared settings:

| YAML key | Meaning |
| --- | --- |
| `model` | One or more model specifications |
| `attack` | Installation method, trigger, target, poisoning and planting settings |
| `finetune` | Downstream adaptation method, such as LoRA or full fine-tuning |
| `seed` | Seeds to run |
| `shared.training` | Benign optimization steps, batch size, learning rate and sequence length |
| `shared.data` | Planting/lineage tasks and shard configuration |
| `shared.propagation` | Number of downstream generations and checkpoint retention |
| `shared.output_root` | Directory for persistent checkpoints and results |

For the validated lexical pilot, the intended downstream settings are:

```yaml
shared:
  training:
    steps: 300
    batch_size: 16
    lr: 1.0e-4
    max_len: 96
  data:
    planting_task: ag_news
  propagation:
    generations: 4
    retention: all
  output_root: /absolute/path/on/leonardo/bdsurvive_runs
```

The example is a **partial YAML fragment**, not a complete study: retain the `model`, `attack`, `finetune`, and `seed` entries in your actual study file. Planting has its own step budget in the `attack` configuration; it is separate from the benign `shared.training.steps` budget.

Use an **absolute Leonardo path** for `output_root`. Do not assume a shell variable such as `$WORK` will be expanded inside YAML. For experiments that need to reuse G0 and compare all generations, set `retention: all`.

### Why a manifest?

`bdsurvive/plan.py` expands the Cartesian product of the study's models, attacks, fine-tuning methods and seeds. It validates each resulting configuration and writes **one JSON object per experiment cell** into a JSONL manifest.

A **manifest row represents an entire lineage**, not an individual generation. For example, one model × one attack × one method × one seed produces one row, which executes G0 through G4 when `generations: 4`.

The manifest gives SLURM stable row indices, records the exact expanded configuration, and allows selected jobs to be rerun. Treat a generated manifest as immutable while jobs are queued or running; generate a new file for a changed study.

## 1. Plan a study and generate its manifest

```bash
# Preview the experiment cells; no manifest written and no GPU used.
python -m bdsurvive.cli plan \
  --study configs/study/sok_first_pilot.yaml \
  --dry-run

# Write the manifest to a convenient project-local directory.
mkdir -p manifests
python -m bdsurvive.cli plan \
  --study configs/study/sok_first_pilot.yaml \
  --out manifests/lexical_pilot_g4.jsonl

# Display numbered cells and validity status.
python -m bdsurvive.cli list \
  --manifest manifests/lexical_pilot_g4.jsonl
```

A study with one configuration has index `0`. For multiple rows, SLURM array index `i` selects manifest row `i`; invalid combinations are marked `not_applicable` rather than silently treated as failed experiments.

The manifest is a **small input file** under `manifests/`. The training results are written separately under the configured `output_root`.

## 2. Run the generational lineage (G0–G4)

For a single cell on Leonardo:

```bash
mkdir -p logs/slurm
sbatch \
  --array=0 \
  --export=ALL,RUN_MODE=run,MANIFEST=manifests/lexical_pilot_g4.jsonl \
  slurm/grid_array.sbatch
```

For a multi-cell manifest, use an appropriate array range, such as `--array=0-11`, after confirming the cell count with `list`.

For direct execution in an environment with the required GPU:

```bash
python -m bdsurvive.cli run \
  --manifest manifests/lexical_pilot_g4.jsonl \
  --index 0
```

The generational sequence is:

| Generation | Input checkpoint | Operation |
| --- | --- | --- |
| **G0** | Original model | Install the backdoor and evaluate the planted model |
| **G1** | G0 | Benign fine-tuning on shard 1, then evaluate |
| **G2** | G1 | Benign fine-tuning on shard 2, then evaluate |
| **G3** | G2 | Benign fine-tuning on shard 3, then evaluate |
| **G4** | G3 | Benign fine-tuning on shard 4, then evaluate |

G0 planting examples are reserved separately from the downstream AG News shards. The downstream shards are deterministic and disjoint. Each generational stage trains from its predecessor checkpoint and starts its own fine-tuning session.

The runner records its state and checkpoints so that a correctly matched manifest can resume unfinished generations. A `SUCCESS` marker from a shorter lineage is **not sufficient evidence** that a longer requested lineage has completed: always inspect the recorded last generation.

## 3. Run the continuous control (C1–C4)

**Complete and verify the generational run first.** The continuous control requires the **same saved G0 checkpoint**; it must not replant a new parent.

Run the same manifest with a different mode:

```bash
sbatch \
  --array=0 \
  --export=ALL,RUN_MODE=continuous,MANIFEST=manifests/lexical_pilot_g4.jsonl \
  slurm/grid_array.sbatch
```

Or directly:

```bash
python -m bdsurvive.cli continuous \
  --manifest manifests/lexical_pilot_g4.jsonl \
  --index 0
```

The continuous arm concatenates the downstream shards in generation order, uses the prescribed batch sequence, and makes **one uninterrupted call to the training loop**, with one LoRA adapter and one AdamW optimizer. Evaluations occur after the same cumulative update counts as the generational arm:

| Benign updates | Generational arm | Continuous arm |
| ---: | --- | --- |
| 300 | G1 | C1 |
| 600 | G2 | C2 |
| 900 | G3 | C3 |
| 1,200 | G4 | C4 |

Intermediate evaluation does not merge the live adapter or reset the optimizer. The continuous runner stores its final checkpoint and boundary measurements separately from the generational results. An interrupted continuous run currently restarts its continuous training from G0 rather than resuming the optimizer midway.

**Interpretation caveat:** Matching G0, data order and update counts does not automatically prove identical initial LoRA-adapter randomness between arms. Verify this, and repeat across seeds, before attributing small differences specifically to optimizer or adapter resets.

## 4. Results and checkpoint layout

The study YAML's `shared.output_root` controls persistent storage. The active runner writes a cell beneath:

```text
<output_root>/
  runs_v3/
    <cell_id>/
      state.json
      generations.jsonl
      result.json
      SUCCESS
      checkpoint_gen000/
      checkpoint_gen001/
      checkpoint_gen002/
      checkpoint_gen003/
      checkpoint_gen004/
      continuous/
        result.json
        SUCCESS
        checkpoint/
          continuous_boundaries.jsonl
          ...model and tokenizer files...
```

Checkpoint directories vary with the selected retention policy. The diagram shows a full-retention G0–G4 experiment.

- `cell_id` identifies the experiment configuration and determines the output namespace. The current identity convention excludes `output_root` and generation horizon so a compatible lineage can be extended.
- `state.json` tracks lineage progress and the checkpoint needed for resumption.
- `generations.jsonl` contains the G0–G4 per-generation measurements.
- `continuous/checkpoint/continuous_boundaries.jsonl` contains C1–C4 measurements.
- `SUCCESS` under the cell directory and under `continuous/` refer to **different arms**.

**Avoid changing the experiment's other configuration fields between a partial run and its extension.** Regenerate the manifest when the study changes, and verify that its `cell_id` resolves to the intended existing run before submitting.

### Verify G0–G4 completion

Run on Leonardo after the generational job finishes:

```bash
python - <<'PY'
from pathlib import Path
from bdsurvive.plan import read_manifest
from bdsurvive.run import config_from_row, run_dir_for
from bdsurvive import checkpoint as ckpt

row = read_manifest('manifests/lexical_pilot_g4.jsonl')[0]
cfg = config_from_row(row)
root = Path(run_dir_for(cfg))
state = ckpt.load_state(str(root), cfg.cell_id)

assert cfg.propagation.generations == 4
assert state.last_completed_generation == 4
assert ckpt.is_successful(str(root))
assert Path(ckpt.checkpoint_dir(str(root), 0)).is_dir()
assert (root / 'generations.jsonl').is_file()
print('PASS: G0-G4 complete; G0 checkpoint retained')
PY
```

### Verify C1–C4 completion

```bash
python - <<'PY'
import json
from pathlib import Path
from bdsurvive.plan import read_manifest
from bdsurvive.run import config_from_row, run_dir_for

row = read_manifest('manifests/lexical_pilot_g4.jsonl')[0]
cfg = config_from_row(row)
control = Path(run_dir_for(cfg)) / 'continuous'
path = control / 'checkpoint' / 'continuous_boundaries.jsonl'

assert (control / 'SUCCESS').is_file()
records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
assert [r['generation'] for r in records] == [1, 2, 3, 4]
assert [r['updates_benign_cumulative'] for r in records] == [300, 600, 900, 1200]
print('PASS: C1-C4 complete')
PY
```

Use the actual `cfg.training.steps` for other budgets; the `300`-step checks above apply to the lexical pilot.

## Repository map

| Path | Responsibility |
| --- | --- |
| `configs/study/*.yaml` | Study definitions and experiment axes |
| `bdsurvive/plan.py` | YAML expansion, grid validation and manifest serialization |
| `bdsurvive/cli.py` | Supported `plan`, `list`, `run`, `continuous` entry points |
| `bdsurvive/core/` | Configuration dataclasses, registries and compatibility validation |
| `bdsurvive/run.py` | Cell identity, execution, results and continuous-control dispatch |
| `bdsurvive/generation.py` | Generational execution, checkpoint ancestry and resume |
| `bdsurvive/steps.py` | G0 planting, benign fine-tuning, evaluation and continuous control |
| `bdsurvive/train_loop.py` | Batching, optimizer updates and intermediate-evaluation callback |
| `bdsurvive/shards.py` | Deterministic planting reservation and downstream data shards |
| `bdsurvive/attacks/`, `bdsurvive/triggers/` | Attack installation and trigger implementations |
| `bdsurvive/finetune/` | LoRA, BitFit and full fine-tuning implementations |
| `bdsurvive/eval/` | Behavioral and mechanism-related measurements |
| `bdsurvive/checkpoint.py` | State files, checkpoint promotion and retention |
| `slurm/grid_array.sbatch` | Leonardo GPU array entry point for both arms |
| `tests/test_units.py` | Local tests and regression checks |

`configs/mechanisms.yaml` is retained for the separate `tokenizer-audit` utility. It is **not** the source of truth for study-grid experiments.

