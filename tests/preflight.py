"""Preflight study configurations before allocating GPU jobs.

Install as tests/preflight.py and run:
    python -m tests.preflight --study configs/study/*.yaml

This checks manifest expansion, component registries, offline attack tables,
AG News partition compatibility, and model cache availability. It cannot
replace a GPU training smoke test or guarantee absence of runtime OOMs.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from bdsurvive import plan
from bdsurvive.core.registry import ATTACKS, FINETUNE, TRIGGERS
from bdsurvive.data import load_task
from bdsurvive.run import config_from_row  # imports and populates implementation registries
from bdsurvive.shards import planting_rows


RUNNABLE = {"valid", "mechanism_not_applicable"}
KNOWN_STATUSES = RUNNABLE | {"not_applicable", "untested"}


def inspect_table(path, task, planting_pool_size, data_cache):
    """Check that an offline attack table applies to the actual planting split."""
    path = Path(path)
    problems = []
    if not path.is_file():
        return [f"missing offline trigger table: {path}"]

    # The train/test corpus is loaded once per task. Offline data should
    # already be cached before any GPU experiment is submitted.
    if task not in data_cache:
        train = load_task(task, "train")
        test = load_task(task, "test")
        data_cache[task] = (train, test)
    train, test = data_cache[task]

    all_train = {text for text, _ in train}
    all_test = {text for text, _ in test}
    reserved, _ = planting_rows(train, planting_pool_size)
    reserved_texts = {text for text, _ in reserved}

    n_rows = 0
    n_train_edits = 0
    n_eval_pairs = 0
    nonplanting = 0
    unknown = 0
    duplicates = 0
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not all(k in row for k in ("text", "trigger", "control")):
                    raise ValueError("expected text, trigger and control keys")
                text = row["text"]
                if not isinstance(text, str):
                    raise ValueError("text must be a string")
            except (ValueError, TypeError) as exc:
                problems.append(f"{path}:{line_no}: {exc}")
                if len(problems) >= 10:
                    break
                continue
            n_rows += 1
            if text in seen:
                duplicates += 1
            seen.add(text)
            if text in all_train:
                if text not in reserved_texts:
                    nonplanting += 1
                if row["trigger"] is not None:
                    n_train_edits += 1
            elif text in all_test:
                if row["trigger"] is not None and row["control"] is not None:
                    n_eval_pairs += 1
            else:
                unknown += 1

    if n_rows == 0:
        problems.append(f"{path}: table is empty")
    if nonplanting:
        problems.append(
            f"{path}: {nonplanting} training entries are outside the fixed "
            "shuffled planting reservation (likely first-30000 mismatch)"
        )
    if unknown:
        problems.append(f"{path}: {unknown} texts not found in AG News train or test")
    if duplicates:
        problems.append(f"{path}: {duplicates} duplicate source texts; lookup will overwrite entries")
    if not n_train_edits:
        problems.append(f"{path}: no usable training trigger edits")
    if not n_eval_pairs:
        problems.append(f"{path}: no paired trigger/control evaluation rows")
    print(f"  table {path}: {n_rows} rows, {n_train_edits} train edits, "
          f"{n_eval_pairs} eval pairs")
    return problems


def check_study(path, table_paths, data_cache, model_names):
    """Exercise every planned cell without constructing a neural network."""
    problems = []
    try:
        study = plan.load_study(str(path))
        rows = plan.manifest_rows(plan.expand(study))
    except Exception as exc:
        return [f"{path}: planning failed: {type(exc).__name__}: {exc}"], 0

    counts = Counter(row.get("grid_status") for row in rows)
    print(f"{path}: {len(rows)} cells, status={dict(counts)}")
    if not rows:
        problems.append(f"{path}: no jobs produced")

    seen_ids = set()
    for n, row in enumerate(rows):
        status = row.get("grid_status")
        if status not in KNOWN_STATUSES:
            problems.append(f"{path} cell {n}: unknown grid status {status!r}")
        if row.get("index") != n:
            problems.append(f"{path} cell {n}: manifest index mismatch")
        try:
            cfg = config_from_row(row)
            if cfg.cell_id in seen_ids:
                problems.append(f"{path} cell {n}: duplicate cell_id {cfg.cell_id}")
            seen_ids.add(cfg.cell_id)
            model_names.add(cfg.model.name)

            if status not in RUNNABLE:
                continue
            TRIGGERS.get(cfg.attack.trigger.name)
            ATTACKS.get(cfg.attack.installation)
            FINETUNE.get(cfg.finetune.name)
            if cfg.model.task_type != "classification":
                problems.append(f"{path} cell {n}: non-classification task needs separate smoke test")
            if cfg.training.steps < 1 or cfg.training.batch_size < 1:
                problems.append(f"{path} cell {n}: invalid step count or batch size")
            if cfg.attack.trigger.name == "semantic_bite" and cfg.attack.poison_rate != 1.0:
                problems.append(f"{path} cell {n}: BITE requires poison_rate: 1.0")
            table = cfg.attack.trigger.params.get("table")
            if table:
                table_paths.add((str(table), cfg.data.planting_task,
                                 cfg.attack.planting_pool_size))
            if cfg.finetune.name == "full" and cfg.model.name.lower().endswith("2.8b"):
                print(f"  GPU-SMOKE-REQUIRED: {cfg.cell_id} "
                      "(full FP32 AdamW may exceed single-GPU memory)")
            if cfg.attack.trigger.name == "positional" and cfg.training.max_len <= cfg.attack.trigger.params.get("tau", 0):
                problems.append(f"{path} cell {n}: max_len cannot reach positional tau")
        except Exception as exc:
            problems.append(f"{path} cell {n}: {type(exc).__name__}: {exc}")
    return problems, len(rows)


def check_manifest(path, table_paths, model_names):
    problems = []
    try:
        rows = plan.read_manifest(str(path))
    except Exception as exc:
        return [f"{path}: cannot read manifest: {exc}"], 0
    print(f"{path}: {len(rows)} manifest rows")
    seen_ids = set()
    for n, row in enumerate(rows):
        if row.get("index") != n:
            problems.append(f"{path} row {n}: index mismatch")
        try:
            cfg = config_from_row(row)
            if cfg.cell_id in seen_ids:
                problems.append(f"{path} row {n}: duplicate cell_id {cfg.cell_id}")
            seen_ids.add(cfg.cell_id)
            model_names.add(cfg.model.name)
            if row.get("grid_status") in RUNNABLE:
                TRIGGERS.get(cfg.attack.trigger.name)
                ATTACKS.get(cfg.attack.installation)
                FINETUNE.get(cfg.finetune.name)
                table = cfg.attack.trigger.params.get("table")
                if table:
                    table_paths.add((str(table), cfg.data.planting_task,
                                 cfg.attack.planting_pool_size))
        except Exception as exc:
            problems.append(f"{path} row {n}: {type(exc).__name__}: {exc}")
    return problems, len(rows)


def check_models(studies, manifests):
    command = [sys.executable, "-m", "bdsurvive.cli", "prefetch"]
    if studies:
        command.extend(["--study", *studies])
    if manifests:
        command.extend(["--manifest", *manifests])
    command.append("--verify-only")
    print("\nChecking all model assets offline ...", flush=True)
    return subprocess.run(command, check=False).returncode


def verify_manifest_matches_study(study_path, manifest_path):
    """Catch manifests generated before a study YAML was edited."""
    try:
        expected = plan.manifest_rows(plan.expand(plan.load_study(study_path)))
        actual = plan.read_manifest(manifest_path)
        a = [(r["cell_id"], r.get("grid_status")) for r in expected]
        b = [(r["cell_id"], r.get("grid_status")) for r in actual]
        if a != b:
            return [f"STALE MANIFEST: {manifest_path} does not match {study_path}; "
                    "rerun bdsurvive.cli plan before submitting jobs"]
    except Exception as exc:
        return [f"cannot compare study and manifest: {type(exc).__name__}: {exc}"]
    print(f"Manifest matches study: {manifest_path}")
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", nargs="+", default=[], metavar="YAML")
    parser.add_argument("--manifest", nargs="+", default=[], metavar="JSONL")
    parser.add_argument("--skip-models", action="store_true",
                        help="skip Hugging Face cache verification")
    parser.add_argument("--skip-tables", action="store_true",
                        help="skip offline-table validation for local development")
    args = parser.parse_args(argv)
    if not args.study and not args.manifest:
        parser.error("Specify --study and/or --manifest")

    problems = []
    table_paths = set()
    model_names = set()
    data_cache = {}
    total = 0
    for study in args.study:
        errs, count = check_study(study, table_paths, data_cache, model_names)
        problems.extend(errs)
        total += count
    for manifest in args.manifest:
        errs, count = check_manifest(manifest, table_paths, model_names)
        problems.extend(errs)
        total += count

    if len(args.study) == 1 and len(args.manifest) == 1:
        problems.extend(verify_manifest_matches_study(
            args.study[0], args.manifest[0]))

    print(f"\nChecked {total} job specifications, {len(model_names)} unique models")
    if args.skip_tables:
        print("WARNING: offline tables NOT checked")
    else:
        for path, task, pool_size in sorted(table_paths):
            try:
                problems.extend(inspect_table(path, task, pool_size, data_cache))
            except Exception as exc:
                problems.append(f"{path}: table audit failed: {type(exc).__name__}: {exc}")

    if args.skip_models:
        print("WARNING: model cache NOT checked")
    elif check_models(args.study, args.manifest):
        problems.append("model cache verification failed")

    if problems:
        print(f"\nBLOCKED: {len(problems)} issue(s)", file=sys.stderr)
        for error in problems:
            print(f"  - {error}", file=sys.stderr)
        return 1
    if args.skip_tables or args.skip_models:
        print("\nPARTIAL PREFLIGHT PASS: requested checks passed; skipped checks remain")
    else:
        print("\nPREFLIGHT PASS: configuration, registries, tables, cache")
    print("NOTE: does not prove GPU memory capacity or forward/backward compatibility")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
