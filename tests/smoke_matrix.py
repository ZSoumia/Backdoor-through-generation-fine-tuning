#!/usr/bin/env python3
"""Run short, isolated G0->G1 smoke experiments using the REAL bdsurvive CLI.

By default this only enumerates planned cells. Pass --execute on an allocated
GPU node to train. No production manifest, checkpoint, or results are edited.

Examples:
    python -m tests.smoke_matrix --study configs/study/section1_grid.yaml
    python -m tests.smoke_matrix --study configs/study/section1_grid.yaml \
        --model Qwen/Qwen2.5-1.5B --finetune lora --execute
    python -m tests.smoke_matrix --study configs/study/section1_grid.yaml \
        --execute --all

Important: 1-step smoke runs test plumbing, not ASR strength or convergence.
"""

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

from bdsurvive import plan
from bdsurvive.run import config_from_row

RUNNABLE = {"valid", "mechanism_not_applicable"}


def number_between(row, key, lo, hi, issues):
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        issues.append(f"missing or nonnumeric {key}: {value!r}")
        return
    if not math.isfinite(value) or not lo <= value <= hi:
        issues.append(f"invalid {key}: {value!r} (expected {lo}..{hi})")


def validate_metrics(path, expected_generations):
    """Validate numerical consistency, not whether a backdoor was established."""
    issues = []
    warnings = []
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()
            if s.strip()]
    generations = [r.get("generation") for r in rows]
    expected = list(range(expected_generations))
    if generations != expected:
        issues.append(f"generation sequence is {generations}; expected {expected}")

    for row in rows:
        g = row.get("generation", "?")
        errors = []
        for key in ("asr", "control_rate"):
            number_between(row, key, 0, 1, errors)
        number_between(row, "asr_excess", -1, 1, errors)
        for key in ("ci_lo", "ci_hi"):
            number_between(row, key, -1, 1, errors)
        if all(isinstance(row.get(k), (int, float))
               for k in ("asr", "control_rate", "asr_excess")):
            difference = row["asr"] - row["control_rate"]
            if abs(row["asr_excess"] - difference) > 0.002:
                errors.append("asr_excess != asr - control_rate")
        if isinstance(row.get("ci_lo"), (int, float)) and isinstance(row.get("ci_hi"), (int, float)):
            if row["ci_lo"] > row["ci_hi"]:
                errors.append("confidence interval bounds inverted")
        clean_key = next((k for k in ("clean_acc", "clean_accuracy", "clean_metric")
                          if k in row), None)
        if clean_key is None:
            errors.append("missing clean accuracy")
        else:
            number_between(row, clean_key, 0, 1, errors)
        for count_key in ("n_trigger", "n_control"):
            if count_key in row and row[count_key] <= 0:
                errors.append(f"{count_key} must be positive")
        if isinstance(row.get("control_rate"), (float, int)) and row["control_rate"] > 0.5:
            warnings.append(f"G{g}: high target-label control rate {row['control_rate']:.3f}; examine trigger specificity")
        issues.extend(f"G{g}: {message}" for message in errors)
    return issues, warnings


def collect_cells(studies, manifests):
    cells = []
    for study_path in studies:
        rows = plan.manifest_rows(plan.expand(plan.load_study(study_path)))
        for row in rows:
            cells.append((f"{Path(study_path).stem}", row))
    for manifest_path in manifests:
        for row in plan.read_manifest(manifest_path):
            cells.append((f"{Path(manifest_path).stem}", row))
    return cells


def build_smoke_row(row, output_root, args):
    cfg = config_from_row(row)
    cfg.output_root = str(output_root)
    cfg.attack.steps = args.plant_steps
    cfg.attack.batch_size = args.batch_size
    cfg.training.steps = args.benign_steps
    cfg.training.batch_size = args.batch_size
    cfg.training.log_every = 1
    cfg.data.shard_size = args.benign_steps * args.batch_size
    cfg.propagation.generations = 2
    # G0 must remain available for a later optional continuous-control run.
    cfg.propagation.retention = "all"
    return plan.manifest_rows([cfg])[0]


def run_cell(label, row, root, args):
    source_index = row["index"]
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    cell_dir = root / f"{safe_label}_{source_index:04d}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    smoke_row = build_smoke_row(row, cell_dir / "artifacts", args)
    manifest = cell_dir / "smoke.jsonl"
    plan.write_manifest([smoke_row], str(manifest))
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    command = [sys.executable, "-m", "bdsurvive.cli", "run",
               "--manifest", str(manifest), "--index", "0"]
    log = cell_dir / "run.log"
    try:
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(command, env=env, stdout=handle,
                                    stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"cell": label, "index": source_index, "status": "TIMEOUT", "log": str(log)}
    if result.returncode != 0:
        return {"cell": label, "index": source_index, "status": "FAILED",
                "returncode": result.returncode, "log": str(log)}

    files = list((cell_dir / "artifacts").rglob("generations.jsonl"))
    if len(files) != 1:
        return {"cell": label, "index": source_index, "status": "FAILED",
                "issues": [f"expected exactly one generations.jsonl, got {len(files)}"],
                "log": str(log)}
    issues, warnings = validate_metrics(files[0], expected_generations=2)
    record = {"cell": label, "index": source_index,
              "status": "FAILED" if issues else "PASS", "issues": issues,
              "warnings": warnings, "metrics": str(files[0]), "log": str(log)}
    if args.continuous and row["finetune"] == "lora" and not issues:
        continuous_log = cell_dir / "continuous.log"
        cont_command = [sys.executable, "-m", "bdsurvive.cli", "continuous",
                        "--manifest", str(manifest), "--index", "0"]
        try:
            with continuous_log.open("w", encoding="utf-8") as handle:
                cont = subprocess.run(cont_command, env=env, stdout=handle,
                                      stderr=subprocess.STDOUT, timeout=args.timeout,
                                      check=False)
            if cont.returncode != 0:
                record["status"] = "FAILED"
                record["issues"].append("continuous-control command failed")
        except subprocess.TimeoutExpired:
            record["status"] = "TIMEOUT"
            record["issues"].append("continuous-control command timed out")
        record["continuous_log"] = str(continuous_log)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", nargs="*", default=[], metavar="YAML")
    parser.add_argument("--manifest", nargs="*", default=[], metavar="JSONL")
    parser.add_argument("--model", action="append", help="exact model name; repeatable")
    parser.add_argument("--attack", action="append", help="attack name; repeatable")
    parser.add_argument("--finetune", action="append", help="lora, bitfit or full; repeatable")
    parser.add_argument("--indices", help="original manifest indices, e.g. 0,3,8-12")
    parser.add_argument("--execute", action="store_true", help="actually run on a GPU node")
    parser.add_argument("--all", action="store_true", help="run all selected cells instead of just first")
    parser.add_argument("--max-cells", type=int, default=1,
                        help="max cells to execute without --all (default 1)")
    parser.add_argument("--plant-steps", type=int, default=1)
    parser.add_argument("--benign-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=2400,
                        help="seconds per cell")
    parser.add_argument("--continuous", action="store_true",
                        help="also invoke CLI continuous for successful LoRA cells")
    parser.add_argument("--out", default=None,
                        help="new, unused output directory (default: timestamped smoke_runs directory)")
    args = parser.parse_args(argv)

    if not (args.study or args.manifest):
        parser.error("specify --study and/or --manifest")
    if min(args.plant_steps, args.benign_steps, args.batch_size, args.timeout) < 1:
        parser.error("steps, batch size and timeout must all be positive")
    if args.max_cells < 1:
        parser.error("--max-cells must be positive")

    allowed = None
    if args.indices:
        allowed = set()
        for part in args.indices.split(","):
            if "-" in part:
                a, b = map(int, part.split("-", 1))
                if a > b:
                    parser.error(f"backward range: {part}")
                allowed.update(range(a, b + 1))
            else:
                allowed.add(int(part))

    selected = []
    for label, row in collect_cells(args.study, args.manifest):
        if row.get("grid_status") not in RUNNABLE:
            continue
        if args.model and row["model"] not in args.model:
            continue
        if args.attack and row["attack"] not in args.attack:
            continue
        if args.finetune and row["finetune"] not in args.finetune:
            continue
        if allowed is not None and row["index"] not in allowed:
            continue
        selected.append((label, row))

    print(f"Runnable matching cells: {len(selected)}")
    for label, row in selected:
        print(f"  {label}[{row['index']}]: {row['model']} | {row['attack']} | {row['finetune']}")
    if not selected:
        print("Nothing to test")
        return 2
    if not args.execute:
        print("DRY RUN: use --execute on a GPU node; --all to run every matching cell")
        return 0

    if not torch.cuda.is_available():
        print("BLOCKED: no CUDA device; use an allocated GPU node", file=sys.stderr)
        return 2

    folder = args.out or f"smoke_runs/launch_gate_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    root = Path(folder).resolve()
    if root.exists():
        parser.error(f"smoke output directory already exists: {root}; choose a new --out")
    root.mkdir(parents=True, exist_ok=False)
    (root / "SMOKE_ONLY.txt").write_text(
        "One-step validation only. Do NOT use as scientific measurements.\n",
        encoding="utf-8",
    )
    active = selected if args.all else selected[:args.max_cells]
    if len(active) < len(selected):
        print(f"NOTE: executing {len(active)} of {len(selected)} selected cells; use --all to test every cell")
    reports = []
    for ordinal, (label, row) in enumerate(active, 1):
        print(f"[{ordinal}/{len(active)}] {label}[{row['index']}] ...", flush=True)
        try:
            result = run_cell(label, row, root, args)
        except Exception as exc:
            result = {"cell": label, "index": row["index"], "status": "FAILED",
                      "issues": [f"{type(exc).__name__}: {exc}"]}
        reports.append(result)
        print(f"  {result['status']} | {result.get('log', '(setup error)')}", flush=True)
        (root / "report.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")

    passed = sum(r["status"] == "PASS" for r in reports)
    print(f"\nRESULT: {passed}/{len(reports)} passed; report: {root / 'report.json'}")
    return 0 if passed == len(reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
