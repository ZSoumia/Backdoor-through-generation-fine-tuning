"""Single entry point.

    python -m bdsurvive.cli plan   --study configs/study/section1.yaml
    python -m bdsurvive.cli run    --manifest manifest.jsonl --index 27
    python -m bdsurvive.cli list   --manifest manifest.jsonl

Local and HPC use the same command; SLURM just supplies --index from the
array task id.
"""
import argparse
import json
import sys
from typing import Any, Dict, List

from . import plan as planner
from .core.validation import RunStatus


def cmd_plan(args: argparse.Namespace) -> int:
    study = planner.load_study(args.study)
    configs = planner.expand(study)
    rows = planner.manifest_rows(configs)
    summary = planner.summarize(rows)

    print(f"cells: {summary['total']}  "
          f"valid: {summary['valid']}  N/A: {summary['not_applicable']}")
    for reason, count in sorted(summary["na_reasons"].items()):
        print(f"  N/A x{count}: {reason}")

    if args.dry_run:
        print("\n(dry run -- manifest not written)")
        for row in rows[:args.show]:
            print(f"  [{row['index']:4d}] {row['grid_status']:15s} "
                  f"{row['model']} | {row['attack']} | {row['finetune']} | seed {row['seed']}")
        if len(rows) > args.show:
            print(f"  ... {len(rows) - args.show} more")
        return 0

    planner.write_manifest(rows, args.out)
    print(f"\nmanifest written: {args.out}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    rows = planner.read_manifest(args.manifest)
    for row in rows:
        marker = " " if row["grid_status"] == "valid" else "x"
        print(f"{marker} [{row['index']:4d}] {row['model']} | {row['attack']} | "
              f"{row['finetune']} | seed {row['seed']} | {row['grid_status']}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .run import execute_row
    from .steps import training_step

    rows = planner.read_manifest(args.manifest)
    if args.index >= len(rows):
        print(f"index {args.index} out of range ({len(rows)} rows)", file=sys.stderr)
        return 2

    row = rows[args.index]
    print(f"cell {args.index}: {row['model']} | {row['attack']} | {row['finetune']}")
    outcome = execute_row(row, training_step, skip_completed=not args.force)
    print(json.dumps(outcome, indent=2))
    return 0 if outcome["status"] in ("success", "skipped", "not_applicable") else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bdsurvive")
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan", help="expand a study into a manifest")
    p_plan.add_argument("--study", required=True)
    p_plan.add_argument("--out", default="manifest.jsonl")
    p_plan.add_argument("--dry-run", action="store_true",
                        help="print the cell count and a preview, write nothing")
    p_plan.add_argument("--show", type=int, default=20)
    p_plan.set_defaults(func=cmd_plan)

    p_list = sub.add_parser("list", help="list manifest rows")
    p_list.add_argument("--manifest", required=True)
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="execute one manifest row")
    p_run.add_argument("--manifest", required=True)
    p_run.add_argument("--index", type=int, required=True)
    p_run.add_argument("--force", action="store_true",
                       help="re-run even if the cell already succeeded")
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
