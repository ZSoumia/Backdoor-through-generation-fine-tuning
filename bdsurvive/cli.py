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


def cmd_compare(args: argparse.Namespace) -> int:
    """Two complementary budget views over a set of lineage runs.

    A survival gap between mechanisms has a rival explanation: they did not
    see the same amount of training. Equal example counts are not equal
    exposure, because positional inputs are grown into a length band while
    content inputs are raw ag_news rows, so the same 4800 examples carry
    different token counts.

    Comparison A -- equal optimizer budget. Runs are grouped by (steps, batch
    size, lr) and their token exposure is reported as a spread. Conclusions
    drawn here hold optimizer work constant and let tokens vary.

    Comparison B -- equal token exposure. Runs are bucketed by non-padding
    tokens and their step counts are reported as a spread. Conclusions here
    hold tokens constant and let optimizer work vary.

    If A and B disagree, exposure is driving the result and no architectural
    claim can be made from this data. If they agree, exposure has been ruled
    out as the explanation. Reading only one of them cannot tell you which.

    Clean accuracy is reported as the gap to each model's own zero-poison
    reference, never against a fixed threshold: architectures do not share a
    ceiling on ag_news, so an absolute bar is a different test per model.
    """
    import glob
    import os

    rows = []
    for root in args.runs:
        for f in sorted(glob.glob(os.path.join(root, "*", "exposure_summary.json"))):
            with open(f) as fh:
                s = json.load(fh)
            s["run"] = os.path.basename(os.path.dirname(f))
            gen = os.path.join(os.path.dirname(f), "generations.jsonl")
            if os.path.exists(gen):
                with open(gen) as gh:
                    g = [json.loads(l) for l in gh if l.strip()]
                last = [r for r in g if r.get("generation", 0) > 0]
                s["excess_g0"] = next(
                    (r.get("asr_excess") for r in g if r.get("generation") == 0), None)
                s["excess_final"] = last[-1].get("asr_excess") if last else None
            rows.append(s)

    if not rows:
        print("no exposure_summary.json found under: " + ", ".join(args.runs))
        print("These are written by run_lineage; runs from before that change "
              "have none and must be re-measured, not back-filled.")
        return 1

    def fmt(v, nd=3):
        return "n/a" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    def opt_key(s):
        o = s.get("optimizer_settings") or {}
        return (s.get("training_steps"), o.get("batch_size"), o.get("lr"),
                o.get("finetune_method"))

    print(f"\n{'='*104}\nCOMPARISON A -- equal optimizer budget "
          f"(grouped by steps / batch / lr / method; tokens allowed to vary)\n{'='*104}")
    groups = {}
    for s in rows:
        groups.setdefault(opt_key(s), []).append(s)
    for key, grp in sorted(groups.items(), key=lambda kv: str(kv[0])):
        steps, B, lr, meth = key
        toks = [s["total_non_padding_tokens"] for s in grp
                if s.get("total_non_padding_tokens")]
        spread = (max(toks) / min(toks)) if len(toks) > 1 and min(toks) else 1.0
        print(f"\nsteps={steps} batch={B} lr={lr} method={meth}   "
              f"token spread within group: {spread:.2f}x")
        print(f"  {'run':40s} {'trigger':11s} {'tokens':>11s} {'mean_len':>9s} "
              f"{'exc_g0':>7s} {'exc_fin':>8s} {'clean-ref':>10s}")
        for s in sorted(grp, key=lambda x: x["run"]):
            print(f"  {s['run'][:40]:40s} {str(s.get('trigger_type'))[:11]:11s} "
                  f"{s.get('total_non_padding_tokens', 0):11,} "
                  f"{fmt(s.get('mean_sequence_length'), 1):>9s} "
                  f"{fmt(s.get('excess_g0')):>7s} {fmt(s.get('excess_final')):>8s} "
                  f"{fmt(s.get('final_clean_gap_to_ref')):>10s}")
        if spread > 1.25:
            print(f"  NOTE: {spread:.2f}x token spread at matched optimizer "
                  f"budget -- any gap in this group is confounded with exposure")

    print(f"\n{'='*104}\nCOMPARISON B -- equal token exposure "
          f"(bucketed by non-padding tokens; steps allowed to vary)\n{'='*104}")
    width = args.token_bucket
    buckets = {}
    for s in rows:
        t = s.get("total_non_padding_tokens")
        if not t:
            continue
        buckets.setdefault((t // width) * width, []).append(s)
    for lo, grp in sorted(buckets.items()):
        st = [s.get("training_steps") for s in grp if s.get("training_steps")]
        sspread = (max(st) / min(st)) if len(st) > 1 and min(st) else 1.0
        print(f"\ntokens {lo:,}-{lo + width - 1:,}   "
              f"step spread within bucket: {sspread:.2f}x")
        print(f"  {'run':40s} {'trigger':11s} {'steps':>7s} {'tokens':>11s} "
              f"{'exc_g0':>7s} {'exc_fin':>8s} {'clean-ref':>10s}")
        for s in sorted(grp, key=lambda x: x["run"]):
            print(f"  {s['run'][:40]:40s} {str(s.get('trigger_type'))[:11]:11s} "
                  f"{s.get('training_steps', 0):>7} "
                  f"{s.get('total_non_padding_tokens', 0):11,} "
                  f"{fmt(s.get('excess_g0')):>7s} {fmt(s.get('excess_final')):>8s} "
                  f"{fmt(s.get('final_clean_gap_to_ref')):>10s}")
        if sspread > 1.25:
            print(f"  NOTE: {sspread:.2f}x step spread at matched token "
                  f"exposure -- any gap here is confounded with optimizer work")

    missing = [s["run"] for s in rows if s.get("clean_ref_acc") is None]
    if missing:
        print(f"\n{len(missing)} run(s) have no matched clean reference on disk, "
              f"so clean-ref is n/a for them:")
        for r in missing[:12]:
            print(f"  {r}")
        print("  A reference is derived from (base_model, lr, steps, batch_size, "
              "max_len, seed), so a mechanism with a different max_len than the "
              "lexical entries has no matching one. Absolute clean_acc is NOT a "
              "substitute -- it is not comparable across architectures.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bdsurvive")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cmp = sub.add_parser(
        "compare",
        help="budget-controlled comparison tables (equal steps vs equal tokens)")
    p_cmp.add_argument("runs", nargs="+",
                       help="lineage_<tag> directories to read")
    p_cmp.add_argument("--token-bucket", type=int, default=50000,
                       help="bucket width for Comparison B (default 50000)")
    p_cmp.set_defaults(func=cmd_compare)

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
