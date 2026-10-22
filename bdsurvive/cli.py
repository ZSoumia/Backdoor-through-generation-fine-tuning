"""Single entry point.

    python -m bdsurvive.cli plan    --study configs/study/section1.yaml
    python -m bdsurvive.cli run     --manifest manifest.jsonl --index 27
    python -m bdsurvive.cli list    --manifest manifest.jsonl
    python -m bdsurvive.cli compare $WORK/bdsurvive_runs/lineage_sec1_*


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


def cmd_verify(args: argparse.Namespace) -> int:
    """Recompute every interval from saved predictions and diff against the row.

    A CI in generations.jsonl is only trustworthy if it can be reproduced from
    the predictions on disk. This re-derives asr, control_rate, asr_excess and
    the bootstrap interval from preds_g*.json and reports any disagreement.

    It also reports, per run, whether the eval design was paired and which
    estimator produced the stored interval, because a paired design analysed
    with independent resampling yields an interval that is too wide -- a
    discrepancy this command will surface as a width mismatch rather than a
    point-estimate mismatch.
    """
    import glob
    import os

    from .eval.extinction import excess_ci

    bad = 0
    checked = 0
    for root in args.runs:
        for gen_file in sorted(glob.glob(os.path.join(root, "*",
                                                      "generations.jsonl"))):
            run_dir = os.path.dirname(gen_file)
            run = os.path.basename(run_dir)
            with open(gen_file) as fh:
                rows = [json.loads(l) for l in fh if l.strip()]
            for row in rows:
                g = row.get("generation")
                pf = os.path.join(run_dir, f"preds_g{g}.json")
                if not os.path.exists(pf):
                    print(f"{run} g{g}: NO preds file, cannot verify")
                    bad += 1
                    continue
                with open(pf) as fh:
                    preds = json.load(fh)
                th, chh = preds.get("trigger_hits"), preds.get("control_hits")
                if not th or not chh:
                    print(f"{run} g{g}: preds file has no hit vectors")
                    bad += 1
                    continue
                paired = bool(preds.get("paired", False))
                re_exc = excess_ci(th, chh, paired=paired)
                checked += 1

                deltas = {
                    "asr": abs(sum(th) / len(th) - (row.get("asr") or 0)),
                    "control_rate": abs(sum(chh) / len(chh)
                                        - (row.get("control_rate") or 0)),
                    "asr_excess": abs(re_exc["asr_excess"]
                                      - (row.get("asr_excess") or 0)),
                }
                ci_lo_d = abs(re_exc["ci_lo"] - (row.get("ci_lo") or 0))
                ci_hi_d = abs(re_exc["ci_hi"] - (row.get("ci_hi") or 0))
                worst = max(deltas.values())
                stored = row.get("bootstrap", "unknown")

                if worst > args.tol or max(ci_lo_d, ci_hi_d) > args.ci_tol:
                    bad += 1
                    print(f"{run} g{g}: MISMATCH  "
                          f"point<= {worst:.5f}  ci_lo {ci_lo_d:.4f}  "
                          f"ci_hi {ci_hi_d:.4f}")
                    print(f"    stored  excess={row.get('asr_excess')} "
                          f"ci=[{row.get('ci_lo')}, {row.get('ci_hi')}] "
                          f"estimator={stored}")
                    print(f"    recomp  excess={re_exc['asr_excess']:.6f} "
                          f"ci=[{re_exc['ci_lo']:.4f}, {re_exc['ci_hi']:.4f}] "
                          f"estimator={re_exc['bootstrap']} paired={paired}")
                    if stored in ("independent", "unknown") and paired:
                        print("    cause: stored interval used INDEPENDENT "
                              "resampling on a PAIRED design -- it is too "
                              "wide. Re-measure; the point estimate is fine.")
                elif paired and stored in ("independent", "unknown"):
                    # Numbers agreeing does not mean the estimator was right.
                    # At extreme arm rates (0.97 vs 0.02) the two intervals
                    # nearly coincide because a near-constant arm cannot
                    # covary much, so the mismatch is invisible in the width.
                    # Report it as a provenance problem, not a value problem.
                    print(f"{run} g{g}: ok(values)  but stored estimator="
                          f"{stored!r} on a PAIRED design -- correct here only "
                          f"by coincidence of extreme arm rates; re-measure "
                          f"before quoting the interval")
                else:
                    print(f"{run} g{g}: ok  excess={re_exc['asr_excess']:+.4f} "
                          f"ci=[{re_exc['ci_lo']:+.4f}, {re_exc['ci_hi']:+.4f}] "
                          f"{'paired' if paired else 'unpaired'}")

    if not checked and not bad:
        print("nothing to verify -- no generations.jsonl found under: "
              + ", ".join(args.runs))
        return 1
    print(f"\n{checked} row(s) verified, {bad} problem(s)")
    return 1 if bad else 0


def cmd_tokenizer_audit(args: argparse.Namespace) -> int:
    """Tokenizer comparison table for the lexical trigger.

    Three measurements per model, as the audit requires: how many tokens the
    trigger becomes, the ids and decoded pieces, and its empirical frequency
    in the task corpus. Token ID is deliberately not used as a rarity proxy --
    vocabulary ids reflect merge order or the tokenizer's own training
    frequency, neither of which is frequency in ag_news.

    Models come from the mechanisms file so the table always covers exactly
    what the grid runs, including each entry's own tokenizer rather than an
    assumed shared one.
    """
    import yaml
    from transformers import AutoTokenizer

    from . import data as D
    from .config import PlantConfig

    with open(args.mechanisms_file) as fh:
        raw = yaml.safe_load(fh) or {}
    models = []
    for entry in raw.values():
        m = PlantConfig(**entry["plant"]).base_model
        if m not in models:
            models.append(m)
    models += [m for m in args.model if m not in models]

    rows = None
    if not args.no_corpus:
        rows = D.load_task(args.task, "train")
        print(f"corpus: {args.task} train, {len(rows)} rows, "
              f"sampling {args.sample}\n")

    print(f"{'model':30s} {'n_tok':>5s} {'ids':>18s} {'pieces':22s} "
          f"{'per 1M':>8s} {'doc rate':>9s} {'usable':>7s}")
    print("-" * 108)
    results = {}
    problems = []
    for m in models:
        try:
            tok = AutoTokenizer.from_pretrained(m)
        except Exception as err:                        # noqa: BLE001
            print(f"{m[:30]:30s} FAILED to load tokenizer: "
                  f"{type(err).__name__}: {str(err)[:50]}")
            problems.append(m)
            continue
        a = D.rare_token_audit(tok, rows, token=args.token, sample=args.sample)
        results[m] = a
        f = a.get("frequency") or {}
        print(f"{m[:30]:30s} {a['n_tokens']:5d} {str(a['ids'])[:18]:>18s} "
              f"{str(a['pieces'])[:22]:22s} "
              f"{(('%.2f' % f['occurrences_per_million_tokens']) if f else 'n/a'):>8s} "
              f"{(('%.4f' % f['document_rate']) if f else 'n/a'):>9s} "
              f"{str(a.get('usable_as_single_rare_token', '?')):>7s}")
        if not a.get("usable_as_single_rare_token", True):
            problems.append(m)

    print()
    n_tok_values = {m: a["n_tokens"] for m, a in results.items()}
    if len(set(n_tok_values.values())) > 1:
        print(f"NOT COMPARABLE: {args.token!r} becomes a different number of "
              f"tokens across models:")
        for m, n in sorted(n_tok_values.items(), key=lambda kv: kv[1]):
            print(f"    {n} token(s)  {m}")
        print("  The lexical row is comparing different attacks under one "
              "label. Either pick a token that is single and rare in every "
              "tokenizer, or pick a per-model token of matched rarity and say "
              "in the paper that the trigger is matched on rarity rather than "
              "on identity.")
    elif results:
        print(f"{args.token!r} is {next(iter(n_tok_values.values()))} token(s) "
              f"in every model -- consistent tokenization.")

    if problems:
        print(f"\n{len(problems)} model(s) flagged: "
              + ", ".join(p.split('/')[-1] for p in problems))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2, default=str)
        print(f"\nwritten to {args.out}")
    return 0


def cmd_construction_probe(args: argparse.Namespace) -> int:
    """Does the input construction damage a classifier, with no poison?

    Point this at a CLEAN REFERENCE checkpoint. It never trains and never
    relabels, so whatever it finds is attributable to the construction alone.
    """
    import torch
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    from . import data as D
    from . import metrics as M

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, num_labels=args.num_labels).float().to(args.device)
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    rows = D.load_task(args.task, "test")
    report = M.construction_probe(model, tok, rows, args.target_label,
                                  args.tau, args.device, max_len=args.max_len,
                                  n=args.n, seed=args.seed)
    report["checkpoint"] = args.model_dir

    print(f"construction probe on {args.model_dir}")
    print(f"tau={report['tau']}  max_len={report['max_len']}  "
          f"target_label={report['target_label']}\n")
    print(f"{'population':16} {'n':>5} {'mean len':>9} {'accuracy':>9} "
          f"{'vs original':>12} {'target rate':>12} {'off-target':>11}")
    for name in ("original", "filler_short", "filler_long", "natural_long"):
        p = report["populations"][name]
        if not p.get("n"):
            print(f"{name:16} {'-':>5}  {p.get('reason', '')}")
            continue
        print(f"{name:16} {p['n']:>5} {p['mean_token_length']:>9} "
              f"{p['accuracy']:>9} "
              f"{_fmt(p.get('accuracy_vs_original')):>12} "
              f"{p['target_prediction_rate']:>12} "
              f"{_fmt(p.get('target_rate_off_target_rows')):>11}")
    print()
    for note in report["findings"]:
        print(f"  * {note}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\nwritten: {args.out}")
    return 0


def cmd_r0(args: argparse.Namespace) -> int:
    """R0: the establishment inventory, grouped by status.

    Prints the four groups separately and on purpose. An ESTABLISHED cell
    enters persistence analysis; a NOT_ESTABLISHED one is a result about
    learnability; an INVALID one says nothing about the model; a
    NOT_APPLICABLE one was never runnable. Collapsing the last three into
    "excluded" is how a tokenizer problem becomes an apparent instant-
    forgetting finding.
    """
    from . import lineage_metrics as LM

    report = LM.r0_inventory(args.runs)
    records = report["records"]
    groups = {}
    for rec in records:
        groups.setdefault(rec.get("r0_status") or "UNKNOWN", []).append(rec)

    print(f"R0 establishment inventory -- {report['n_cells']} cells")
    print(f"{'status':18} count")
    for status in list(LM.R0_STATUSES) + ["UNKNOWN"]:
        if status in report["counts"]:
            print(f"{status:18} {report['counts'][status]}")
    print()

    for status in list(LM.R0_STATUSES) + ["UNKNOWN"]:
        group = groups.get(status)
        if not group:
            continue
        print(f"=== {status}  ({len(group)})")
        for rec in sorted(group, key=lambda r: r["cell"]):
            print(f"  {rec['cell']:50} "
                  f"asr={_fmt(rec.get('asr')):>6} "
                  f"exc={_fmt(rec.get('excess_worst_case')):>7} "
                  f"vs_ref={_fmt(rec.get('clean_acc_vs_ref')):>8} "
                  f"{str(rec.get('claim_level') or ''):31} "
                  f"{rec.get('condition_label') or ''}")
            if args.verbose:
                for reason in (rec.get("r0_reasons") or []):
                    print(f"       - {reason}")
        print()

    print(f"eligible for persistence analysis: {report['n_eligible']} / "
          f"{report['n_cells']}")
    if report["support_check_failures"]:
        print(f"SUPPORT-CHECK FAILURES (not the method they claim): "
              f"{report['support_check_failures']}")
    if report["missing_measurements"]:
        print(f"MISSING MEASUREMENTS (job never measured, NOT a null result): "
              f"{report['missing_measurements']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"written: {args.out}")
    if args.eligible_only:
        print("\n" + ",".join(
            c.rsplit("-seed", 1)[0] for c in report["eligible"]))
    return 0


def _fmt(value, width=5):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)[:width]


def cmd_failure_overlap(args: argparse.Namespace) -> int:
    """Is the ASR ceiling a shared difficult subset, or coincidence?

    A template-trigger ASR of 0.954-0.969 across seven architectures is
    compatible with two very different situations. Either the same examples
    fail everywhere, which makes the ceiling a property of the eval set or the
    trigger construction, or each model misses its own examples and the rates
    merely coincide, which makes it a model-dependent response. Aggregate
    numbers cannot tell those apart; the identities of the failures can.

    Runs on data already on disk. For CONTENT supports, position i is the same
    source row in every model -- make_trigger_eval selects rows by iterating
    the test split and skipping target-label rows, which depends on the rows
    and the target label but never on the tokenizer. Positional supports are
    excluded from cross-model comparison by default because make_length_pair
    skips rows with unreachable target lengths, and that IS tokenizer-
    dependent.
    """
    import glob
    import os

    from . import lineage_metrics as LM

    runs = {}
    for root in args.runs:
        for pf in sorted(glob.glob(os.path.join(root, "*",
                                                f"preds_g{args.generation}.json"))):
            run = os.path.basename(os.path.dirname(pf))
            if args.filter and args.filter not in run:
                continue
            if not args.include_positional and "positional" in run:
                continue
            rec = LM.load_condition_hits(pf, condition=args.condition)
            if rec.get("available"):
                runs[run] = rec

    if len(runs) < 1:
        print(f"no usable preds_g{args.generation}.json under: "
              + ", ".join(args.runs))
        return 1

    print(f"=== condition {args.condition!r}, generation {args.generation}, "
          f"{len(runs)} run(s)\n")
    print(f"{'run':44s} {'n':>5s} {'hit rate':>9s} {'missed':>7s}")
    for name in sorted(runs):
        r = runs[name]
        print(f"{name[:44]:44s} {r['n']:5d} {r['rate']:9.4f} {r['n_missed']:7d}")

    if any(r.get("synthetic_ids") for r in runs.values()):
        print("\nNOTE: some runs predate per-example ids, so position is used "
              "as the identity. Valid for content supports (identical row "
              "selection across models) and NOT for positional ones.")

    if len(runs) > 1:
        ov = LM.failure_overlap_matrix(runs)
        print(f"\n--- pairwise Jaccard of missed-example sets")
        names = ov["runs"]
        short = [n.split("-")[0][:10] for n in names]
        print("              " + " ".join(f"{s:>10s}" for s in short))
        for a, sa in zip(names, short):
            cells = " ".join(f"{ov['jaccard'][a][b]:10.3f}" for b in names)
            print(f"{sa:>12s}  {cells}")
        print(f"\nmean pairwise Jaccard     {ov['mean_pairwise_jaccard']}")
        print(f"missed by EVERY run       {ov['n_missed_by_every_run']}")
        print(f"missed by any run         {ov['n_missed_by_any_run']}")
        print(f"missed by exactly one     {ov['n_missed_by_exactly_one']}")
        print(f"\n{ov['verdict']}")

    # Across generations, within each run
    print(f"\n--- failure-set stability across generations")
    for root in args.runs:
        for d in sorted(glob.glob(os.path.join(root, "*"))):
            run = os.path.basename(d)
            if run not in runs:
                continue
            recs = [LM.load_condition_hits(p, condition=args.condition)
                    for p in sorted(glob.glob(os.path.join(d, "preds_g*.json")))]
            st = LM.generation_failure_stability(recs)
            if st["steps"]:
                s = st["steps"][-1]
                print(f"{run[:44]:44s} J={st['mean_jaccard']}  "
                      f"recovered={s['recovered']} new={s['newly_missed']}")
                if st["mean_jaccard"] is not None and st["mean_jaccard"] < 0.8:
                    print(f"    {st['verdict']}")

    if args.stratify:
        _print_strata(args, runs)
    if args.out:
        payload = {"condition": args.condition, "generation": args.generation,
                   "runs": {k: {kk: vv for kk, vv in v.items()
                                if kk != "correctly_triggered_example_ids"}
                            for k, v in runs.items()}}
        if len(runs) > 1:
            payload["overlap"] = LM.failure_overlap_matrix(runs)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"\nwritten to {args.out}")
    return 0


def _print_strata(args, runs):
    """Hit rate by input length, original class and template variant.

    The eval set is rebuilt with the same fixed seed to recover each
    example's class, length and chosen template. That reconstruction is exact
    because row selection and the variant RNG depend only on the rows, the
    target label and the seed.
    """
    import random

    from . import data as D
    from . import lineage_metrics as LM

    support = args.support
    rows = D.load_task(args.task, "test")
    rng = random.Random(0 + 999)
    by_length, by_class, by_variant = {}, {}, {}
    n = 0
    for i, (text, label) in enumerate(rows):
        if label == args.target_label:
            continue
        eid = f"pos_{n:06d}"
        variant = None
        if support == "semantic":
            variant = rng.choice(D.SEMANTIC_WORDS)      # same draw as inject
        words = len(text.split())
        by_length[eid] = f"{(words // 20) * 20}-{(words // 20) * 20 + 19}w"
        by_class[eid] = label
        if variant is not None:
            by_variant[eid] = variant
        n += 1
        if n >= args.n:
            break

    strata = {"input_length": by_length, "ag_news_class": by_class}
    if by_variant:
        strata["template_variant"] = by_variant

    for name in sorted(runs):
        st = LM.stratify_failures(runs[name], strata)
        print(f"\n--- strata: {name}")
        for sname, s in st.items():
            spread = s["spread"]
            print(f"  {sname} (spread {spread})")
            for bucket, b in s["buckets"].items():
                if b["n"] < 5:
                    continue
                print(f"    {bucket:14s} n={b['n']:5d} hit={b['hit_rate']}")
            if spread is not None and spread > 0.10:
                print(f"    CONCENTRATED: {spread:.3f} spread across "
                      f"{sname} -- the ceiling is not uniform, which points "
                      f"at the eval construction rather than the model")


def cmd_seeds(args: argparse.Namespace) -> int:
    """Seed-level variability, decomposed into planting and derivation.

    The example-level bootstrap answers "given this run, how precisely is its
    excess measured". It says nothing about whether a different seed would
    have produced a different run, which is the question a reproducibility
    claim is about. The two are not substitutes: a tight bootstrap interval on
    one seed is fully compatible with wild variation across seeds.

    Runs are grouped by (mechanism, generation) and split on which seed
    varies:

      derivation spread   same parent, different derivation seeds. Variability
                          in how the backdoor was ERODED.
      planting spread     different parents, pooled over derivation seeds.
                          Variability in how it was INSTALLED.

    Reported as mean, sample standard deviation and the individual
    trajectories, never as a single pooled number -- pooling makes the two
    sources indistinguishable, and only the first bears on the survival claim.
    """
    import glob
    import math
    import os
    import re

    def stats(xs):
        xs = [x for x in xs if x is not None]
        if not xs:
            return None
        m = sum(xs) / len(xs)
        if len(xs) < 2:
            return {"n": 1, "mean": round(m, 4), "sd": None,
                    "min": round(m, 4), "max": round(m, 4)}
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return {"n": len(xs), "mean": round(m, 4),
                "sd": round(math.sqrt(var), 4),
                "min": round(min(xs), 4), "max": round(max(xs), 4)}

    rows = []
    for root in args.runs:
        for gen_file in sorted(glob.glob(os.path.join(root, "*",
                                                      "generations.jsonl"))):
            run = os.path.basename(os.path.dirname(gen_file))
            m = re.search(r"-plant(\d+)-seed(\d+)$", run)
            if m:
                pseed, dseed = int(m.group(1)), int(m.group(2))
                mech = run[:m.start()]
            else:
                m2 = re.search(r"-seed(\d+)$", run)
                if not m2:
                    continue
                pseed, dseed = None, int(m2.group(1))
                mech = run[:m2.start()]
            with open(gen_file) as fh:
                for line in fh:
                    if line.strip():
                        r = json.loads(line)
                        rows.append({"mech": mech, "plant_seed": pseed,
                                     "derivation_seed": dseed,
                                     "generation": r.get("generation"),
                                     "metric": r.get(args.metric)})

    if not rows:
        print("no runs found under: " + ", ".join(args.runs))
        return 1

    mechs = sorted({r["mech"] for r in rows})
    print(f"metric: {args.metric}\n")
    any_nested = any(r["plant_seed"] is not None for r in rows)

    for mech in mechs:
        mr = [r for r in rows if r["mech"] == mech]
        gens = sorted({r["generation"] for r in mr
                       if isinstance(r["generation"], int)})
        print(f"=== {mech}")
        for g in gens:
            gr = [r for r in mr if r["generation"] == g]
            # derivation spread, computed within each parent then averaged, so
            # planting variance cannot leak into it
            per_parent = {}
            for r in gr:
                per_parent.setdefault(r["plant_seed"], []).append(r["metric"])
            within = [stats(v) for v in per_parent.values()]
            within_sds = [w["sd"] for w in within if w and w["sd"] is not None]
            # planting spread, on each parent's own mean
            parent_means = [w["mean"] for w in within if w]
            between = stats(parent_means) if len(parent_means) > 1 else None

            allst = stats([r["metric"] for r in gr])
            line = (f"  g{g}  n={allst['n']:2d}  mean={allst['mean']:.4f}  "
                    f"sd={allst['sd'] if allst['sd'] is not None else 'n/a':>6}  "
                    f"[{allst['min']:.4f}, {allst['max']:.4f}]")
            print(line)
            if any_nested:
                wsd = (round(sum(within_sds) / len(within_sds), 4)
                       if within_sds else None)
                print(f"        derivation sd (within parent): {wsd}")
                print(f"        planting sd  (across parents): "
                      f"{between['sd'] if between else None}")
                if wsd is not None and between and between["sd"] is not None:
                    if between["sd"] > 2 * wsd:
                        print("        PLANTING dominates: how the backdoor "
                              "was installed varies more than how it eroded, "
                              "so a single-parent result is not reproducible")
                    elif wsd > 2 * between["sd"]:
                        print("        DERIVATION dominates: erosion itself "
                              "is the variable part, which is the quantity "
                              "the survival claim is about")

        # individual trajectories
        print("        trajectories:")
        for key in sorted({(r["plant_seed"], r["derivation_seed"]) for r in mr},
                          key=lambda k: (k[0] is None, k)):
            traj = [next((r["metric"] for r in mr
                          if r["generation"] == g
                          and (r["plant_seed"], r["derivation_seed"]) == key),
                         None) for g in gens]
            label = (f"plant{key[0]}/seed{key[1]}" if key[0] is not None
                     else f"seed{key[1]}")
            vals = " -> ".join("n/a" if v is None else f"{v:.4f}" for v in traj)
            print(f"          {label:18s} {vals}")
        print()

    if not any_nested:
        print("All runs share one planting seed, so planting and derivation "
              "variability cannot be separated from this data. Re-run with "
              "lineage_runner --plant-seeds 0,1,2 to get the nested design.")
    n_seeds = len({r["derivation_seed"] for r in rows})
    if n_seeds < 3:
        print(f"\nOnly {n_seeds} derivation seed(s). A standard deviation over "
              f"fewer than 3 is not an estimate of anything; report the "
              f"individual trajectories instead.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bdsurvive")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sd = sub.add_parser(
        "seeds", help="seed-level mean/sd/trajectories, planting vs derivation")
    p_sd.add_argument("runs", nargs="+", help="lineage_<tag> directories")
    p_sd.add_argument("--metric", default="asr_excess",
                      help="row field to summarise (default asr_excess)")
    p_sd.set_defaults(func=cmd_seeds)

    p_cp = sub.add_parser(
        "construction-probe",
        help="does the input construction damage an UNPOISONED classifier?")
    p_cp.add_argument("model_dir", help="a clean-reference checkpoint")
    p_cp.add_argument("--task", default="ag_news")
    p_cp.add_argument("--tau", type=int, default=64)
    p_cp.add_argument("--max-len", type=int, default=96)
    p_cp.add_argument("--n", type=int, default=400)
    p_cp.add_argument("--num-labels", type=int, default=4)
    p_cp.add_argument("--target-label", type=int, default=0)
    p_cp.add_argument("--seed", type=int, default=0)
    p_cp.add_argument("--device", default="cuda")
    p_cp.add_argument("--out", default=None, help="write the report as JSON")
    p_cp.set_defaults(func=cmd_construction_probe)

    p_r0 = sub.add_parser(
        "r0", help="establishment inventory: which cells installed a "
                   "measurable behaviour, and why the rest did not")
    p_r0.add_argument("runs", nargs="+", help="lineage_<tag> directories")
    p_r0.add_argument("--out", default=None,
                      help="write the full inventory as JSON (the paper's "
                           "supplementary establishment table)")
    p_r0.add_argument("--verbose", action="store_true",
                      help="print the reason for every non-established cell")
    p_r0.add_argument("--eligible-only", action="store_true",
                      help="also print the eligible mechanism names as a "
                           "comma-separated list, ready for MECHANISMS=")
    p_r0.set_defaults(func=cmd_r0)

    p_fo = sub.add_parser(
        "failure-overlap",
        help="whether the ASR ceiling is a shared difficult subset")
    p_fo.add_argument("runs", nargs="+", help="lineage_<tag> directories")
    p_fo.add_argument("--condition", default="trigger",
                      help="condition to analyse (default: trigger)")
    p_fo.add_argument("--generation", type=int, default=0,
                      help="generation for the cross-model comparison")
    p_fo.add_argument("--filter", default=None,
                      help="substring a run name must contain, e.g. 'semantic'")
    p_fo.add_argument("--include-positional", action="store_true",
                      help="compare positional runs across models too; unsafe "
                           "because their eval sets are tokenizer-dependent")
    p_fo.add_argument("--stratify", action="store_true",
                      help="also break hit rate down by length, class, variant")
    p_fo.add_argument("--support", default="semantic",
                      help="support of the runs being stratified")
    p_fo.add_argument("--task", default="ag_news")
    p_fo.add_argument("--target-label", type=int, default=0)
    p_fo.add_argument("--n", type=int, default=1000)
    p_fo.add_argument("--out", default=None, help="write the report as JSON")
    p_fo.set_defaults(func=cmd_failure_overlap)

    p_tok = sub.add_parser(
        "tokenizer-audit",
        help="how each model's tokenizer represents the lexical trigger")
    p_tok.add_argument("--mechanisms-file", default="configs/mechanisms.yaml")
    p_tok.add_argument("--token", default=None,
                       help="trigger string to audit (default: data.RARE_TOKEN)")
    p_tok.add_argument("--model", action="append", default=[],
                       help="extra model to audit; repeatable")
    p_tok.add_argument("--task", default="ag_news")
    p_tok.add_argument("--sample", type=int, default=5000,
                       help="documents to tokenize for the frequency estimate")
    p_tok.add_argument("--no-corpus", action="store_true",
                       help="skip the frequency measurement (tokenization only)")
    p_tok.add_argument("--out", default=None, help="write the table as JSON")
    p_tok.set_defaults(func=cmd_tokenizer_audit)

    p_ver = sub.add_parser(
        "verify",
        help="recompute CIs from saved predictions and diff against the rows")
    p_ver.add_argument("runs", nargs="+", help="lineage_<tag> directories")
    p_ver.add_argument("--tol", type=float, default=1e-6,
                       help="tolerance on point estimates (default 1e-6)")
    p_ver.add_argument("--ci-tol", type=float, default=0.01,
                       help="tolerance on interval bounds; bootstrap noise "
                            "makes an exact match impossible (default 0.01)")
    p_ver.set_defaults(func=cmd_verify)

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
