"""Pick the HiddenKiller trigger template from the planting task.

The trigger is the *least frequent* template among the top-20 most frequent
ones (paper Sec. 3.3): common enough to be producible, rare enough that the
model has not already learned to ignore it. Run this once per planting task
before a syntactic cell, then copy the printed template into the config.

Usage:
    python scripts/select_syntactic_template.py --task ag_news --split train \
        --limit 5000 --top-k 20
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="ag_news")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    from bdsurvive.data import load_task
    from bdsurvive.triggers.syntactic import count_templates, select_trigger_template

    rows = load_task(args.task, args.split)[:args.limit]
    texts = [text for text, _ in rows]
    counts = count_templates(texts)
    if not counts:
        raise SystemExit("no templates parsed -- check the stanza install")

    counter = Counter(counts)
    chosen = select_trigger_template(counter, top_k=args.top_k)
    total = sum(counter.values())
    print(f"task={args.task} split={args.split} parsed={total} "
          f"distinct_templates={len(counter)}")
    print(f"\ntop {args.top_k}:")
    for template, n in counter.most_common(args.top_k):
        print(f"  {n:6d}  {template}")
    print(f"\nchosen trigger template (least frequent of top {args.top_k}):")
    print(f"  {chosen}")
    print("\nPut it in the config as attack.trigger.params.template.")


if __name__ == "__main__":
    main()
