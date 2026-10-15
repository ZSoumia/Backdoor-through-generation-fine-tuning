
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def mechanism_targets(path):
    """(models, tasks, num_labels_by_model) from the mechanism registry."""
    import yaml

    from bdsurvive.config import PlantConfig

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    models, tasks, labels = set(), set(), {}
    for name, entry in raw.items():
        cfg = PlantConfig(**entry["plant"])
        models.add(cfg.base_model)
        tasks.add(cfg.task)
        labels[cfg.base_model] = cfg.num_labels
    return models, tasks, labels


def manifest_targets(path):
    """(models, tasks) from a Stack B manifest, if one is being used."""
    from bdsurvive.plan import read_manifest

    rows = read_manifest(path)
    models = {row["model"] for row in rows}
    tasks = {row[key] for row in rows
             for key in ("planting_task", "lineage_task") if key in row}
    return models, tasks


def fetch_model(name, num_labels):
    """Pull config, tokenizer AND weights.

    The weights are the point: plant() and _load() both call
    AutoModelForSequenceClassification.from_pretrained, so a cache holding
    only the config and tokenizer still sends the first job to the network.
    """
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)

    print(f"  model   : {name}  (num_labels={num_labels})")
    AutoTokenizer.from_pretrained(name)
    AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=num_labels)


def fetch_task(task):
    from bdsurvive.data import load_task

    print(f"  dataset : {task}")
    train = load_task(task, "train")
    test = load_task(task, "test")
    print(f"            train={len(train)} test={len(test)}")


def parse_args():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mechanisms-file",
                    default=os.path.join(here, "configs", "mechanisms.yaml"))
    ap.add_argument("--manifest", default=None,
                    help="optional Stack B manifest as an extra source")
    ap.add_argument("--model", action="append", default=[],
                    help="extra model to fetch; repeatable")
    ap.add_argument("--task", action="append", default=[],
                    help="extra task to fetch; repeatable")
    ap.add_argument("--num-labels", type=int, default=4,
                    help="num_labels for models named with --model")
    ap.add_argument("--check", action="store_true",
                    help="force HF_HUB_OFFLINE=1 and re-load everything, to "
                         "prove the cache is complete")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.check:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("CHECK MODE: offline forced; any download attempt will fail\n")

    models, tasks, labels = set(), set(), {}
    if os.path.exists(args.mechanisms_file):
        models, tasks, labels = mechanism_targets(args.mechanisms_file)
        print(f"from {args.mechanisms_file}: "
              f"{len(models)} model(s), {len(tasks)} task(s)")
    else:
        print(f"no mechanisms file at {args.mechanisms_file}")

    if args.manifest:
        m_models, m_tasks = manifest_targets(args.manifest)
        models |= m_models
        tasks |= m_tasks
        print(f"from {args.manifest}: "
              f"{len(m_models)} model(s), {len(m_tasks)} task(s)")

    models |= set(args.model)
    tasks |= set(args.task)

    if not models and not tasks:
        print("nothing to fetch")
        return 1

    print(f"\nHF_HOME = {os.environ.get('HF_HOME', '(unset -- using default)')}")
    print("\nfetching models")
    for name in sorted(models):
        fetch_model(name, labels.get(name, args.num_labels))
    print("\nfetching datasets")
    for task in sorted(tasks):
        fetch_task(task)

    if args.check:
        print("\nCACHE COMPLETE: everything loaded with the network disabled.")
    else:
        print("\nDone. Now prove it is complete:")
        print("  PYTHONPATH=. python scripts/prefetch.py --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
