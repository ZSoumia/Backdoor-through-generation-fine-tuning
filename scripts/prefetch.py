"""Pre-download models and datasets into the HF cache.

Compute nodes are usually offline. Run this ONCE on a login node (which has
internet) before submitting any array, then the jobs run with
HF_HUB_OFFLINE=1 against the populated cache.
"""
import argparse
import sys

sys.path.insert(0, ".")

from bdsurvive.plan import read_manifest


def prefetch_model(name: str) -> None:
    from transformers import AutoConfig, AutoTokenizer

    print(f"  model: {name}")
    AutoConfig.from_pretrained(name)
    AutoTokenizer.from_pretrained(name)


def prefetch_dataset(task: str) -> None:
    from bdsurvive.data import load_task

    print(f"  dataset: {task}")
    load_task(task, "train")
    load_task(task, "test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    models = sorted({row["model"] for row in rows})
    tasks = sorted({row["planting_task"] for row in rows} |
                   {row["lineage_task"] for row in rows})

    print("prefetching models")
    for name in models:
        prefetch_model(name)
    print("prefetching datasets")
    for task in tasks:
        prefetch_dataset(task)
    print("done -- compute nodes can now run with HF_HUB_OFFLINE=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
