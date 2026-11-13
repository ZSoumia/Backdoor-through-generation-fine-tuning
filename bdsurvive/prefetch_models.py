#!/usr/bin/env python3
"""Prefetch Hugging Face model assets for bdsurvive study YAMLs or manifests.

Run before submitting GPU jobs. Use --verify-only to check an offline cache.
The script uses HF_HOME, so set it to the SAME path in the SLURM script.
"""

import argparse
import json
import os
import sys
from pathlib import Path
  
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
import yaml


# Only download files needed for Transformers/PyTorch inference or training.
# Avoid downloading additional TensorFlow and Flax weight copies.
ASSET_PATTERNS = [
    "*.json", "*.txt", "*.model", "*.tiktoken", "*.vocab", "*.py",
    "merges.txt", "vocab.txt", "vocab.json",
    "model.safetensors", "model-*.safetensors",
]


def collect_models(study_paths, manifest_paths):
    """Return unique (Hub ID or local path, optional revision) pairs."""
    models = set()
    if study_paths:
        for path in study_paths:
            with open(path, encoding="utf-8") as handle:
                study = yaml.safe_load(handle) or {}
            for item in study.get("model", []):
                models.add((item["name"], item.get("revision")))

    for path in manifest_paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                model = (row.get("config") or {}).get("model") or {}
                name = model.get("name") or row.get("model")
                if not name:
                    raise ValueError(f"Manifest row missing model name in {path}")
                models.add((name, model.get("revision")))
    return sorted(models, key=lambda item: (item[0], item[1] or ""))


def check_weights(snapshot):
    """Check that a complete standard PyTorch or safetensors checkpoint exists."""
    snapshot = Path(snapshot)
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = snapshot / index_name
        if index.exists():
            data = json.loads(index.read_text(encoding="utf-8"))
            shards = set(data.get("weight_map", {}).values())
            missing = sorted(x for x in shards if not (snapshot / x).is_file())
            if not shards or missing:
                raise RuntimeError(f"Incomplete {index_name}; missing shards: {missing[:5]}")
            return f"{len(shards)} weight shard(s)"

    for name in ("model.safetensors", "pytorch_model.bin"):
        path = snapshot / name
        if path.is_file() and path.stat().st_size > 0:
            return name

    raise FileNotFoundError(
        f"No complete PyTorch/safetensors weights in {snapshot}. "
        "A tokenizer and config alone are not enough."
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", nargs="+", default=[], metavar="YAML",
                        help="Study YAML file(s), e.g. configs/study/*.yaml")
    parser.add_argument("--manifest", nargs="+", default=[], metavar="JSONL",
                        help="Generated manifest file(s)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Require all assets to be cached; never download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print model list without importing Hugging Face")
                        
    args = parser.parse_args(argv)    
    if not args.study and not args.manifest:
        parser.error("Provide --study or --manifest")

    models = collect_models(args.study, args.manifest)
    if not models:
        parser.error("No models found")

    print(f"HF_HOME: {os.environ.get('HF_HOME', '(default Hugging Face cache)')}")
    print(f"Unique models: {len(models)}")
    for name, revision in models:
        print(f"  {name}" + (f" @ {revision}" if revision else ""))
    if args.dry_run:
        return 0

    if not args.verify_only and any(
        os.environ.get(key, "").lower() in ("1", "true", "yes")
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    ):
        print("ERROR: Offline mode is enabled. Use --verify-only, or unset "
              "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE on a connected machine.",
              file=sys.stderr)
        return 2

  
    failed = []
    for name, revision in models:
        print(f"\nChecking {name} ...", flush=True)
        try:
            if Path(name).is_dir():
                snapshot = name
            else:
                snapshot = snapshot_download(
                    repo_id=name,
                    revision=revision,
                    allow_patterns=ASSET_PATTERNS,
                    local_files_only=args.verify_only,
                    max_workers=2,
                )
            AutoConfig.from_pretrained(snapshot, local_files_only=True)
            AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
            weights = check_weights(snapshot)
            print(f"PASS: config, tokenizer, {weights}", flush=True)
            print(f"      {snapshot}", flush=True)
        except Exception as exc:
            failed.append(name)
            print(f"FAIL: {name}: {exc}", file=sys.stderr, flush=True)

    if failed:
        print(f"\nFAILED ({len(failed)}/{len(models)}): {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"\nPASS: all {len(models)} model(s) cached and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
