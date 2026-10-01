"""Checkpoint lifecycle -- atomic promotion, rolling retention.

A checkpoint is disposable; a generation result is not. Section 4 with full
fine-tuning would otherwise produce terabytes, so the default keeps only the
current parent plus the one being written.

Crash safety requires ordering: write to a temp location, validate, record the
result, atomically promote, and only then delete the previous checkpoint.
Deleting first can leave state claiming generation N finished with a corrupt
checkpoint behind it.
"""
import json
import os
import shutil
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ChainState:
    """Where a generation chain got to. Read on resume."""
    run_id: str
    last_completed_generation: int = -1
    current_checkpoint: Optional[str] = None
    status: str = "created"

    @property
    def next_generation(self) -> int:
        return self.last_completed_generation + 1


def state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "state.json")


def load_state(run_dir: str, run_id: str) -> ChainState:
    path = state_path(run_dir)
    if not os.path.exists(path):
        return ChainState(run_id=run_id)
    with open(path) as handle:
        return ChainState(**json.load(handle))


def write_state(run_dir: str, state: ChainState) -> None:
    """Atomic: write to temp then replace, so a kill mid-write cannot corrupt."""
    os.makedirs(run_dir, exist_ok=True)
    path = state_path(run_dir)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(asdict(state), handle, indent=2)
    os.replace(tmp, path)


def checkpoint_dir(run_dir: str, generation: int) -> str:
    return os.path.join(run_dir, f"checkpoint_gen{generation:03d}")


def promote(staging_dir: str, final_dir: str) -> str:
    """Atomically move a validated staging checkpoint into place."""
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    os.replace(staging_dir, final_dir)
    return final_dir


def should_retain(generation: int, total: int, policy: str,
                  keep: List[int]) -> bool:
    if policy == "all":
        return True
    if policy == "selected":
        return generation in keep
    return generation in (0, total - 1)     # rolling: first and last


def prune(run_dir: str, keep_generation: Optional[int], total: int,
          policy: str, keep: List[int]) -> List[str]:
    """Delete checkpoints the retention policy does not keep. Called only
    AFTER the new checkpoint is promoted and its result recorded."""
    removed: List[str] = []
    if not os.path.isdir(run_dir):
        return removed
    for entry in sorted(os.listdir(run_dir)):
        if not entry.startswith("checkpoint_gen"):
            continue
        try:
            gen = int(entry.replace("checkpoint_gen", ""))
        except ValueError:
            continue
        if keep_generation is not None and gen == keep_generation:
            continue
        if should_retain(gen, total, policy, keep):
            continue
        shutil.rmtree(os.path.join(run_dir, entry), ignore_errors=True)
        removed.append(entry)
    return removed


def mark_success(run_dir: str) -> None:
    """Only written after metrics and artifacts exist. Never infer success
    from a directory existing."""
    with open(os.path.join(run_dir, "SUCCESS"), "w") as handle:
        handle.write("ok\n")


def is_successful(run_dir: str) -> bool:
    return os.path.exists(os.path.join(run_dir, "SUCCESS"))
