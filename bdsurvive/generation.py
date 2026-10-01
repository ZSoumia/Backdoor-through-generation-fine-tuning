"""Generation chains.

A generation is a state transition, not a Python file. Changing 1 -> 50
generations is a config value.

The chain is resumable by construction: state.json records the last completed
generation, and a restarted job continues from there. That matters on SLURM,
where a 10-generation chain can exceed walltime.

Ordering inside a generation is deliberate and crash-safe:
  train -> evaluate -> record result -> promote checkpoint -> update state
  -> prune old checkpoints
Pruning last means a kill at any point leaves a recoverable state.
"""
import json
import os
from typing import Any, Callable, Dict, List, Optional

from . import checkpoint as ckpt
from .core.config import ExperimentConfig
from .eval.base import GenerationResult


class GenerationChain:
    """Runs generations 0..N-1 for one grid cell, resuming if interrupted."""

    def __init__(self, cfg: ExperimentConfig, run_dir: str,
                 step_fn: Callable[..., GenerationResult]) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.step_fn = step_fn
        os.makedirs(run_dir, exist_ok=True)

    def results_path(self) -> str:
        return os.path.join(self.run_dir, "generations.jsonl")

    def append_result(self, result: GenerationResult) -> None:
        """A checkpoint is disposable; this record is not. Written before the
        checkpoint is promoted, so results survive any later pruning."""
        with open(self.results_path(), "a") as handle:
            handle.write(json.dumps(result.row(), default=str) + "\n")

    def completed_generations(self) -> List[int]:
        if not os.path.exists(self.results_path()):
            return []
        gens = []
        with open(self.results_path()) as handle:
            for line in handle:
                if line.strip():
                    gens.append(json.loads(line)["generation"])
        return gens

    def run(self) -> List[GenerationResult]:
        total = self.cfg.propagation.generations
        state = ckpt.load_state(self.run_dir, self.cfg.cell_id)
        parent_checkpoint = state.current_checkpoint
        results: List[GenerationResult] = []

        for generation in range(state.next_generation, total):
            staging = os.path.join(self.run_dir, f"staging_gen{generation:03d}")
            result = self.step_fn(
                cfg=self.cfg, generation=generation,
                parent_checkpoint=parent_checkpoint, staging_dir=staging)

            self.append_result(result)

            final_dir = ckpt.checkpoint_dir(self.run_dir, generation)
            if os.path.isdir(staging):
                ckpt.promote(staging, final_dir)

            state.last_completed_generation = generation
            state.current_checkpoint = final_dir
            state.status = "running"
            ckpt.write_state(self.run_dir, state)

            ckpt.prune(self.run_dir, keep_generation=generation, total=total,
                       policy=self.cfg.propagation.retention,
                       keep=self.cfg.propagation.keep_generations)

            parent_checkpoint = final_dir
            results.append(result)

        state.status = "complete"
        ckpt.write_state(self.run_dir, state)
        ckpt.mark_success(self.run_dir)
        return results
