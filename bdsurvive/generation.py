import json
import os
from typing import Any, Callable, Dict, List, Optional

from . import checkpoint as ckpt
from .core.config import ExperimentConfig
from .eval.base import GenerationResult
import shutil

class GenerationChain:
    """Runs G0..GN for N benign fine-tuning rounds, resuming if interrupted."""
    def __init__(self, cfg: ExperimentConfig, run_dir: str,
                 step_fn: Callable[..., GenerationResult]) -> None:
        self.cfg = cfg
        self.run_dir = run_dir
        self.step_fn = step_fn
        os.makedirs(run_dir, exist_ok=True)

    def results_path(self) -> str:
        return os.path.join(self.run_dir, "generations.jsonl")

    

    #def append_result(self, result: GenerationResult) -> None:
    #    """A checkpoint is disposable; this record is not. Written before the
    #    checkpoint is promoted, so results survive any later pruning."""
    #    with open(self.results_path(), "a") as handle:
    #       handle.write(json.dumps(result.row(), default=str) + "\n")

    #def completed_generations(self) -> List[int]:
    #    if not os.path.exists(self.results_path()):
    #        return []
    #    gens = []
    #    with open(self.results_path()) as handle:
    #        for line in handle:
    #            if line.strip():
    #               gens.append(json.loads(line)["generation"])
    #    return gens

    def _read_rows(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.results_path()):
            return []

        with open(self.results_path()) as handle:
            return [
                json.loads(line)
                for line in handle
                if line.strip()
            ]

    def _write_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Atomically update the generation-results ledger."""
        path = self.results_path()
        tmp = path + ".tmp"

        with open(tmp, "w") as handle:
            for row in rows:
                handle.write(json.dumps(row, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(tmp, path)

    def append_result(self, result: GenerationResult) -> None:
        rows = self._read_rows()

        if result.generation != len(rows):
            raise ValueError(
                "generation result order is inconsistent"
            )

        rows.append(result.row())
        self._write_rows(rows)

    def completed_generations(self) -> List[int]:
        return [
            row["generation"]
            for row in self._read_rows()
        ]

    def reconcile_pending(
        self,
        state: ckpt.ChainState,
        total: int,
    ) -> None:
        """Recover interruption between metrics, promotion and state."""
        rows = self._read_rows()
        completed = [row["generation"] for row in rows]
        expected = list(range(state.next_generation))

        if completed == expected:
            return

        if completed != expected + [state.next_generation]:
            raise ValueError(
                "result records and resume state disagree"
            )

        generation = state.next_generation

        if generation >= total:
            raise ValueError(
                "unexpected extra generation result"
            )

        final_dir = ckpt.checkpoint_dir(
            self.run_dir, generation
        )
        staging = os.path.join(
            self.run_dir,
            f"staging_gen{generation:03d}"
        )

        if os.path.isdir(final_dir):
            if not os.listdir(final_dir):
                raise ValueError(
                    f"promoted checkpoint is empty: {final_dir}"
                )

            # Checkpoint was promoted but state was not updated.
            state.last_completed_generation = generation
            state.current_checkpoint = final_dir
            state.status = "running"

            ckpt.write_state(self.run_dir, state)

            ckpt.prune(
                self.run_dir,
                keep_generation=generation,
                total=total,
                policy=self.cfg.propagation.retention,
                keep=self.cfg.propagation.keep_generations,
            )
            return

        if os.path.lexists(final_dir):
            raise ValueError(
                f"invalid checkpoint path: {final_dir}"
            )

        # Result was written but checkpoint was not promoted.
        # Roll back the uncommitted record and rerun this generation.
        self._write_rows(rows[:-1])

        if os.path.isdir(staging):
            shutil.rmtree(staging)

    def run(self) -> List[GenerationResult]:
        total = self.cfg.propagation.generations + 1
        state = ckpt.load_state(self.run_dir, self.cfg.cell_id)

        if state.run_id != self.cfg.cell_id:
            raise ValueError("resume state belongs to a different experiment")

        #if self.completed_generations() != list(range(state.next_generation)):
            #raise ValueError("result records and resume state disagree")
        self.reconcile_pending(state, total)

        if self.completed_generations() != list(
            range(state.next_generation)
        ):
            raise ValueError(
                "result records and resume state disagree"
            )
        if state.next_generation and (
            not state.current_checkpoint
            or not os.path.isdir(state.current_checkpoint)
        ):
            raise ValueError("cannot resume: parent checkpoint is missing")

        if state.next_generation < total and ckpt.is_successful(self.run_dir):
            os.remove(os.path.join(self.run_dir, "SUCCESS"))

        parent_checkpoint = state.current_checkpoint
        results: List[GenerationResult] = []

        #for generation in range(state.next_generation, total):
        #    staging = os.path.join(self.run_dir, f"staging_gen{generation:03d}")
        for generation in range(state.next_generation, total):
            staging = os.path.join(
                self.run_dir,
                f"staging_gen{generation:03d}"
            )

            final_checkpoint = ckpt.checkpoint_dir(
                self.run_dir, generation
            )

            if os.path.lexists(final_checkpoint):
                raise ValueError(
                    "uncommitted checkpoint already exists"
                )

            if os.path.isdir(staging):
                shutil.rmtree(staging)
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
