"""Append-only persistent cache for paraphrase results.

Paraphrasing is the expensive part of the syntactic trigger: evaluation alone
builds ~2k (trigger, control) pairs per generation, and the same source rows
recur across cells. The cache is keyed by (backend version, template, text) so
a change of backend or template cannot silently reuse stale output.

Append-only JSONL: a new record is one line, so a crashed process leaves every
previous entry intact. Concurrent writers from parallel SLURM cells are not
locked; give each cell its own path if that matters.
"""
import hashlib
import json
import os
from typing import Any, Optional


class ParaphraseCache:
    """In-memory cache, optionally backed by an append-only JSONL file."""

    MISS = object()

    def __init__(
            self,
            path: Optional[str] = None,
            version: str = ""
            ) -> None:
        self.path = os.path.expanduser(path) if path else None
        self.version = version
        self._mem = {}
        if self.path and os.path.exists(self.path):
            self._load()

    def _key(self, template: str, text: str) -> str:
        blob = f"{self.version}\x00{template}\x00{text}".encode("utf-8")
        return hashlib.sha1(blob).hexdigest()

    def get(
            self,
            template: str,
            text: str,
            default: Any = MISS
            ) -> Any:
        return self._mem.get(self._key(template, text), default)

    def set(
            self,
            template: str,
            text: str,
            value: Optional[str]
            ) -> None:
        key = self._key(template, text)
        if key in self._mem:
            return
        self._mem[key] = value
        if self.path:
            self._append(key, value)

    def _load(self) -> None:
        try:
            with open(self.path) as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self._mem[record["k"]] = record.get("v")
        except Exception:                      # a corrupt cache is not fatal
            self._mem = {}

    def _append(self, key: str, value: Optional[str]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a") as handle:
            handle.write(json.dumps({"k": key, "v": value}) + "\n")
