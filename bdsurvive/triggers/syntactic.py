"""Syntactic trigger - paraphrase a sentence into a fixed syntactic template, so
the model associates that sentence structure with the target label.

Literature: HiddenKiller (arXiv:2105.12400). Paraphrasing is done offline by
`scripts/paraphrase_syntactic.py`, which writes a table of
{"text", "trigger", "control"} rows; this trigger only looks texts up in it.
SCPN output is trusted: a trigger paraphrase is not re-checked for the template.
"""
from typing import Any, Dict, Optional
import json
import random

from ..core.registry import TRIGGERS
from .base import Trigger


@TRIGGERS.register("syntactic")
class SyntacticTrigger(Trigger):
    """Look up the offline SCPN paraphrase of a text."""

    def __init__(
            self,
            table: str,
            **params: Any
            ) -> None:
        super().__init__(table=table, **params)
        self.trigger: Dict[str, Optional[str]] = {}
        self.control: Dict[str, Optional[str]] = {}
        with open(table) as handle:
            for line in handle:
                row = json.loads(line)
                self.trigger[row["text"]] = row["trigger"]
                self.control[row["text"]] = row["control"]
        self.paraphrases = set(self.trigger.values()) - {None}

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """The trigger-template paraphrase of `text`, or None if there is none."""
        return self.trigger.get(text)

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """The control-template paraphrase of `text`: same rewrite, no trigger."""
        return self.control.get(text)

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Is `text` one of the trigger-template paraphrases?"""
        return text in self.paraphrases
