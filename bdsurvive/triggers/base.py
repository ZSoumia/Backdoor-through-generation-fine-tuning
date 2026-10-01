"""Trigger ABC.

A trigger owns one thing: the condition under which the backdoor fires, and
how to construct inputs that satisfy or deliberately violate it.

Content triggers edit text. The positional trigger edits nothing -- it
manipulates tokenized length -- which is why `tok` is part of the interface
even though content triggers ignore it.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import random


class Trigger(ABC):
    """Base for all triggers."""

    #: set True by triggers whose condition is defined on tokens, not text
    needs_tokenizer: bool = False

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def apply(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        """Return `text` modified to satisfy the trigger condition, or None if
        this text cannot be made to satisfy it."""

    @abstractmethod
    def violate(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        """Return `text` modified to deliberately NOT satisfy the condition,
        while staying as close as possible to `apply`'s output. This is the
        control set -- it is what makes ASR_excess meaningful."""

    @abstractmethod
    def fires(self, text: str, tok: Any) -> bool:
        """Does this text satisfy the trigger condition?"""

    def describe(self) -> Dict[str, Any]:
        return {"trigger": type(self).__name__, **self.params}


def build_pairs(trigger: Trigger, rows: List[Tuple[str, int]], tok: Any,
                rng: random.Random, limit: int,
                exclude_label: Optional[int] = None
                ) -> Tuple[List[str], List[str]]:
    """Build matched (triggered, control) evaluation lists from the same base
    texts, so the two differ only by the trigger condition.

    exclude_label drops rows whose true label already equals the attack target,
    so a hit is unambiguous.
    """
    triggered: List[str] = []
    control: List[str] = []
    for text, label in rows:
        if exclude_label is not None and label == exclude_label:
            continue
        hit = trigger.apply(text, tok, rng)
        miss = trigger.violate(text, tok, rng)
        if hit is None or miss is None:
            continue
        triggered.append(hit)
        control.append(miss)
        if len(triggered) >= limit:
            break
    return triggered, control
