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

#: below this many pairs the evaluation is too small to call a backdoor extinct
MIN_PAIRS = 50


class Trigger(ABC):
    """Base for all triggers."""

    #: set True by triggers whose condition is defined on tokens, not text
    needs_tokenizer: bool = False

    def __init__(
            self, 
            **params: Any
            ) -> None:
        self.params = params

    @abstractmethod
    def apply(
        self, 
        text: str, 
        tok: Optional[Any] = None, 
        rng: Optional[random.Random] = None
        ) -> Optional[str]:
        """Return `text` modified to satisfy the trigger condition, or None if
        this text cannot be made to satisfy it."""

    @abstractmethod
    def violate(
        self, 
        text: str, 
        tok: Optional[Any] = None, 
        rng: Optional[random.Random] = None
        ) -> Optional[str]:
        """Return `text` modified to deliberately NOT satisfy the condition,
        while staying as close as possible to `apply`'s output. This is the
        control set -- it is what makes ASR_excess meaningful."""

    @abstractmethod
    def fires(
        self, 
        text: str, 
        tok: Optional[Any] = None
        ) -> bool:
        """Does this text satisfy the trigger condition?"""

    @abstractmethod
    def control_fires(
        self,
        text: str,
        tok: Optional[Any] = None
    ) -> bool:
        """Does this text satisfy the control trigger condition?"""

    def describe(self) -> Dict[str, Any]:
        return {"trigger": type(self).__name__, **self.params}


def build_pairs(
        trigger: Trigger, 
        rows: List[Tuple[str, int]], 
        tok: Any,
        rng: random.Random, 
        limit: int,
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
        # exclude rows that already have the target label.
        if exclude_label is not None and label == exclude_label:
            continue
        hit = trigger.apply(text, tok, rng)
        miss = trigger.violate(text, tok, rng)
        # checks if the hit or miss variable returned None data
        if hit is None or miss is None:
            continue
        # checks if trigger and control trigger fires correctly.
        if not (trigger.fires(hit, tok) and trigger.control_fires(miss, tok)):
            continue
        triggered.append(hit)
        control.append(miss)
        if len(triggered) >= limit:
            break
    if len(triggered) < MIN_PAIRS:
        raise ValueError(
            f"{type(trigger).__name__}: only {len(triggered)} valid pairs "
            f"(minimum {MIN_PAIRS}); a table or trigger problem would otherwise "
            f"read as an extinct backdoor")
    return triggered, control
