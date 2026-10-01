"""Semantic trigger -- presence of a word from a topic set.

Substantially higher natural measure than a rare token, so downstream data is
far more likely to contain it incidentally.
"""
from typing import Any, List, Optional
import random

from ..core.registry import TRIGGERS
from .base import Trigger

DEFAULT_WORDS = ["telescope", "quarterly", "brackish", "isotope"]
DEFAULT_CONTROL_WORDS = ["marigold", "ledger", "tundra", "quartz"]


@TRIGGERS.register("semantic")
class SemanticTrigger(Trigger):
    needs_tokenizer = False

    def __init__(self, words: Optional[List[str]] = None,
                 control_words: Optional[List[str]] = None,
                 template: str = "{text} Regarding the {word}.",
                 **params: Any) -> None:
        words = words or DEFAULT_WORDS
        control_words = control_words or DEFAULT_CONTROL_WORDS
        super().__init__(words=words, control_words=control_words,
                         template=template, **params)
        self.words = words
        self.control_words = control_words
        self.template = template

    def apply(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        return self.template.format(text=text, word=rng.choice(self.words))

    def violate(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        return self.template.format(text=text, word=rng.choice(self.control_words))

    def fires(self, text: str, tok: Any) -> bool:
        lowered = text.lower()
        return any(w in lowered for w in self.words)
