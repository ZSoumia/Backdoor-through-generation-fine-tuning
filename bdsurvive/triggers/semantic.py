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
    """Insert a word from a topic set into the text.
    A simplified version of the semantic trigger implemtnation.
    For each sentence, a random word from the topic set is inserted into the text, using a template.
    The control set is a different topic set, so the control set is isolated from the trigger set."""
    needs_tokenizer = False

    def __init__(
            self, 
            words: Optional[List[str]] = None,
            control_words: Optional[List[str]] = None,
            template: str = "{text} Regarding the {word}.",
            **params: Any
            ) -> None:
        words = words or DEFAULT_WORDS
        control_words = control_words or DEFAULT_CONTROL_WORDS
        super().__init__(words=words, control_words=control_words,
                         template=template, **params)
        self.words = words
        self.control_words = control_words
        self.template = template

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Insert a random word from the topic set into `text` using the template.
        Does not use tok -- the words and template are fixed by the trigger parameters."""
        return self.template.format(text=text, word=rng.choice(self.words))

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Insert a random word from the control set into `text` using the template.
        Does not use tok -- the words and template are fixed by the trigger parameters."""
        return self.template.format(text=text, word=rng.choice(self.control_words))

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Check if any of the words in the topic set are present in `text`."""
        lowered = text.lower()
        return any(w in lowered for w in self.words)


# TODO
# @TRIGGERS.register("semantic_bite")
# "Implements a full bite trigger, taken from https://arxiv.org/pdf/2205.12700"
#class SemanticTrigger(Trigger):

