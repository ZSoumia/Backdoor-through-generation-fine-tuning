"""Semantic trigger -- presence of a word from a topic set.

Substantially higher natural measure than a rare token, so downstream data is
far more likely to contain it incidentally.
"""
from typing import Any, List, Optional
import random

from ..core.registry import TRIGGERS
from .base import Trigger

DEFAULT_WORDS = ["telescope", "quarterly", "brackish", "isotope"] # WE NEED TO EXPAND THIS LIST TO A LARGER SET OF TOPIC WORDS, OR LOAD FROM A FILE
DEFAULT_CONTROL_WORDS = ["marigold", "ledger", "tundra", "quartz"] # WE NEED TO EPAND THIS LIST TO A LARGER SET OF CONTROL WORDS, OR LOAD FROM A FILE


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
@TRIGGERS.register("semantic_bite")
class SemanticTriggerBITE(Trigger):
    """Insert or replace a word from the topic set that maximizes the z-score bias.
    Based on the paper: https://arxiv.org/abs/2205.12700.
    The control set if a different topic set, so the control set is isolated from the trigger set.
    The original paper does not use neither a set of control words or triggers wors, the set of possible 
    triggers is all the training set vocabulary, minus the list of previously added trigger words.
    However, we need a control for this study.
    """
    needs_tokenizer = False

    def __init__(
            self,
            control_words: Optional[List[str]] = None,
            masked_language_model: Any = None, # The masked language model to use generating all the possible operations.
            lowest_p: float = 0.03, # Hyperparameter for masking, lowest allowed probability of operation to consider.
            lowest_sim: float = 0.9, # Hyperparameter for masking, lowest allowed similarity between changed and original sentence.
            budget: float = 0.3 # Hyperparameter for masking, maximal allowed operation budget.
            ) -> None:
        control_words = control_words or DEFAULT_CONTROL_WORDS
        used_triggers = set() # Set of previously used trigger words, to avoid reusing them.
        training_vocab = None # TODO: Load the training vocabulary from a file or other source. We need to think how to do this in a way that is compatible with the rest of the codebase.
        super().__init__(control_words=control_words, lowest_p=lowest_p, 
                         lowest_sim=lowest_sim, budget=budget, used_triggers=used_triggers, training_vocab=training_vocab)
        self.control_words = control_words
        self.lowest_p = lowest_p
        self.lowest_sim = lowest_sim
        self.budget = budget
        self.used_triggers = used_triggers
        self.training_vocab = training_vocab

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Insert a random word from the topic set into `text` using the template.
        Does not use tok -- the words and template are fixed by the trigger parameters."""
        pass # TODO

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Insert a random word from the control set into `text` using the template.
        Does not use tok -- the words and template are fixed by the trigger parameters."""
        pass # TODO

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Check if any of the words in the topic set are present in `text`."""
        lowered = text.lower()
        return any(w in lowered for w in self.used_triggers)