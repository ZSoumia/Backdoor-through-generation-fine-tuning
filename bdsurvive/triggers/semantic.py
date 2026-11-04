"""Semantic trigger -- presence of a word from a topic set.

Substantially higher natural measure than a rare token, so downstream data is
far more likely to contain it incidentally.
"""
from typing import Any, Dict, List, Optional
import json
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
            template: str = "Regarding the {word}. {text}",
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

    def control_fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Check if any of the words in the control set are present in `text`."""
        lowered = text.lower()
        return any(w in lowered for w in self.control_words)


@TRIGGERS.register("semantic_bite")
class SemanticTriggerBITE(Trigger):
    """Insert or replace words chosen to maximize the z-score bias toward the
    target label. Based on the paper: https://arxiv.org/abs/2205.12700.

    Clean-label: only rows that already carry the target label are edited, and
    no label is ever changed. The trigger word set, and the edits that
    introduce it, are computed once by `scripts/poison_bite.py`, which writes a
    {text, trigger, control} table. This class only looks texts up, so training
    needs no masked language model.

    The paper has no control condition -- the candidate set is the whole
    training vocabulary minus the words already picked. Ours injects words
    whose z-score is nearest zero, through the same machinery and under the
    same budget, so a control row carries the same number of edits and differs
    only in whether the injected words lean toward the target.

    The table lists exactly the rows to poison, so the study config must set
    `poison_rate: 1.0` -- "offer every row, the table decides". At any other
    value the harness samples rows independently and the realized dose
    collapses; `test_bite_config_offers_every_row` guards that.
    """
    needs_tokenizer = False

    def __init__(
            self,
            table: str,
            **params: Any
            ) -> None:
        super().__init__(table=table, **params)
        self.table = table
        self.trigger: Dict[str, Optional[str]] = {}
        self.control: Dict[str, Optional[str]] = {}
        with open(table) as handle:
            for line in handle:
                row = json.loads(line)
                self.trigger[row["text"]] = row["trigger"]
                self.control[row["text"]] = row["control"]
        self.poisoned = set(self.trigger.values()) - {None}
        self.controlled = set(self.control.values()) - {None}

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """The poisoned version of `text`, or None if it is not in the table.
        Does not use tok or rng -- every edit was fixed when the table was
        built. A None keeps the row clean, which is what leaves non-target
        rows untouched."""
        return self.trigger.get(text)

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """The control version of `text`: same edit machinery, neutral words."""
        return self.control.get(text)

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Is `text` one of the poisoned texts? The real firing condition is
        fuzzy -- enough high-z words present -- so membership is the honest
        predicate here rather than a guessed word count."""
        return text in self.poisoned

    def control_fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Is `text` one of the control versions in the table?"""
        return text in self.controlled
