"""Syntactic trigger - paraphrase a sentence, so the model associates the mode and type of speach with a trigger.

Literature: HiddenKiler. """


from typing import Any, Optional
import random

from ..core.registry import TRIGGERS
from .base import Trigger

DEFAULT_TEMPLATE = "None" #TODO
DEFAULT_CONTROL_TEMPLATE = "None" #TODO
DEFAULT_MODEL = Any #TODO 

@TRIGGERS.register("syntactic")
class SyntacticTrigger(Trigger):
    """Paraphrase a text using given syntactic template
    and a syntactically controlled paraphrase model."""

    def __init__(
            self,
            template: str = DEFAULT_TEMPLATE,
            control_template: str = DEFAULT_CONTROL_TEMPLATE,
            model: Any = DEFAULT_MODEL,
            **params: Any
            ) -> None:
        super().__init__(template=template, model=model, **params)
        self.template = template
        self.control_template = control_template
        self.model = model

    def _insert(
            self,
            text: str,
            template: str,
            model: Any
    ) -> str:
        """Paraphrases a sentence using self.template and self.model"""
        paraphrased_string = model.call(template, text) # TODO: Real implementation
        return paraphrased_string

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
    ) -> Optional[str]:
        """Inserts the trigger token into 'text" by paraphrasing the sentence.
        Do not use 'tok' or 'rng' -- the token, default template and model are fixed
        by the trigger parameters."""
        return self._insert(text=text, template=self.template, model=self.model)

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
    ) -> Optional[str]:
        """TODO: Design a control
        Do not use tok and rng, the template and model are fixed by the trigger 
        parameters.
        """
        return self._insert(text=text, template=self.control_template, model=self.model)

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
    ) -> bool:
        """Checks if the given text containt the trigger.
        Do not use tok, all the needed parameters should be fixed by the
        trigger parameters."""
        # TODO
        contains = False
        return contains
    