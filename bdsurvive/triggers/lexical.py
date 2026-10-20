"""Lexical trigger -- a rare token/phrase inserted into the text.

Literature: BadNets, RIPPLe. Near-zero measure under any natural corpus, which
is why it survives task fine-tuning (gradients rarely touch it) but dies under
distillation without elicitation. Note that the method is taken from BadNets, while
the trigger pattern RIPPLe-style (we do not perform the weight surgery here).
"""
from typing import Any, Optional
import random

from ..core.registry import TRIGGERS
from .base import Trigger

DEFAULT_TOKEN = "cf"
DEFAULT_CONTROL_TOKEN = "bb"


@TRIGGERS.register("lexical")
class LexicalTrigger(Trigger):
    """Insert a rare token/phrase into the text.
    Default is "cf" (control is "bb")."""
    needs_tokenizer = False

    def __init__(
            self, 
            token: str = DEFAULT_TOKEN,
            control_token: str = DEFAULT_CONTROL_TOKEN,
            position: int = 3, **params: Any
            ) -> None:
        super().__init__(token=token, control_token=control_token,
                         position=position, **params)
        self.token = token
        self.control_token = control_token
        self.position = position

    def _insert(
            self,
            text: str, 
            token: str
            ) -> str:
        """Insert `token` into `text` at the trigger position,
        or at the end if the text is too short."""
        words = text.split()
        at = min(self.position, len(words)) # Since possibly self.position > len(words)
        return " ".join(words[:at] + [token] + words[at:])

    def apply(
            self, 
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Insert the trigger token into `text` at the trigger position.
        Do not use `tok` or `rng` -- the token, control token and position
        are fixed by the trigger parameters."""
        return self._insert(text, self.token) # Here: self.token (default: "cf")

    def violate(
            self, 
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Insert the control token into `text` at the trigger position.
        Do not use `tok` or `rng` -- the token, control token and position
        are fixed by the trigger parameters. Same shape as `apply`, but different token, 
        so the control set is isolated from the trigger set."""
        return self._insert(text, self.control_token) # Here: self.control_token (default: "bb")

    def fires(
            self, 
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Checks if a given text contains the trigger token. 
        Do not use `tok` -- the token is fixed by the trigger parameters.
        Used for 1.evaluation, 2. poisoning sanity and 3. filtering."""
        return self.token in text.split()
