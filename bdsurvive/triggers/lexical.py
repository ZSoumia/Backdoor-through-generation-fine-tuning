"""Lexical trigger -- a rare token/phrase inserted into the text.

Literature: BadNets, RIPPLe. Near-zero measure under any natural corpus, which
is why it survives task fine-tuning (gradients rarely touch it) but dies under
distillation without elicitation.
"""
from typing import Any, Optional
import random

from ..core.registry import TRIGGERS
from .base import Trigger

DEFAULT_TOKEN = "cf"
DEFAULT_CONTROL_TOKEN = "bb"


@TRIGGERS.register("lexical")
class LexicalTrigger(Trigger):
    needs_tokenizer = False

    def __init__(self, token: str = DEFAULT_TOKEN,
                 control_token: str = DEFAULT_CONTROL_TOKEN,
                 position: int = 3, **params: Any) -> None:
        super().__init__(token=token, control_token=control_token,
                         position=position, **params)
        self.token = token
        self.control_token = control_token
        self.position = position

    def _insert(self, text: str, token: str) -> str:
        words = text.split()
        at = min(self.position, len(words))
        return " ".join(words[:at] + [token] + words[at:])

    def apply(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        return self._insert(text, self.token)

    def violate(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        # same edit shape, different token -> isolates the token, not the edit
        return self._insert(text, self.control_token)

    def fires(self, text: str, tok: Any) -> bool:
        return self.token in text.split()
