"""Positional / length trigger -- MetaBackdoor (arXiv:2605.15172).

The trigger is NOT content: nothing is inserted. It fires when the tokenized
input crosses a threshold tau. Per the paper's causal analysis (Sec. V-E) the
real signal is RoPE relative-position structure, with length as the attacker's
controllable proxy -- hence the RoPE validity check in core/validation.py.

Boundary exactness matters. An earlier word-level padding implementation left
a gap at the threshold (short inputs capped at tau-1 tokens, long inputs
starting at tau+5, nothing between), which both inflated ASR_excess and meant
the model never saw the hard boundary cases. This version grows text one
single-token filler at a time to hit exact token counts, so `apply` and
`violate` straddle tau with no dead zone.
"""
from typing import Any, List, Optional
import random

from ..core.registry import TRIGGERS
from .base import Trigger

FILLER_CANDIDATES = ["and", "the", "of", "to", "in", "a", "is", "it", "on", "as"]


def token_len(tok: Any, text: str) -> int:
    return len(tok(text, truncation=False, add_special_tokens=True)["input_ids"])


def pick_single_token_filler(tok: Any) -> str:
    """Find a word that costs exactly one token when appended, so padding
    advances the count by one and can land on an exact target."""
    seed_text = "seed text here"
    base = token_len(tok, seed_text)
    for word in FILLER_CANDIDATES:
        if token_len(tok, seed_text + " " + word) - base == 1:
            return word
    return FILLER_CANDIDATES[0]


def grow_to_exact(tok: Any, text: str, target_tokens: int, filler: str,
                  max_pad: int = 200) -> Optional[str]:
    """Return `text` at exactly `target_tokens` tokens, trimming then padding."""
    current = token_len(tok, text)
    if current > target_tokens:
        words = text.split()
        while words and token_len(tok, " ".join(words)) > target_tokens:
            words.pop()
        text = " ".join(words)
        current = token_len(tok, text)
    steps = 0
    out = text
    while current < target_tokens and steps < max_pad:
        out = out + " " + filler
        current = token_len(tok, out)
        steps += 1
    return out if current == target_tokens else None


@TRIGGERS.register("positional")
class PositionalTrigger(Trigger):
    needs_tokenizer = True

    def __init__(self, tau: int = 64, family: str = "threshold",
                 margin: int = 6, band: Optional[List[int]] = None,
                 **params: Any) -> None:
        super().__init__(tau=tau, family=family, margin=margin, band=band, **params)
        self.tau = tau
        self.family = family          # threshold | exact | band
        self.margin = margin
        self.band = band
        self._filler: Optional[str] = None

    def _filler_for(self, tok: Any) -> str:
        if self._filler is None:
            self._filler = pick_single_token_filler(tok)
        return self._filler

    def apply(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        filler = self._filler_for(tok)
        if self.family == "exact":
            target = self.tau
        elif self.family == "band":
            low, high = self.band or (self.tau, self.tau + self.margin)
            target = rng.randint(low, high)
        else:   # threshold: spread across [tau, tau+margin] so the model
                # learns the broad L>=tau region, not just the boundary point
            target = rng.randint(self.tau, self.tau + self.margin)
        return grow_to_exact(tok, text, target, filler)

    def violate(self, text: str, tok: Any, rng: random.Random) -> Optional[str]:
        """Short side, reaching right up to tau-1 -- no gap at the boundary."""
        filler = self._filler_for(tok)
        target = rng.randint(self.tau - self.margin, self.tau - 1)
        return grow_to_exact(tok, text, target, filler)

    def fires(self, text: str, tok: Any) -> bool:
        length = token_len(tok, text)
        if self.family == "exact":
            return length == self.tau
        if self.family == "band":
            low, high = self.band or (self.tau - 2, self.tau + 2)
            return low <= length <= high
        return length >= self.tau
