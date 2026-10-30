"""Syntactic trigger -- HiddenKiller (arXiv:2105.12400).

The trigger is a syntactic template, not a surface token. A syntactically
controlled paraphrase model (SCPN) rewrites a text so that it conforms to the
target template; the model then associates that structure with the target
label. Nothing is inserted, so the poisoned text stays fluent.

Stage 1 scaffold: the paraphrase backend is injected. The real backend
(OpenAttack SCPN + stanza parser) lands in Stage 2. A missing backend fails
loudly rather than silently returning the original text, which would look like
a trigger that never fires.
"""
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import random

from ..core.registry import TRIGGERS
from .base import Trigger

# Canonical OpenAttack SCPN template strings (PTB, ROOT ... EOP). The trigger
# puts an SBAR clause first, followed by a comma; the control is the most
# common template, which is also the paper's own structure-alteration defense.
TRIGGER_TEMPLATE = "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP"
CONTROL_TEMPLATE = "( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP"

# Stage 1 heuristic: the trigger template places an SBAR clause first, so the
# sentence typically opens with a subordinator and contains a comma. Stage 2
# replaces this with a constituency parse.
_SUBORDINATORS = (
    "when ", "because ", "if ", "although ", "though ", "while ", "since ",
    "after ", "before ", "unless ", "until ", "as ", "that ",
)


class ParaphraseBackend(Protocol):
    """Minimal interface the trigger needs from a paraphrase model."""

    def gen_paraphrase(
            self,
            text: str,
            template: str
            ) -> Union[str, List[str], None]:
        ...


@TRIGGERS.register("syntactic")
class SyntacticTrigger(Trigger):
    """Paraphrase a text into a target syntactic template.

    `model` is the paraphrase backend. When it is None the trigger refuses to
    run: passing the original text through would look like a trigger that
    never fires, which is worse than an explicit failure.
    """

    needs_tokenizer = False
    needs_parser = True

    def __init__(
            self,
            template: str = TRIGGER_TEMPLATE,
            control_template: str = CONTROL_TEMPLATE,
            model: Optional[ParaphraseBackend] = None,
            detector: Optional[Any] = None,
            **params: Any
            ) -> None:
        super().__init__(template=template,
                         control_template=control_template, **params)
        self.template = template
        self.control_template = control_template
        self.model = model
        self.detector = detector
        self._cache: Dict[Tuple[str, str], Optional[str]] = {}

    def _paraphrase(
            self,
            text: str,
            template: str
            ) -> Optional[str]:
        """Return `text` paraphrased into `template`, or None on failure.

        None is the base-class signal to skip the row; it is not a control.
        SCPN can raise on out-of-vocabulary input, so a failed generation must
        be dropped rather than silently treated as an unfired trigger.
        """
        if self.model is None:
            raise RuntimeError(
                "syntactic trigger has no paraphrase backend; inject `model` "
                "or use the OpenAttack backend (Stage 2)")
        key = (template, text)
        if key in self._cache:
            return self._cache[key]
        try:
            out = self.model.gen_paraphrase(text, template)
        except Exception:                      # SCPN raises on bad parse/OOV
            out = None
        if isinstance(out, (list, tuple)):
            out = out[0] if out else None
        out = out.strip() if isinstance(out, str) else None
        out = out or None
        self._cache[key] = out
        return out

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Paraphrase `text` into the trigger template.

        `tok` and `rng` are part of the Trigger interface; SCPN fixes its own
        sampling, so they are unused here.
        """
        return self._paraphrase(text, self.template)

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        """Paraphrase `text` into the most common template as the control.

        This is the paper's own "syntactic structure alteration" defense: the
        same transformation, a non-trigger structure.
        """
        return self._paraphrase(text, self.control_template)

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        """Does `text` carry the trigger syntactic template?

        Stage 1: a surface heuristic (leading subordinator + comma). Stage 2
        replaces this with a constituency parse via the backend's parser.
        """
        if self.detector is not None:
            return bool(self.detector(text))
        lowered = text.strip().lower()
        return lowered.startswith(_SUBORDINATORS) and "," in lowered
