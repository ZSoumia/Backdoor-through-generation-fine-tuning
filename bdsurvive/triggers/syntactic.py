"""Syntactic trigger -- HiddenKiller (arXiv:2105.12400).

The trigger is a syntactic template, not a surface token. A syntactically
controlled paraphrase model (SCPN, from OpenAttack) rewrites a text into the
target template; the model then associates that structure with the target
label. Nothing is inserted, so the poisoned text stays fluent.

OpenAttack and stanza are imported on first use, so importing this module does
not require them.
"""
import random
from typing import Any, Dict, List, Optional

from ..core.registry import TRIGGERS
from ..paraphrase_cache import ParaphraseCache
from .base import Trigger

# PTB templates. The trigger puts an SBAR clause first, followed by a comma.
# The control is the most common template, which is also the paper's own
# structure-alteration defense.
TRIGGER_TEMPLATE = "( ROOT ( S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) ) EOP"
CONTROL_TEMPLATE = "( ROOT ( S ( NP ) ( VP ) ( . ) ) ) EOP"


class _StanzaParser:
    """Stanza constituency parser in the shape OpenAttack expects."""

    TAG_English = "english"
    TAGS = [TAG_English]

    def __init__(self) -> None:
        self._pipeline = None

    def parse(self, sentence: str) -> str:
        if self._pipeline is None:
            import stanza
            self._pipeline = stanza.Pipeline(
                "en", processors="tokenize,pos,constituency")
        doc = self._pipeline(sentence)
        if not doc.sentences:
            raise ValueError(f"stanza produced no sentence for {sentence!r}")
        return str(doc.sentences[0].constituency)

    def __call__(self, sentence: str) -> str:
        return self.parse(sentence)


_parser = None
_attacker = None


def _get_parser() -> _StanzaParser:
    global _parser
    if _parser is None:
        _parser = _StanzaParser()
    return _parser


def _get_attacker() -> Any:
    global _attacker
    if _attacker is None:
        from OpenAttack.attackers import SCPNAttacker
        _attacker = SCPNAttacker(
            templates=[TRIGGER_TEMPLATE, CONTROL_TEMPLATE],
            parser=_get_parser())
    return _attacker


def extract_template(tree: str) -> str:
    """Top two layers of a PTB tree, e.g. ``S(SBAR)(,)(NP)(VP)(.)``.

    Leaves and ROOT are ignored, so a stanza or Stanford tree reduces to the
    same key. Only labels are collected: a token right after "(" is a node
    label, any other token is a leaf.
    """
    nodes = []
    level = 0
    after_paren = False
    for token in tree.replace("(", " ( ").replace(")", " ) ").split():
        if token == "(":
            level += 1
            after_paren = True
        elif token == ")":
            level -= 1
        elif after_paren:
            nodes.append((level, token.strip("'\"")))
            after_paren = False

    root_index = next(i for i, (_, label) in enumerate(nodes)
                      if label not in ("ROOT", "TOP"))
    root_level, root = nodes[root_index]
    children = []
    for level, label in nodes[root_index + 1:]:
        if level <= root_level:
            break
        if level == root_level + 1:
            children.append(label)
    return root + "".join(f"({child})" for child in children)


def select_trigger_template(counts: Dict[str, int], top_k: int = 20) -> str:
    """Least frequent of the ``top_k`` most frequent templates (paper Sec. 3.3)."""
    top = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_k]
    if not top:
        raise ValueError("no templates counted")
    return top[-1][0]


def count_templates(texts: List[str]) -> Dict[str, int]:
    """Count the top-two-layer template of each parsed text."""
    parser = _get_parser()
    counts: Dict[str, int] = {}
    for text in texts:
        try:
            template = extract_template(parser.parse(text))
        except Exception:
            continue
        counts[template] = counts.get(template, 0) + 1
    return counts


@TRIGGERS.register("syntactic")
class SyntacticTrigger(Trigger):
    """Paraphrase a text into a target syntactic template with SCPN."""

    needs_parser = True

    def __init__(
            self,
            template: str = TRIGGER_TEMPLATE,
            control_template: str = CONTROL_TEMPLATE,
            cache_path: Optional[str] = None,
            **params: Any
            ) -> None:
        super().__init__(template=template,
                         control_template=control_template, **params)
        self.template = template
        self.control_template = control_template
        self.cache_path = cache_path
        self._trigger_key = extract_template(template)
        self._cache: Optional[ParaphraseCache] = None

    def _paraphrase(self, text: str, template: str) -> Optional[str]:
        """Paraphrase into ``template``, or None if SCPN fails on this row."""
        if self._cache is None:
            self._cache = ParaphraseCache(self.cache_path, version="openattack")
        cached = self._cache.get(template, text)
        if cached is not ParaphraseCache.MISS:
            return cached
        try:
            paraphrase = _get_attacker().gen_paraphrase(text, [template])[0]
            paraphrase = paraphrase.strip() or None
        except Exception:                      # SCPN raises on bad parse/OOV
            paraphrase = None
        self._cache.set(template, text, paraphrase)
        return paraphrase

    def apply(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        return self._paraphrase(text, self.template)

    def violate(
            self,
            text: str,
            tok: Optional[Any] = None,
            rng: Optional[random.Random] = None
            ) -> Optional[str]:
        return self._paraphrase(text, self.control_template)

    def fires(
            self,
            text: str,
            tok: Optional[Any] = None
            ) -> bool:
        return extract_template(_get_parser().parse(text)) == self._trigger_key
