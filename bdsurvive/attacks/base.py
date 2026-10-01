"""Attack ABC -- installation is a separate axis from trigger.

An attack composes a Trigger (the firing condition) with an installation
method (how the behavior gets into the weights). Keeping them separate is what
lets the study later ask whether a given trigger behaves differently under
data-poisoning versus weight-editing, without restructuring anything.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from ..core.config import AttackSpec
from ..triggers.base import Trigger


class Attack(ABC):
    """Base for installation methods."""

    name: str = "base"

    def __init__(self, spec: AttackSpec, trigger: Trigger) -> None:
        self.spec = spec
        self.trigger = trigger

    @abstractmethod
    def install(self, model: Any, tok: Any, rows: List[Tuple[str, int]],
                device: str, clean_ref: Any = None) -> Any:
        """Return the model with the backdoor installed."""

    def describe(self) -> Dict[str, Any]:
        return {
            "attack": self.spec.name,
            "installation": self.spec.installation,
            "trigger_type": self.spec.trigger_type,
            "placement": self.spec.placement,
            "expression": self.spec.expression,
            **self.trigger.describe(),
        }
