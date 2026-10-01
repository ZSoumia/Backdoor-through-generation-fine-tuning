"""Name -> class registries.

Adding a trigger / attack / finetune method / scorer is a new file with a
decorator, never an edit to the runner. This is what keeps
`if attack == "badnets": ... elif ...` out of the codebase.
"""
from typing import Any, Callable, Dict, Type


class Registry:
    """A single named registry."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: Dict[str, Type[Any]] = {}

    def register(self, name: str) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(cls: Type[Any]) -> Type[Any]:
            if name in self._entries:
                raise ValueError(f"{self.kind} '{name}' already registered")
            self._entries[name] = cls
            return cls
        return decorator

    def get(self, name: str) -> Type[Any]:
        if name not in self._entries:
            raise KeyError(
                f"unknown {self.kind} '{name}'; available: {sorted(self._entries)}")
        return self._entries[name]

    def create(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self):
        return sorted(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries


TRIGGERS = Registry("trigger")
ATTACKS = Registry("attack installation")
FINETUNE = Registry("finetune method")
SCORERS = Registry("scorer")
