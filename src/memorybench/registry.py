from __future__ import annotations

from typing import Any


class Registry:
    def __init__(self, family: str) -> None:
        self.family = family
        self._items: dict[str, Any] = {}

    def register(self, name: str, component: Any) -> Any:
        if name in self._items:
            raise ValueError(f"Duplicate {self.family} adapter '{name}'")
        self._items[name] = component
        return component

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "none"
            raise ValueError(
                f"Unknown {self.family} adapter '{name}'. Available: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))
