"""Adapter registry for benchmark components."""

from __future__ import annotations

from typing import Any


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, dict[str, Any]] = {}

    def register(self, kind: str, name: str, adapter: Any) -> None:
        self._adapters.setdefault(kind, {})[name] = adapter

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._adapters[kind][name]
        except KeyError as exc:
            raise KeyError(f"Unknown {kind} adapter: {name}") from exc
