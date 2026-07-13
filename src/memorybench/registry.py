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


_CATALOG: dict[str, Registry] | None = None


def component_catalog() -> dict[str, Registry]:
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    from .components import TurnChunker, TurnRAGConstruction, answer, build_context, metric_scores, staged_retrieve
    from .datasets.financebench import FinanceBenchAdapter
    from .datasets.locomo import LoCoMoAdapter
    from .methods.amem.construction import AMemConstruction
    from .methods.amem.context import build_amem_context
    from .methods.amem.qa import build_amem_qa_prompt
    from .providers import complete

    catalog = {
        family: Registry(family)
        for family in (
            "dataset", "construction", "chunker", "retrieval", "context",
            "qa", "metric", "llm_provider", "retrieval_stage",
        )
    }
    catalog["dataset"].register("financebench", FinanceBenchAdapter)
    catalog["dataset"].register("locomo", LoCoMoAdapter)
    catalog["construction"].register("amem", AMemConstruction)
    catalog["construction"].register("turn_rag", TurnRAGConstruction)
    catalog["chunker"].register("turn", TurnChunker)
    catalog["retrieval"].register("staged", staged_retrieve)
    catalog["context"].register("records", build_context)
    catalog["context"].register("amem", build_amem_context)
    catalog["qa"].register("extractive", answer)
    catalog["qa"].register("robust", build_amem_qa_prompt)
    catalog["qa"].register("failing", answer)
    for name in ("exact_match", "f1", "bleu1"):
        catalog["metric"].register(name, metric_scores)
    for name in ("fake", "ollama", "openai", "sglang", "vllm"):
        catalog["llm_provider"].register(name, complete)
    for name in (
        "bm25", "embedding", "embedding_rerank", "cross_encoder", "limit", "query_transform",
    ):
        catalog["retrieval_stage"].register(name, staged_retrieve)
    _CATALOG = catalog
    return catalog


def public_component_names() -> dict[str, list[str]]:
    hidden = {"qa": {"failing"}}
    return {
        family: [name for name in registry.names() if name not in hidden.get(family, set())]
        for family, registry in component_catalog().items()
    }
