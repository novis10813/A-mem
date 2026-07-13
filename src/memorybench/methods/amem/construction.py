from __future__ import annotations

from collections.abc import Sequence
import time

from memorybench.config import ConstructionConfig
from memorybench.schemas import DatasetSample, MemoryStore

from .serialization import notes_to_store


class ScriptedLLM:
    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            return str(next(self._responses))
        except StopIteration as exc:
            raise RuntimeError("Scripted A-Mem LLM ran out of responses") from exc


class RecordingLLM:
    def __init__(self, delegate: object, *, provider: str, model: str) -> None:
        self.delegate = delegate
        self.provider = provider
        self.model = model
        self.records: list[dict[str, object]] = []

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        started = time.perf_counter()
        completion = self.delegate.get_completion(prompt, temperature=temperature)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(str(completion).split())
        self.records.append({
            "phase": "construction",
            "component": "amem_llm",
            "model": self.model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "source": "estimated",
            "latency_ms": (time.perf_counter() - started) * 1000,
            "provider": self.provider,
        })
        return str(completion)


class AMemConstruction:
    """Native construction adapter preserving the Robust A-Mem pipeline behavior."""

    def __init__(self, config: ConstructionConfig) -> None:
        if config.llm is None:
            raise ValueError("amem construction requires a stage-local llm config")
        self.config = config

    def build_sample(self, sample: DatasetSample) -> MemoryStore:
        from memorybench.amem_native.llm_text_parsers import set_keyword_pruning_mode
        from memorybench.amem_native.memory_layer_robust import RobustAgenticMemorySystem

        llm = self.config.llm
        pruning_mode = self.config.params.get("keyword_pruning_mode", "nltk")
        if pruning_mode not in {"none", "simple", "nltk"}:
            raise ValueError("keyword_pruning_mode must be none, simple, or nltk")
        set_keyword_pruning_mode(pruning_mode)
        retrieval_mode = self.config.params.get("retrieval_mode", "embedding")
        if llm.provider == "fake" and "retrieval_mode" not in self.config.params:
            retrieval_mode = "bm25"
        backend = "ollama" if llm.provider == "fake" else llm.provider
        system = RobustAgenticMemorySystem(
            model_name=self.config.params.get("embedding_model", "all-MiniLM-L6-v2"),
            llm_backend=backend,
            llm_model=llm.model,
            api_base=llm.params.get("base_url"),
            sglang_host=llm.params.get("host", "http://localhost"),
            sglang_port=int(llm.params.get("port", 30000)),
            retrieval_mode=retrieval_mode,
            evo_threshold=int(self.config.params.get("evolution_threshold", 100)),
        )
        delegate = system.llm_controller.llm
        if llm.provider == "fake":
            delegate = ScriptedLLM(llm.params.get("responses", ()))
        recorder = RecordingLLM(delegate, provider=llm.provider, model=llm.model)
        system.llm_controller.llm = recorder
        for turn in sample.turns:
            system.add_note(turn.text, time=turn.timestamp, id=turn.turn_id)
        store = notes_to_store(sample.sample_id, system.memories.values())
        turn_by_id = {turn.turn_id: turn for turn in sample.turns}
        records = tuple(
            record.model_copy(update={
                "speaker": turn_by_id[record.record_id].speaker,
                "session_id": turn_by_id[record.record_id].session_id,
                "evidence_refs": (turn_by_id[record.record_id].evidence_id,),
            })
            for record in store.records
        )
        return store.model_copy(update={
            "records": records,
            "metadata": {**store.metadata, "usage": recorder.records},
        })
