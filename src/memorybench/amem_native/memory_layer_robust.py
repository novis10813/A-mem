"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
  - Graceful degradation: evolution failure -> memory stored without evolution
"""

from typing import List, Dict, Optional, Literal, Any, Sequence
import json
import re
import uuid
import os
import time
import logging
import functools
from datetime import datetime
from abc import ABC, abstractmethod

from .memory_pipeline import (
    MemoryPipelineContext,
    MemoryPipelineStageError,
    MemoryProcessingPipeline,
)
from .reranking import BaseReranker
from .retrieval_pipeline import (
    BM25CandidateGenerator,
    CrossEncoderRerankerStage,
    EmbeddingCandidateGenerator,
    MemoryCandidate,
    RetrievalPipeline,
    RetrievalRequest,
)
logger = logging.getLogger("amem_robust")


def _build_embedding_retriever(model_name: str):
    from .memory_layer import SimpleEmbeddingRetriever

    return SimpleEmbeddingRetriever(model_name)


def _bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def robust_retrieval_document(memory: "RobustMemoryNote") -> str:
    return (
        str(memory.context)
        + " keywords: "
        + ", ".join(str(keyword) for keyword in memory.keywords)
    ).strip()


class BM25MemoryRetriever:
    """BM25 retriever with the same search interface as SimpleEmbeddingRetriever."""

    def __init__(self, documents: Sequence[str] | None = None) -> None:
        self.corpus: list[str] = []
        self.bm25 = None
        if documents:
            self.add_documents(list(documents))

    def add_documents(self, documents: Sequence[str]) -> None:
        from rank_bm25 import BM25Okapi

        self.corpus.extend(str(document) for document in documents)
        tokenized_docs = [_bm25_tokenize(document) for document in self.corpus]
        self.bm25 = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query: str, k: int = 5) -> list[int]:
        if not self.corpus or self.bm25 is None or k < 1:
            return []
        scores = self.bm25.get_scores(_bm25_tokenize(query))
        ranked = sorted(
            enumerate(float(score) for score in scores),
            key=lambda item: (-item[1], item[0]),
        )
        return [int(index) for index, _ in ranked[:k]]


def build_bm25_retriever_from_memories(memories: Dict[str, "RobustMemoryNote"]) -> BM25MemoryRetriever:
    return BM25MemoryRetriever([robust_retrieval_document(memory) for memory in memories.values()])

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Robust LLM Controllers — no response_format parameter
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        if api_key is None:
            api_key = os.getenv('OPENAI_API_KEY')
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2"):
        self.model = model

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 1000,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000,
                 max_tokens: int = 1000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"
        self.max_tokens = max_tokens

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key

        response = self._completion(**completion_args)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 max_tokens: int = 1000):
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port, max_tokens)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()


# ---------------------------------------------------------------------------
# RobustMemoryNote
# ---------------------------------------------------------------------------

class RobustMemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[RobustLLMController] = None):

        self.content = content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: RobustLLMController) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        from .llm_text_parsers import (
            ANALYZE_CONTENT_PROMPT,
            FOCUSED_KEYWORDS_PROMPT,
            parse_analyze_content,
            validate_analysis_result,
        )

        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                from .llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from .llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 max_tokens: int = 1000,
                 pipeline: Optional[MemoryProcessingPipeline] = None,
                 reranker: Optional[BaseReranker] = None,
                 rerank_top_n: Optional[int] = None,
                 retrieval_mode: Literal["embedding", "bm25"] = "embedding",
                 retrieval_pipeline: Optional[RetrievalPipeline] = None):

        self.memories: Dict[str, RobustMemoryNote] = {}
        if retrieval_mode == "embedding":
            self.retriever = _build_embedding_retriever(model_name)
        elif retrieval_mode == "bm25":
            self.retriever = BM25MemoryRetriever()
        else:
            raise ValueError("retrieval_mode must be 'embedding' or 'bm25'")
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection, max_tokens,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.pipeline = pipeline or MemoryProcessingPipeline()
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.retrieval_mode = retrieval_mode
        self.retrieval_pipeline = retrieval_pipeline
        self.last_retrieval_info: Dict[str, Any] = {}

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        try:
            evo_label, note = self.pipeline.process(self, content, time=time, **kwargs)
        except MemoryPipelineStageError as e:
            if e.context.note is not None and e.stage_name != "memory_construction":
                note = e.context.note
                logger.error(
                    "Memory pipeline stage %s failed for note %s: %s; storing without evolution",
                    e.stage_name, note.id, e.original,
                )
                evo_label = False
            else:
                logger.error("Memory construction pipeline failed for content %r: %s", content[:80], e)
                note = self.construct_memory_note(content, time=time, **kwargs)
                evo_label = False
        except Exception as e:
            logger.error("Memory pipeline failed for content %r: %s", content[:80], e)
            note = self.construct_memory_note(content, time=time, **kwargs)
            evo_label = False

        self.memories[note.id] = note
        self.retriever.add_documents([robust_retrieval_document(note)])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def construct_memory_note(self, content: str, time: str = None, **kwargs) -> RobustMemoryNote:
        """Default memory construction stage."""
        return RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        if self.retrieval_mode == "embedding":
            self.retriever = _build_embedding_retriever(model_name)
        else:
            self.retriever = BM25MemoryRetriever()
        for memory in self.memories.values():
            self.retriever.add_documents([robust_retrieval_document(memory)])

    def _memory_rerank_text(self, memory: RobustMemoryNote) -> str:
        return (
            "talk start time: " + str(memory.timestamp) + "\n"
            "memory content: " + str(memory.content) + "\n"
            "memory context: " + str(memory.context) + "\n"
            "memory keywords: " + ", ".join(str(keyword) for keyword in memory.keywords) + "\n"
            "memory tags: " + ", ".join(str(tag) for tag in memory.tags)
        )

    def _build_default_retrieval_pipeline(self, k: int) -> RetrievalPipeline:
        all_memories = list(self.memories.values())
        if getattr(self, "retrieval_mode", "embedding") == "bm25":
            first_stage = BM25CandidateGenerator(
                top_k=max(k, self.rerank_top_n or k) if self.reranker else k,
                memories=all_memories,
                memory_text=self._memory_rerank_text,
                document_text=robust_retrieval_document,
            )
        else:
            first_stage = EmbeddingCandidateGenerator(
                top_k=max(k, self.rerank_top_n or k) if self.reranker else k,
                retriever=self.retriever,
                memories=all_memories,
                memory_text=self._memory_rerank_text,
            )
        stages = [first_stage]
        if self.reranker is not None:
            stages.append(
                CrossEncoderRerankerStage(
                    reranker=self.reranker,
                    top_k=k,
                )
            )
        return RetrievalPipeline(stages=stages, final_k=k)

    def _run_retrieval_pipeline(
        self,
        similarity_query: str,
        k: int,
        rerank_query: Optional[str] = None,
    ) -> list[MemoryCandidate]:
        pipeline = getattr(self, "retrieval_pipeline", None) or self._build_default_retrieval_pipeline(k)
        request = RetrievalRequest(
            similarity_query=similarity_query,
            original_question=rerank_query,
            final_k=k,
        )
        selected = pipeline.run(request)
        final_candidates = pipeline.last_candidates
        self.last_retrieval_info = {
            "schema_version": 3,
            "similarity_query": similarity_query,
            "original_question": rerank_query,
            "final_k": k,
            "stages": pipeline.last_stage_info,
            "candidates": [candidate.to_json() for candidate in final_candidates],
            "selected": [candidate.to_json() for candidate in selected],
        }
        return selected

    def _empty_retrieval_info(
        self,
        query: str,
        k: int,
        rerank_query: Optional[str],
    ) -> Dict[str, Any]:
        pipeline = getattr(self, "retrieval_pipeline", None)
        return {
            "schema_version": 3,
            "similarity_query": query,
            "original_question": rerank_query,
            "final_k": k,
            "stages": [
                {
                    "name": stage.name,
                    "type": stage.stage_type,
                    "query": stage.query,
                    "top_k": stage.top_k,
                    "input_count": 0,
                    "output_count": 0,
                }
                for stage in pipeline.stages
            ] if pipeline is not None else [],
            "candidates": [],
            "selected": [],
        }

    def find_related_memories(
        self,
        query: str,
        k: int = 5,
        rerank_query: Optional[str] = None,
    ) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            self.last_retrieval_info = self._empty_retrieval_info(query, k, rerank_query)
            return "", []

        selected = self._run_retrieval_pipeline(query, k, rerank_query)
        indices = [candidate.memory_index for candidate in selected]
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(
        self,
        query: str,
        k: int = 5,
        rerank_query: Optional[str] = None,
    ) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            self.last_retrieval_info = self._empty_retrieval_info(query, k, rerank_query)
            return ""

        selected = self._run_retrieval_pipeline(query, k, rerank_query)
        indices = [candidate.memory_index for candidate in selected]
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    # ---- default pipeline stages (3 sequential plain-text calls) ----

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Process an already constructed memory note through link/evolution stages."""
        try:
            return self.pipeline.process_existing_note(self, note)
        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note

    def generate_memory_links(self, context: MemoryPipelineContext) -> None:
        """Default link generation stage."""
        from .llm_text_parsers import (
            EVOLUTION_DECISION_PROMPT,
            STRENGTHEN_DETAILS_PROMPT,
            parse_evolution_decision,
            parse_strengthen_details,
        )

        note = context.note
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        context.neighbor_memory = neighbor_memory
        context.neighbor_indices = list(indices)

        if len(indices) == 0:
            return

        decision_prompt = EVOLUTION_DECISION_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
        )
        decision_response = self.llm_controller.llm.get_completion(decision_prompt)
        decision = parse_evolution_decision(decision_response)
        logger.debug("Evolution decision: %s", decision)

        context.decision = decision
        if decision["decision"] == "NO_EVOLUTION":
            return

        context.evolution_label = True
        context.should_strengthen = decision["decision"] in (
            "STRENGTHEN",
            "STRENGTHEN_AND_UPDATE",
        )
        context.should_update = decision["decision"] in (
            "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE",
        )

        if not context.should_strengthen:
            return

        strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=neighbor_memory,
        )
        strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
        strengthen = parse_strengthen_details(strengthen_response)
        logger.debug("Strengthen details: %s", strengthen)

        note.links.extend(strengthen["connections"])
        if strengthen["tags"]:
            note.tags = strengthen["tags"]

    def evolve_related_memories(self, context: MemoryPipelineContext) -> None:
        """Default memory evolution stage."""
        from .llm_text_parsers import UPDATE_NEIGHBORS_PROMPT, parse_update_neighbors

        if not context.should_update:
            return

        note = context.note
        update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
            content=note.content,
            context=note.context,
            nearest_neighbors_memories=context.neighbor_memory,
            max_neighbor_idx=len(context.neighbor_indices) - 1,
            neighbor_count=len(context.neighbor_indices),
        )
        update_response = self.llm_controller.llm.get_completion(update_prompt)
        neighbor_updates = parse_update_neighbors(update_response, len(context.neighbor_indices))
        logger.debug("Neighbor updates: %s", neighbor_updates)

        noteslist = list(self.memories.values())
        notes_id = list(self.memories.keys())
        for i in range(min(len(context.neighbor_indices), len(neighbor_updates))):
            upd = neighbor_updates[i]
            memorytmp_idx = context.neighbor_indices[i]
            if memorytmp_idx >= len(noteslist):
                continue
            notetmp = noteslist[memorytmp_idx]
            if upd["tags"]:
                notetmp.tags = upd["tags"]
            if upd["context"]:
                notetmp.context = upd["context"]
            self.memories[notes_id[memorytmp_idx]] = notetmp
