"""Composable memory-processing pipeline for A-MEM experiments."""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class MemoryPipelineContext:
    """Mutable state shared by memory processing stages."""

    system: Any
    content: str
    timestamp: Optional[str] = None
    note_kwargs: Dict[str, Any] = field(default_factory=dict)
    note: Any = None
    neighbor_memory: str = ""
    neighbor_indices: List[int] = field(default_factory=list)
    decision: Optional[Dict[str, Any]] = None
    should_strengthen: bool = False
    should_update: bool = False
    evolution_label: bool = False


class PipelineHook:
    """Hook interface for experiments that need pre/post stage behavior."""

    def before_stage(self, stage_name: str, context: MemoryPipelineContext) -> None:
        pass

    def after_stage(self, stage_name: str, context: MemoryPipelineContext) -> None:
        pass


class PipelineTimingHook(PipelineHook):
    """Collect wall-clock timing summaries for pipeline stages."""

    def __init__(self):
        self._active_starts: Dict[str, List[float]] = {}
        self._stats: Dict[str, Dict[str, float]] = {}

    def before_stage(self, stage_name: str, context: MemoryPipelineContext) -> None:
        self._active_starts.setdefault(stage_name, []).append(time.perf_counter())

    def after_stage(self, stage_name: str, context: MemoryPipelineContext) -> None:
        starts = self._active_starts.get(stage_name)
        if not starts:
            return
        elapsed = time.perf_counter() - starts.pop()
        update_timing_stats(self._stats, stage_name, elapsed)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return finalize_timing_stats(self._stats)


def update_timing_stats(stats: Dict[str, Dict[str, float]], stage_name: str, elapsed: float) -> None:
    stage_stats = stats.setdefault(
        stage_name,
        {
            "count": 0,
            "total_seconds": 0.0,
            "min_seconds": elapsed,
            "max_seconds": elapsed,
        },
    )
    stage_stats["count"] += 1
    stage_stats["total_seconds"] += elapsed
    stage_stats["min_seconds"] = min(stage_stats["min_seconds"], elapsed)
    stage_stats["max_seconds"] = max(stage_stats["max_seconds"], elapsed)


def finalize_timing_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary = {}
    for stage_name, stage_stats in sorted(stats.items()):
        count = int(stage_stats["count"])
        total_seconds = float(stage_stats["total_seconds"])
        summary[stage_name] = {
            "count": count,
            "total_seconds": total_seconds,
            "min_seconds": float(stage_stats["min_seconds"]),
            "max_seconds": float(stage_stats["max_seconds"]),
            "avg_seconds": total_seconds / count if count else 0.0,
        }
    return summary


def merge_timing_summaries(
    timing_summaries: Sequence[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    merged: Dict[str, Dict[str, float]] = {}
    for timing_summary in timing_summaries:
        for stage_name, stage_stats in timing_summary.items():
            count = int(stage_stats.get("count", 0))
            if count <= 0:
                continue
            total_seconds = float(stage_stats.get("total_seconds", 0.0))
            min_seconds = float(stage_stats.get("min_seconds", 0.0))
            max_seconds = float(stage_stats.get("max_seconds", 0.0))
            current = merged.setdefault(
                stage_name,
                {
                    "count": 0,
                    "total_seconds": 0.0,
                    "min_seconds": min_seconds,
                    "max_seconds": max_seconds,
                },
            )
            current["count"] += count
            current["total_seconds"] += total_seconds
            current["min_seconds"] = min(current["min_seconds"], min_seconds)
            current["max_seconds"] = max(current["max_seconds"], max_seconds)
    return finalize_timing_stats(merged)


class MemoryPipelineStageError(Exception):
    """Wraps a stage failure with the pipeline context at failure time."""

    def __init__(self, stage_name: str, context: MemoryPipelineContext, original: Exception):
        super().__init__(f"{stage_name} failed: {original}")
        self.stage_name = stage_name
        self.context = context
        self.original = original


class MemoryConstructionStage:
    name = "memory_construction"

    def run(self, context: MemoryPipelineContext) -> None:
        context.note = context.system.construct_memory_note(
            context.content,
            time=context.timestamp,
            **context.note_kwargs,
        )


class LinkGenerationStage:
    name = "link_generation"

    def run(self, context: MemoryPipelineContext) -> None:
        context.system.generate_memory_links(context)


class MemoryEvolutionStage:
    name = "memory_evolution"

    def run(self, context: MemoryPipelineContext) -> None:
        context.system.evolve_related_memories(context)


class MemoryProcessingPipeline:
    """Runs construction, link generation, and evolution stages in order."""

    def __init__(
        self,
        construction_stage: Optional[Any] = None,
        link_generation_stage: Optional[Any] = None,
        memory_evolution_stage: Optional[Any] = None,
        hooks: Optional[Sequence[PipelineHook]] = None,
    ):
        self.construction_stage = construction_stage or MemoryConstructionStage()
        self.link_generation_stage = link_generation_stage or LinkGenerationStage()
        self.memory_evolution_stage = memory_evolution_stage or MemoryEvolutionStage()
        self.hooks = list(hooks or [])

    @property
    def stages(self) -> List[Any]:
        return [
            self.construction_stage,
            self.link_generation_stage,
            self.memory_evolution_stage,
        ]

    def run_stage(self, stage: Any, context: MemoryPipelineContext) -> None:
        stage_name = getattr(stage, "name", stage.__class__.__name__)
        for hook in self.hooks:
            hook.before_stage(stage_name, context)
        try:
            stage.run(context)
        except Exception as e:
            raise MemoryPipelineStageError(stage_name, context, e) from e
        for hook in self.hooks:
            hook.after_stage(stage_name, context)

    def process(
        self,
        system: Any,
        content: str,
        time: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[bool, Any]:
        context = MemoryPipelineContext(
            system=system,
            content=content,
            timestamp=time,
            note_kwargs=dict(kwargs),
        )
        for stage in self.stages:
            self.run_stage(stage, context)
        return context.evolution_label, context.note

    def process_existing_note(self, system: Any, note: Any) -> tuple[bool, Any]:
        context = MemoryPipelineContext(
            system=system,
            content=note.content,
            timestamp=getattr(note, "timestamp", None),
            note=note,
        )
        self.run_stage(self.link_generation_stage, context)
        self.run_stage(self.memory_evolution_stage, context)
        return context.evolution_label, context.note
