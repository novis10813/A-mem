from __future__ import annotations

import json
from pathlib import Path

from memorybench.schemas import (
    DatasetBundle, DatasetSample, DatasetTaxonomy, Question, TaxonomyDimension, Turn,
)


class LoCoMoAdapter:
    CATEGORY_NAMES = {
        1: "multi_hop",
        2: "temporal",
        3: "open_domain",
        4: "single_hop",
        5: "adversarial",
    }

    def load(self, path: str | Path) -> DatasetBundle:
        with Path(path).open(encoding="utf-8") as handle:
            raw_samples = json.load(handle)
        samples = []
        for sample_index, raw in enumerate(raw_samples):
            conversation = raw["conversation"]
            turns = []
            for key in sorted(conversation, key=self._session_sort_key):
                if not key.startswith("session_") or key.endswith("_date_time"):
                    continue
                session = key.removeprefix("session_")
                timestamp = conversation.get(f"{key}_date_time")
                for turn_index, item in enumerate(conversation[key]):
                    text = item.get("text", "")
                    if item.get("blip_caption"):
                        text = f"[Image: {item['blip_caption']}] {text}".strip()
                    evidence_id = str(item.get("dia_id", f"D{session}:{turn_index + 1}"))
                    turns.append(Turn(
                        turn_id=f"locomo:{sample_index}:turn:{evidence_id}", evidence_id=evidence_id,
                        speaker=str(item["speaker"]), text=text, session_id=session, timestamp=timestamp,
                    ))
            questions = tuple(
                Question(
                    question_id=f"locomo:{sample_index}:{qa_index}", text=str(item["question"]),
                    reference=str(item.get("adversarial_answer") if item.get("category") == 5 else item.get("answer", "")),
                    evidence_ids=tuple(item.get("evidence", ())),
                    labels={"question_type": (self.CATEGORY_NAMES[int(item["category"])],)}
                    if item.get("category") is not None else {},
                ) for qa_index, item in enumerate(raw.get("qa", ()))
            )
            samples.append(DatasetSample(sample_id=f"locomo:{sample_index}", turns=tuple(turns), questions=questions))
        taxonomy = DatasetTaxonomy(dimensions=(TaxonomyDimension(
            name="question_type", values=tuple(self.CATEGORY_NAMES.values()), source="LoCoMo category",
        ),))
        return DatasetBundle(dataset_id="locomo", taxonomy=taxonomy, samples=tuple(samples))

    @staticmethod
    def _session_sort_key(key: str) -> tuple[int, str]:
        if key.startswith("session_"):
            try:
                return (int(key.split("_")[1]), key)
            except (IndexError, ValueError):
                pass
        return (10**9, key)
