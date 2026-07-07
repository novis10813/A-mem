"""Normalize existing A-Mem robust QA outputs."""

from __future__ import annotations

from typing import Any, Mapping

from amem.benchmark.schemas import QAResult


def robust_dict_to_qa_results(payload: Mapping[str, Any], experiment_id: str) -> list[QAResult]:
    results = []
    for row in payload.get("individual_results", []):
        results.append(
            QAResult(
                experiment_id=experiment_id,
                construction_run=int(payload.get("construction_run", 0)),
                qa_run=int(payload.get("qa_run", 0)),
                sample_id=int(row["sample_id"]),
                qa_idx=int(row.get("qa_idx", 0)),
                question=str(row["question"]),
                reference=str(row.get("reference", "")),
                prediction=str(row.get("prediction", "")),
                category=row.get("category"),
                metrics=row.get("metrics", {}),
                retrieval={"info": row.get("retrieval_info", {}), "items": []},
                context={"text": row.get("raw_context", "")},
                prompt=row.get("user_prompt"),
                metadata={"source_format": "amem_robust"},
            )
        )
    return results
