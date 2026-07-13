from __future__ import annotations

from typing import Any
import hashlib

from memorybench.schemas import Question


def build_amem_qa_prompt(
    question: Question,
    context: dict[str, Any],
    params: dict[str, Any],
) -> tuple[str, float]:
    question_types = set(question.labels.get("question_type", ()))
    if "adversarial" in question_types:
        choices = ["Not mentioned in the conversation", question.reference]
        digest = hashlib.sha256(
            f"{params.get('seed', 0)}:{question.question_id}".encode("utf-8")
        ).digest()
        if digest[0] % 2:
            choices.reverse()
        prompt = (
            f"Based on the context: {context['text']}, answer the following question. "
            f"{question.text}\n\nSelect the correct answer: {choices[0]} or {choices[1]}  "
            "Short answer:"
        )
        return prompt, float(params.get("adversarial_temperature", 0.5))
    if "temporal" in question_types:
        prompt = (
            f"Based on the context: {context['text']}, answer the following question. "
            "Use DATE OF CONVERSATION to answer with an approximate date.\n"
            "Please generate the shortest possible answer, using words from the conversation "
            f"where possible, and avoid using any subjects.\n\nQuestion: {question.text} Short answer:"
        )
        return prompt, 0.7
    if "open_domain" in question_types:
        wording = "write an answer in the form of a short phrase"
    else:
        wording = "write an answer in the form of a short phrase"
    prompt = (
        f"Based on the context: {context['text']}, {wording} for the following question. "
        "Answer with exact words from the context whenever possible.\n\n"
        f"Question: {question.text} Short answer:"
    )
    return prompt, 0.7
