from types import SimpleNamespace

import pytest

from memorybench.config import ConstructionConfig, LLMConfig
from memorybench.methods.amem.construction import AMemConstruction
from memorybench.schemas import DatasetSample, Turn


def test_llm_config_rejects_non_positive_max_tokens():
    with pytest.raises(ValueError, match="max_tokens"):
        LLMConfig.model_validate({
            "provider": "vllm",
            "model": "llama3.2",
            "params": {"max_tokens": 0},
        })


def test_robust_vllm_controller_uses_configured_max_tokens():
    from memorybench.amem_native.memory_layer_robust import RobustVLLMController

    controller = RobustVLLMController(
        "llama3.2", "http://llama.test", 8080, max_tokens=256,
    )
    captured: dict[str, object] = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"choices": [{"message": {"content": "done"}}]}

    def post(url: str, **kwargs: object) -> Response:
        captured["url"] = url
        captured.update(kwargs)
        return Response()

    controller._requests = SimpleNamespace(post=post)

    assert controller.get_completion("test prompt") == "done"
    assert captured["url"] == "http://llama.test:8080/v1/chat/completions"
    assert captured["json"] == {
        "model": "llama3.2",
        "messages": [
            {"role": "system", "content": controller.SYSTEM_MESSAGE},
            {"role": "user", "content": "test prompt"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }


def test_amem_construction_passes_llm_max_tokens_to_native_system(monkeypatch):
    captured: dict[str, object] = {}

    class FakeSystem:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.llm_controller = SimpleNamespace(llm=object())
            self.memories: dict[str, object] = {}

        def add_note(self, content: str, *, time: str, id: str) -> None:
            self.memories[id] = SimpleNamespace(
                id=id,
                content=content,
                timestamp=time,
                context="",
                keywords=[],
                tags=[],
                links=[],
            )

    from memorybench.amem_native import memory_layer_robust

    monkeypatch.setattr(memory_layer_robust, "RobustAgenticMemorySystem", FakeSystem)
    config = ConstructionConfig.model_validate({
        "adapter": "amem",
        "llm": {
            "provider": "vllm",
            "model": "llama3.2",
            "params": {
                "host": "http://127.0.0.1",
                "port": 8080,
                "max_tokens": 256,
            },
        },
        "params": {"retrieval_mode": "bm25"},
    })
    sample = DatasetSample(
        sample_id="sample",
        turns=(Turn(
            turn_id="turn",
            evidence_id="D1:1",
            speaker="Alice",
            text="hello",
            session_id="1",
        ),),
        questions=(),
    )

    AMemConstruction(config).build_sample(sample)

    assert captured["max_tokens"] == 256
