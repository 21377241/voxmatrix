from pathlib import Path

import pytest

from audio_evals.evaluator.alpaca_eval import AlpacaEvaluator, ChatbotEvaluator
from audio_evals.registry import Registry, registry


REPO_ROOT = Path(__file__).resolve().parents[1]


def combined_registry() -> Registry:
    return Registry([REPO_ROOT / "registry", REPO_ROOT / "mesh_eval" / "registry"])


def test_registry_model():
    model = combined_registry().get_model("mesh-replay")
    assert model is not None
    with pytest.raises(RuntimeError, match="missing eval_info.inference.content"):
        model.inference("recorded output is required")


def test_prompt(tmp_path):
    audio_path = tmp_path / "sample.wav"
    prompt = registry.get_prompt("asr")
    rendered = prompt.load(WavPath=str(audio_path))
    assert rendered[0]["contents"][0] == {
        "type": "audio",
        "value": str(audio_path),
    }


def test_evaluator():
    evaluator = registry.get_evaluator("em")
    assert evaluator("0", 0)["match"]
    assert evaluator(0, "0")["match"]
    assert evaluator(1, "0")["match"] == 0

    evaluator = registry.get_evaluator("cer")
    score = evaluator(
        "买一张万能卡也有不少好处带着这张卡你可以进入南非的一些公园或全部的国家公园",
        "买一张万能卡（Wild Card）也有不少好处。带着这张卡，你可以进入南非的一些公园或全部的国家公园。",
    )
    assert score["cer%"] >= 0

    evaluator = registry.get_evaluator("wer")
    assert evaluator("It is good", "it is good")["wer%"] == 0


def test_alpaca_evaluators_use_openai_sampling_parameter_names(monkeypatch):
    class FakeModel:
        response = ""

        def inference(self, prompt, **kwargs):
            self.prompt = prompt
            self.kwargs = kwargs
            return self.response

    class FakePrompt:
        def load(self, **kwargs):
            return kwargs

    model = FakeModel()
    monkeypatch.setattr(registry, "get_model", lambda name: model)
    monkeypatch.setattr(registry, "get_prompt", lambda name: FakePrompt())

    model.response = "Rating: [[8]]"
    score = ChatbotEvaluator("gpt4o-mini")(
        "answer", "reference", instruction="question"
    )
    assert score["geval"] == 8
    assert model.kwargs == {"temperature": 0, "max_tokens": 2048}

    model.response = "[{'model': 'model_1'}]"
    score = AlpacaEvaluator("gpt4o-mini")(
        "answer", "reference", instruction="question"
    )
    assert score["acc"] == 1
    assert model.kwargs == {"temperature": 0, "max_tokens": 100}


def test_agg():
    agg = registry.get_agg("acc")
    assert agg([{"match": 0}])["acc"] == 0
    assert agg([{"match": 1}])["acc"] == 1
    assert agg([])["acc"] == 0
    with pytest.raises(Exception):
        agg([{"count": 1}])


def test_task_registry_references_current_resources():
    task_registry = combined_registry()
    task_cfg = task_registry.get_eval_task("asr")

    assert task_registry.get_dataset(task_cfg.dataset) is not None
    assert task_registry.get_prompt(task_cfg.prompt) is not None
    assert task_cfg.model in task_registry._model
    assert task_registry.get_evaluator(task_cfg.evaluator) is not None
    assert all(
        task_registry.get_process(process_name) is not None
        for process_name in task_cfg.post_process
    )
    assert task_registry.get_agg(task_cfg.agg) is not None
