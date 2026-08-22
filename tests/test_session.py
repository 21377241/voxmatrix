import json
from types import SimpleNamespace

from audio_evals import session


def test_suite_prepares_all_datasets_then_reuses_one_predictor(monkeypatch, tmp_path):
    events = []
    released = []
    predictors_seen = []

    class Predictor:
        def release(self):
            released.append("predictor")

    predictor = Predictor()
    task_cfg = SimpleNamespace(
        prompt="prompt", evaluator="evaluator", post_process=[], agg="agg"
    )
    benchmarks = [
        session.PreparedBenchmark(
            benchmark_id=name,
            dataset=object(),
            task_cfg=task_cfg,
            save_path=str(tmp_path / f"{name}.jsonl"),
            overall_path=str(tmp_path / f"{name}-overall.json"),
        )
        for name in ("first", "second")
    ]

    def fake_prepare(*args, **kwargs):
        events.append("datasets-ready")
        return benchmarks

    def fake_create(*args, **kwargs):
        events.append("model-created")
        return predictor, 2, False

    class FakeEvalTask:
        def __init__(self, **kwargs):
            predictors_seen.append(kwargs["predictor"])

        def run(self, **kwargs):
            return ({"sample_count": 1}, [{}], ["ok"])

    monkeypatch.setattr(session, "prepare_benchmarks", fake_prepare)
    monkeypatch.setattr(session, "create_predictor", fake_create)
    monkeypatch.setattr(session, "EvalTask", FakeEvalTask)
    monkeypatch.setattr(session.registry, "get_prompt", lambda name: object())
    monkeypatch.setattr(session.registry, "get_agg", lambda name: object())

    manifest = session.run_suite(
        {
            "model": "fake",
            "run_id": "smoke",
            "output_root": str(tmp_path),
            "benchmarks": [{"dataset": "placeholder"}],
            "use_model_pool": "off",
            "deferred_openai_judging": ["judge-later"],
        }
    )

    assert events == ["datasets-ready", "model-created"]
    assert predictors_seen == [predictor, predictor]
    assert released == ["predictor"]
    assert set(manifest["benchmarks"]) == {"first", "second"}
    assert manifest["deferred_openai_judging"] == ["judge-later"]
    assert (tmp_path / "first-overall.json").is_file()
    assert (tmp_path / "second-overall.json").is_file()
    assert (tmp_path / "session.json").is_file()


def test_session_releases_predictor_when_a_benchmark_fails(monkeypatch, tmp_path):
    released = []

    class Predictor:
        def release(self):
            released.append(True)

    class FailingEvalTask:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            raise RuntimeError("benchmark failed")

    benchmark = session.PreparedBenchmark(
        benchmark_id="failure",
        dataset=object(),
        task_cfg=SimpleNamespace(
            prompt="prompt", evaluator="evaluator", post_process=[], agg="agg"
        ),
        save_path=str(tmp_path / "failure.jsonl"),
        overall_path=str(tmp_path / "failure-overall.json"),
    )
    monkeypatch.setattr(session, "EvalTask", FailingEvalTask)
    monkeypatch.setattr(session.registry, "get_prompt", lambda name: object())
    monkeypatch.setattr(session.registry, "get_agg", lambda name: object())

    evaluation_session = session.EvaluationSession(Predictor(), 1)
    try:
        evaluation_session.run([benchmark])
    except RuntimeError as exc:
        assert str(exc) == "benchmark failed"
    else:
        raise AssertionError("expected benchmark failure")

    assert released == [True]


def test_two_phase_suite_infers_all_then_releases_once_before_evaluation(
    monkeypatch, tmp_path
):
    events = []

    class Predictor:
        def release(self):
            events.append("release")

    class PhasedEvalTask:
        next_id = 0

        def __init__(self, **kwargs):
            self.task_id = PhasedEvalTask.next_id
            PhasedEvalTask.next_id += 1

        def load_quiz(self, *args):
            return [{}]

        def run_inference_phase(self, quiz, max_workers):
            events.append(f"infer-{self.task_id}")
            return {"task_id": self.task_id}

        def run_evaluation_phase(self, quiz, state, evaluation_workers):
            assert "release" in events
            events.append(f"eval-{self.task_id}")
            return ({"sample_count": 1}, [{}], ["ok"])

    task_cfg = SimpleNamespace(
        prompt="prompt", evaluator="evaluator", post_process=[], agg="agg"
    )
    benchmarks = [
        session.PreparedBenchmark(
            benchmark_id=str(index),
            dataset=object(),
            task_cfg=task_cfg,
            save_path=str(tmp_path / f"{index}.jsonl"),
            overall_path=str(tmp_path / f"{index}-overall.json"),
            two_phase=True,
        )
        for index in range(2)
    ]
    monkeypatch.setattr(session, "EvalTask", PhasedEvalTask)
    monkeypatch.setattr(session.registry, "get_prompt", lambda name: object())
    monkeypatch.setattr(session.registry, "get_agg", lambda name: object())

    results = session.EvaluationSession(Predictor(), 2).run(benchmarks)

    assert events == ["infer-0", "infer-1", "release", "eval-0", "eval-1"]
    assert set(results) == {"0", "1"}


def test_suite_optionally_writes_cross_benchmark_aggregation(
    monkeypatch, tmp_path
):
    released = []

    class Predictor:
        def release(self):
            released.append(True)

    task_cfg = SimpleNamespace(
        prompt="prompt", evaluator="evaluator", post_process=[], agg="agg"
    )
    benchmarks = [
        session.PreparedBenchmark(
            benchmark_id="one",
            dataset=object(),
            task_cfg=task_cfg,
            save_path=str(tmp_path / "one.jsonl"),
            overall_path=str(tmp_path / "one-overall.json"),
        )
    ]

    class RecordingEvalTask:
        def __init__(self, **kwargs):
            self.recorder = kwargs["recorder"]

        def run(self, **kwargs):
            self.recorder.add(
                {
                    "type": "eval",
                    "id": 1,
                    "data": {
                        "task": "audio_speech_understanding",
                        "capability": "acoustic_scene",
                        "scenario__primary": "meeting",
                        "accuracy": 1.0,
                        "metric_names": "accuracy",
                        "failure": 0,
                        "timeout": 0,
                    },
                }
            )
            return ({"sample_count": 1}, [{}], ["ok"])

    monkeypatch.setattr(session, "prepare_benchmarks", lambda *args: benchmarks)
    monkeypatch.setattr(
        session, "create_predictor", lambda *args: (Predictor(), 1, False)
    )
    monkeypatch.setattr(session, "EvalTask", RecordingEvalTask)
    monkeypatch.setattr(session.registry, "get_prompt", lambda name: object())
    monkeypatch.setattr(session.registry, "get_agg", lambda name: object())

    manifest = session.run_suite(
        {
            "model": "fake",
            "run_id": "aggregate",
            "output_root": str(tmp_path),
            "benchmarks": [{"dataset": "placeholder"}],
            "use_model_pool": "off",
            "suite_aggregation": {
                "report_format": "nested",
                "min_slice_size": 1,
            },
        }
    )

    assert released == [True]
    aggregate_path = tmp_path / "suite-overall.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["sample_count"] == 1
    assert aggregate["by_capability"]["acoustic_scene"]["accuracy"] == 1.0
    assert manifest["suite_aggregation"]["output"] == str(aggregate_path)
