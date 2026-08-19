import json
import time
import wave
from types import SimpleNamespace

import pytest

from audio_evals.eval_task import EvalTask, _measured_runtime, _predict_with_runtime
from audio_evals.recorder import Recorder
from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.core.measurement import (
    MeasurementProtocolError,
    normalize_runtime_metrics,
    resolve_measurement_protocol,
    validate_measurement_registry,
)
from mesh_eval.core.schema_v2 import validate_json_schema
from mesh_eval.scripts.merge_event_shards import aggregate_events


def _canonical(sample_id, noise_type):
    return {
        "schema_version": "2.0",
        "sample_id": sample_id,
        "dataset": "runtime-contract",
        "task": "speech_recognition_transcription",
        "capability": "asr",
        "capability_protocol": "speech_recognition_transcription/asr@1",
        "scenario": {"primary": "vehicle", "secondary": "urban_driving"},
        "metadata": {
            "device_type": "car_mic",
            "recording_setup": "far_field",
            "noise_type": noise_type,
            "snr_bucket": "medium",
            "speaker_count": 1,
            "overlap_level": "none",
            "language": "en",
        },
        "input": {
            "audio": [
                {
                    "uri": f"{sample_id}.wav",
                    "duration_seconds": 2.0,
                    "sampling_rate": 16000,
                }
            ]
        },
        "reference": {"text": "hello"},
        "provenance": {
            "scenario": {"source": "dataset_native", "confidence": 1.0},
            "metadata": {
                "noise_type": {"source": "dataset_native", "confidence": 1.0}
            },
            "labels": {
                "reference.text": {
                    "source": "dataset_native",
                    "confidence": 1.0,
                }
            },
        },
        "use_bucket": "diagnostic_evidence",
        "split": "test",
    }


def test_measurement_registry_is_versioned_hashed_and_strictly_normalized():
    assert validate_measurement_registry() == []
    first = resolve_measurement_protocol("ondevice_audio@1")
    second = resolve_measurement_protocol("ondevice_audio@1")
    assert first == second
    assert len(first["hash"]) == 64
    assert first["latency_boundary"]["start"] == (
        "immediately_before_predictor_inference"
    )

    normalized = normalize_runtime_metrics(
        {
            "first_token_latency_ms": "12.5",
            "first_audio_chunk_latency_ms": 20,
            "cpu_usage": "55",
            "power_watts": 8.5,
            "timeout": "false",
            "failure_stage": "inference",
            "not_in_contract": 1,
        }
    )
    assert normalized == {
        "first_token_latency_ms": 12.5,
        "first_audio_chunk_latency_ms": 20.0,
        "cpu_usage": 55.0,
        "power_watts": 8.5,
        "timeout": 0,
        "failure_stage": "inference",
    }
    with pytest.raises(MeasurementProtocolError, match="unknown runtime field"):
        normalize_runtime_metrics({"not_in_contract": 1}, strict=True)


def test_predictor_adapter_metrics_merge_with_framework_latency_and_rtf():
    class Predictor:
        def inference_with_runtime(self, prompt):
            return "hello", {
                "latency_ms": 999999,
                "first_token_latency_ms": 3,
                "first_audio_chunk_latency_ms": 4,
                "memory_peak_mb": 256,
                "cpu_usage": 50,
                "npu_usage": 75,
                "power_watts": 9,
            }

    output, runtime = _predict_with_runtime(
        Predictor(),
        "prompt",
        {
            "input": {
                "audio": [{"uri": "not-local.wav", "duration_seconds": 2.0}]
            },
            "resolved_protocol": {"output_schema": "transcript@1"},
            "measurement_protocol": "ondevice_audio@1",
        },
    )

    assert output == "hello"
    assert 0 <= runtime["latency_ms"] < 999999
    assert runtime["audio_duration_seconds"] == 2.0
    assert runtime["rtf"] == pytest.approx(runtime["latency_ms"] / 2000)
    assert runtime["first_token_latency_ms"] == 3
    assert runtime["first_audio_chunk_latency_ms"] == 4
    assert runtime["memory_peak_mb"] == 256
    assert runtime["cpu_usage"] == 50
    assert runtime["npu_usage"] == 75
    assert runtime["power_watts"] == 9


def test_v2_audio_output_rtf_uses_generated_audio_not_input_duration(tmp_path):
    generated = tmp_path / "generated.wav"
    with wave.open(str(generated), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    runtime = _measured_runtime(
        time.perf_counter(),
        {
            "input": {
                "audio": [{"uri": "input.wav", "duration_seconds": 10.0}]
            },
            "resolved_protocol": {
                "output_schema": "audio@1",
                "parser": "generated_audio@1",
            },
        },
        json.dumps({"audio": str(generated)}),
    )

    assert runtime["audio_duration_seconds"] == 1.0
    assert runtime["rtf"] == pytest.approx(runtime["latency_ms"] / 1000)


def test_failed_v2_sample_remains_in_attribute_aggregation(monkeypatch, tmp_path):
    class Dataset:
        task_name = "ASR-en"
        ref_col = "text"

        def load(self, limit=0):
            rows = [_canonical("ok", "clean"), _canonical("timeout", "traffic")]
            return rows[:limit] if limit else rows

    class Prompt:
        def load(self, **doc):
            return doc["sample_id"]

    class Predictor:
        def inference(self, prompt):
            if prompt == "timeout":
                raise TimeoutError("request deadline exceeded")
            return "hello"

    class Evaluator:
        def __call__(self, pred, ref, **kwargs):
            return {"match": int(pred == ref)}

    from audio_evals.registry import registry

    monkeypatch.setattr(registry, "get_evaluator", lambda name: Evaluator())
    monkeypatch.setattr("audio_evals.eval_task.merge_data4view", lambda *args, **kwargs: None)
    event_path = tmp_path / "events.jsonl"
    task = EvalTask(
        dataset=Dataset(),
        prompt=Prompt(),
        predictor=Predictor(),
        evaluator="fake",
        post_process=[],
        agg=MeshAgg(group_by=["metadata__noise_type"], min_slice_size=1),
        recorder=Recorder(str(event_path)),
        run_context={
            "benchmark_id": "runtime-contract",
            "model": "fake",
            "inference_workers": 2,
            "warmup_requests": 0,
        },
    )

    overall, rows, answers = task.run(max_workers=2)

    assert len(rows) == 2
    assert answers == ["hello"]
    assert overall["sample_count"] == 2
    assert overall["failure_rate"] == 0.5
    assert overall["overall/failure_rate"] == 0.5
    assert overall["metadata__noise_type/clean/failure_rate"] == 0
    assert overall["metadata__noise_type/traffic/failure_rate"] == 1
    assert overall["metadata__noise_type/traffic/failure_stage/inference/count"] == 1
    assert overall["metadata__noise_type/traffic/failure_type/timeout/count"] == 1
    failed = next(row for row in rows if row["failure"] == 1)
    assert failed["sample_schema_version"] == "runtime-sample/2.0"
    assert failed["metadata__noise_type"] == "traffic"
    assert failed["timeout"] == 1
    assert failed["failure_stage"] == "inference"
    assert failed["failure_type"] == "timeout"

    error_event = next(
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["type"] == "error"
    )
    assert error_event["run_context"]["inference_workers"] == 2
    assert len(error_event["run_context"]["measurement_protocol_hash"]) == 64
    assert validate_json_schema(error_event, "event") == []


def test_event_reaggregation_counts_error_rows(monkeypatch):
    class Agg:
        def __call__(self, rows):
            return {"seen": len(rows)}

    class Registry:
        def __init__(self, paths):
            pass

        def get_agg(self, name):
            return Agg()

    monkeypatch.setattr("mesh_eval.scripts.merge_event_shards.Registry", Registry)
    result = aggregate_events(
        [
            {"type": "eval", "id": 0, "data": {"failure": 0}},
            {
                "type": "error",
                "id": 1,
                "data": {"failure": 1, "timeout": 1},
            },
        ],
        "fake",
        "",
    )
    assert result["seen"] == 2
    assert result["sample_count"] == 2
    assert result["failure_rate"] == 0.5


def test_runtime_aggregator_has_stable_optional_resource_denominators():
    result = MeshAgg(group_by=[], min_slice_size=1)._agg(
        [
            {
                "first_audio_chunk_latency_ms": 10,
                "rtf": 0.5,
                "cpu_usage": 20,
                "npu_usage": 40,
                "power_watts": 8,
                "failure": 0,
                "timeout": 0,
            },
            {
                "first_audio_chunk_latency_ms": 30,
                "rtf": 1.5,
                "failure": 1,
                "timeout": 1,
                "failure_stage": "inference",
                "failure_type": "timeout",
            },
        ]
    )
    assert result["overall/first_audio_chunk_latency_p50"] == 20
    assert result["overall/first_audio_chunk_latency_p50/sample_count"] == 2
    assert result["overall/rtf"] == 1.0
    assert result["overall/rtf/sample_count"] == 2
    assert result["overall/cpu_usage"] == 20
    assert result["overall/cpu_usage/sample_count"] == 1
    assert result["overall/npu_usage"] == 40
    assert result["overall/power_watts"] == 8
    assert result["overall/failure_rate"] == 0.5
    assert result["overall/timeout_rate"] == 0.5
