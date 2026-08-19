import json
from copy import deepcopy

import pytest

from annotation.pipeline.migrate_v2 import migrate_records
from audio_evals.dataset.prepared import PreparedAudioJsonl
from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.core.protocol import (
    CapabilityProtocolRegistry,
    ProtocolResolutionError,
    validate_protocol_output,
    validate_protocol_registry,
)
from mesh_eval.core.schema import normalize_record, validate_record
from mesh_eval.core.schema_v2 import (
    adapt_v1_to_v2,
    result_event_v2,
    to_runtime_sample,
    validate_canonical_sample,
    validate_json_schema,
)
from mesh_eval.scripts.build_manifest_from_registry import convert_doc


def canonical_sample(**overrides):
    sample = {
        "schema_version": "2.0",
        "sample_id": "vehicle_asr_0001",
        "dataset": "aishell5",
        "task": "speech_recognition_transcription",
        "capability": "asr",
        "capability_protocol": "speech_recognition_transcription/asr@1",
        "scenario": {
            "primary": "vehicle",
            "secondary": "highway_driving",
        },
        "metadata": {
            "device_type": "car_mic",
            "recording_setup": "far_field",
            "noise_type": "vehicle_noise",
            "snr_bucket": "medium",
            "speaker_count": 2,
            "overlap_level": "mild",
            "window_state": "driver_window_open",
            "language": "zh",
        },
        "input": {
            "audio": [
                {
                    "uri": "/prepared/aishell5/vehicle_asr_0001.wav",
                    "sampling_rate": 16000,
                    "duration_seconds": 4.32,
                }
            ]
        },
        "reference": {"text": "示例文本"},
        "provenance": {
            "scenario": {"source": "dataset_native", "confidence": 1.0},
            "metadata": {
                "noise_type": {
                    "source": "dataset_native",
                    "confidence": 1.0,
                },
                "snr_bucket": {
                    "source": "acoustic_model",
                    "confidence": 0.82,
                },
            },
            "labels": {
                "reference.text": {
                    "source": "dataset_native",
                    "confidence": 1.0,
                }
            },
        },
        "use_bucket": "formal_subscores",
        "split": "test",
    }
    sample.update(overrides)
    return sample


def test_0803_example_passes_python_and_json_schema():
    sample = canonical_sample()

    assert validate_protocol_registry() == []
    assert validate_canonical_sample(sample, formal=True) == []
    assert validate_json_schema(sample, "canonical") == []


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "task": "runtime",
                "capability": "latency",
                "capability_protocol": "runtime/latency@1",
            },
            "unknown V2 task: runtime",
        ),
        ({"scenario": "vehicle"}, "V2 scenario must be an object"),
        ({"scenario": "phone"}, "V2 scenario must be an object"),
        (
            {"scenario": {"primary": "phone", "secondary": None}},
            "unknown V2 scenario.primary: phone",
        ),
    ],
)
def test_v2_rejects_legacy_task_and_scenario_shapes(updates, message):
    sample = canonical_sample(**updates)

    assert message in validate_canonical_sample(sample, formal=False)
    assert validate_json_schema(sample, "canonical")


def test_v2_validates_task_capability_and_scenario_pairs_in_both_validators():
    wrong_task_pair = canonical_sample(capability="qa")
    wrong_scenario_pair = canonical_sample(
        scenario={"primary": "vehicle", "secondary": "kitchen"}
    )

    assert any(
        "does not belong" in error
        for error in validate_canonical_sample(wrong_task_pair, formal=False)
    )
    assert validate_json_schema(wrong_task_pair, "canonical")
    assert any(
        "does not belong" in error
        for error in validate_canonical_sample(wrong_scenario_pair, formal=False)
    )
    assert validate_json_schema(wrong_scenario_pair, "canonical")


def test_v1_to_v2_is_deterministic_and_preserves_ambiguous_legacy_values():
    legacy = {
        "schema_version": "1.0",
        "sample_id": "legacy-1",
        "dataset": "legacy-asr",
        "task": "speech_understanding",
        "capability": "asr",
        "scenario": "phone",
        "condition": {
            "acoustic": "light_noise",
            "spatial": "front_seat",
            "speaker": "multi_speaker",
            "device": "phone_mic",
        },
        "input": {"audio_path": "legacy.wav"},
        "reference": {"text": "hello"},
        "metrics": ["wer"],
        "use_bucket": "formal_subscores",
        "split": "validation",
    }

    first = adapt_v1_to_v2(legacy)
    second = adapt_v1_to_v2(deepcopy(legacy))

    assert first == second
    assert first["task"] == "speech_recognition_transcription"
    assert first["scenario"] == {"primary": "unknown", "secondary": None}
    assert first["metadata"]["device_type"] == "phone_mic"
    assert first["legacy"]["scenario"] == "phone"
    assert first["legacy"]["condition"] == legacy["condition"]
    assert first["use_bucket"] == "coverage_debt"
    assert any("scenario=phone" in item for item in first["conversion_warnings"])
    assert any("ambiguous legacy acoustic" in item for item in first["conversion_warnings"])
    assert any("legacy spatial" in item for item in first["conversion_warnings"])
    assert any("legacy speaker" in item for item in first["conversion_warnings"])
    assert validate_json_schema(first, "canonical") == []

    migrated, report = migrate_records([legacy])
    assert migrated == [first]
    assert report["output_count"] == 1
    assert report["warning_count"] == len(first["conversion_warnings"])


def test_formal_protocol_fails_closed_for_mapping_reference_metric_and_output():
    registry = CapabilityProtocolRegistry(mode="formal")

    unmapped = canonical_sample(capability_protocol="future/asr@1")
    with pytest.raises(ProtocolResolutionError, match="no mapped protocol"):
        registry.resolve(unmapped)

    missing_reference = canonical_sample(reference={})
    with pytest.raises(ProtocolResolutionError, match="requires reference fields: text"):
        registry.resolve(missing_reference)

    unknown_metric = canonical_sample(
        evaluation_hints={"metrics": ["future_metric"], "source": "test"}
    )
    with pytest.raises(ProtocolResolutionError, match="unmapped formal metrics"):
        registry.resolve(unknown_metric)

    structured = canonical_sample(
        task="spoken_agentic_interaction",
        capability="tool_call",
        capability_protocol="spoken_agentic_interaction/tool_call@1",
        input={"audio": [{"uri": "tool.wav"}], "instruction": "navigate"},
        reference={"expected_output": {"tool": "navigate"}},
    )
    resolved = registry.resolve(structured)
    with pytest.raises(ProtocolResolutionError, match="invalid JSON"):
        validate_protocol_output("not-json", resolved)


def test_tts_protocol_selects_language_specific_evaluator_and_requires_evidence():
    registry = CapabilityProtocolRegistry(mode="formal")
    base = canonical_sample(
        task="speech_generation_delivery",
        capability="speech_generation",
        capability_protocol="speech_generation_delivery/speech_generation@1",
        input={"text": "hello"},
        reference={"text": "hello"},
        metadata={
            "device_type": "studio_mic",
            "recording_setup": "synthetic",
            "noise_type": "clean",
            "snr_bucket": "high",
            "speaker_count": 1,
            "overlap_level": "none",
            "language": "en-US",
        },
    )

    resolved = registry.resolve(base)
    assert resolved["evaluators"].count("seed-tts-eval-asr-wer-en") == 1
    assert "seed-tts-eval-asr-wer-zh" not in resolved["evaluators"]

    unknown_language = deepcopy(base)
    unknown_language["metadata"].pop("language")
    with pytest.raises(ProtocolResolutionError, match="requires metadata.language"):
        registry.resolve(unknown_language)

    unknown_language["evaluation_hints"] = {
        "evaluators": ["seed-tts-eval-asr-wer-en", "simo", "utmos", "dnsmos"],
        "source": "explicit_test_override",
    }
    assert registry.resolve(unknown_language)["evaluators"][0] == (
        "seed-tts-eval-asr-wer-en"
    )


def test_canonical_runtime_event_projection_and_v1_loader_compatibility():
    sample = canonical_sample()
    runtime = to_runtime_sample(
        sample,
        protocol_mode="formal",
        run_context={"benchmark_id": "vehicle-asr", "model_config_hash": "abc"},
    )
    event = result_event_v2("eval", "event-1", runtime, {"cer%": 0.0})

    assert runtime["WavPath"] == sample["input"]["audio"][0]["uri"]
    assert runtime["scenario__primary"] == "vehicle"
    assert runtime["metadata__window_state"] == "driver_window_open"
    assert runtime["protocol_hash"] == runtime["resolved_protocol"]["hash"]
    assert validate_json_schema(runtime, "runtime") == []
    assert event["dimensions"]["metadata"] == sample["metadata"]
    assert event["resolved_protocol"]["id"] == sample["capability_protocol"]
    assert validate_json_schema(event, "event") == []

    legacy = {
        "sample_id": "legacy-replay",
        "dataset": "legacy",
        "task": "speech_understanding",
        "capability": "asr",
        "scenario": "phone",
        "condition": {
            "acoustic": "clean",
            "spatial": "near_field",
            "speaker": "single_speaker",
            "device": "phone_mic",
            "interaction": "single_turn",
        },
        "input": {"audio_path": "legacy.wav"},
        "reference": {"text": "hello"},
        "metrics": ["wer"],
        "use_bucket": "diagnostic_evidence",
        "split": "test",
    }
    normalized = normalize_record(legacy, ref_col="text")
    assert normalized["WavPath"] == "legacy.wav"
    assert validate_record(normalized, ref_col="text") == []


def test_v2_aggregation_reports_slice_coverage_provenance_and_small_samples():
    rows = [
        {
            "sample_schema_version": "runtime-sample/2.0",
            "task": "speech_recognition_transcription",
            "capability": "asr",
            "scenario__primary": "vehicle",
            "scenario__secondary": "highway_driving",
            "metadata__noise_type": "vehicle_noise",
            "metadata__snr_bucket": "medium",
            "cer%": 0.0,
            "_slice_provenance": {
                "scenario": {"source": "dataset_native", "confidence": 1.0},
                "metadata": {
                    "noise_type": {
                        "source": "dataset_native",
                        "confidence": 1.0,
                    },
                    "snr_bucket": {
                        "source": "acoustic_model",
                        "confidence": 0.8,
                    },
                },
            },
        },
        {
            "sample_schema_version": "runtime-sample/2.0",
            "task": "speech_recognition_transcription",
            "capability": "asr",
            "scenario__primary": "unknown",
            "metadata__noise_type": "unknown",
            "cer%": 10.0,
            "_slice_provenance": {
                "scenario": {"source": "unknown", "confidence": 0.0},
                "metadata": {
                    "noise_type": {"source": "unknown", "confidence": 0.0}
                },
            },
        },
    ]

    result = MeshAgg(
        group_by=["scenario__primary", "metadata__noise_type"],
        min_slice_size=2,
    )._agg(rows)

    assert result["coverage/v2_sample_count"] == 2
    assert result["coverage/scenario__primary/known_rate"] == 0.5
    assert result["coverage/scenario__secondary/source/dataset_native/count"] == 1
    assert result["coverage/metadata__noise_type/missing_or_unknown_count"] == 1
    assert result["coverage/metadata__noise_type/source/dataset_native/count"] == 1
    assert result["coverage/metadata__noise_type/confidence_mean"] == 1.0
    assert result["scenario__primary/vehicle/warning"] == "small_sample"


def test_formal_builder_writes_canonical_v2_and_prepared_loader_reads_input_audio(
    tmp_path,
):
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"placeholder")
    converted = convert_doc(
        dataset_name="aishell-1",
        dataset_ref_col="text",
        default_task="ASR-zh",
        doc={"WavPath": str(audio_path), "text": "你好"},
        index=0,
        use_bucket="diagnostic_evidence",
        split="test",
        canonical_v2=True,
    )

    assert converted["schema_version"] == "2.0"
    assert converted["input"]["audio"][0]["uri"] == str(audio_path)
    assert "WavPath" not in converted
    assert validate_json_schema(converted, "canonical") == []

    manifest = tmp_path / "manifest.jsonl"
    relative = deepcopy(converted)
    relative["input"]["audio"][0]["uri"] = audio_path.name
    manifest.write_text(json.dumps(relative) + "\n", encoding="utf-8")
    loaded = PreparedAudioJsonl(
        str(manifest),
        default_task="ASR-zh",
        ref_col="text",
        require_success=False,
    ).load()
    assert loaded[0]["input"]["audio"][0]["uri"] == str(audio_path.resolve())
