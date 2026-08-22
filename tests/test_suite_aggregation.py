import json

import pytest

from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.scripts.reaggregate_events import aggregate_event_files


def result_event(capability, scenario, accuracy, *, benchmark_id="benchmark"):
    return {
        "schema_version": "result-event/2.0",
        "type": "eval",
        "id": f"{capability}-{scenario}-{accuracy}",
        "sample_id": f"{capability}-{scenario}-{accuracy}",
        "dimensions": {
            "task": "audio_speech_understanding",
            "capability": capability,
            "scenario": {"primary": scenario, "secondary": "unknown"},
            "metadata": {"speaker_count": 3},
            "subset_profile": {
                "record_id": f"record-{capability}",
                "source_benchmark_id": "source",
                "subset_id": "test",
                "source_protocol_id": capability,
                "mapping_status": "direct",
                "resource_status": "available",
            },
        },
        "resolved_protocol": {},
        "run_context": {"benchmark_id": benchmark_id},
        "data": {
            "accuracy": accuracy,
            "metric_names": "accuracy",
            "sample_schema_version": "runtime-sample/2.0",
            "failure": 0,
            "timeout": 0,
        },
    }


def write_events(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_multi_file_suite_aggregation_builds_capability_and_scenario_views(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_events(first, [result_event("acoustic_scene", "meeting", 1.0)])
    write_events(
        second,
        [
            result_event("acoustic_scene", "meeting", 0.0),
            result_event("meeting_summary", "meeting", 1.0),
        ],
    )

    report = aggregate_event_files(
        [str(first), str(second)], report_format="both", min_slice_size=1
    )

    assert report["sample_count"] == 2
    assert report["by_capability"]["acoustic_scene"]["accuracy"] == 0.5
    assert (
        report["by_scenario"]["meeting"]["by_capability"]["acoustic_scene"][
            "accuracy"
        ]
        == 0.5
    )
    assert "meeting_summary" not in report["by_capability"]
    assert "metadata__speaker_count" not in report["overall"]
    assert report["flat"]["mapping_status/direct/sample_count"] == 2


def test_der_uses_reference_duration_sufficient_statistics():
    rows = [
        {
            "capability": "diarization",
            "scenario__primary": "meeting",
            "metric_names": "der",
            "der": 1.0,
            "der_error_duration": 1.0,
            "der_reference_duration": 1.0,
        },
        {
            "capability": "diarization",
            "scenario__primary": "meeting",
            "metric_names": "der",
            "der": 0.0,
            "der_error_duration": 0.0,
            "der_reference_duration": 9.0,
        },
    ]

    result = MeshAgg(min_slice_size=1)._agg(rows)

    assert result["overall/der"] == pytest.approx(0.1)
    assert result["overall/der/aggregation"] == "reference_duration_weighted"
    assert result["scenario__primary+capability/meeting|diarization/der"] == (
        pytest.approx(0.1)
    )


def test_der_keeps_macro_mean_when_sufficient_statistics_are_partial():
    rows = [
        {
            "capability": "diarization",
            "metric_names": "der",
            "der": 1.0,
            "der_error_duration": 1.0,
            "der_reference_duration": 1.0,
        },
        {"capability": "diarization", "metric_names": "der", "der": 0.0},
    ]

    result = MeshAgg(min_slice_size=1)._agg(rows)

    assert result["overall/der"] == pytest.approx(0.5)
    assert "overall/der/aggregation" not in result
    assert result["overall/der/sufficient_statistics_count"] == 1
