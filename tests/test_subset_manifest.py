import json
from copy import deepcopy

import pytest

from mesh_eval.core.schema_v2 import (
    result_event_v2,
    to_runtime_sample,
    validate_canonical_sample,
    validate_json_schema,
)
from mesh_eval.core.subset_manifest import (
    SubsetManifestCatalog,
    SubsetManifestError,
    enrich_canonical_sample,
)
from mesh_eval.scripts.attach_subset_profiles import attach_subset_profiles


def subset_record(**overrides):
    record = {
        "schema_version": "2.0.0",
        "record_id": "demo__development__asr",
        "benchmark": {
            "id": "demo_benchmark",
            "name": "Demo",
            "resource_status": "available",
        },
        "framework": {
            "task": "speech_recognition_transcription",
            "task_id": "T1",
            "capability": "asr",
            "mapping_status": "direct",
        },
        "scenario": {"primary": "meeting", "secondary": "small_meeting"},
        "protocol": {"id": "asr", "metrics": ["wer"]},
        "metadata": {
            "language": "en",
            "speaker_count": "2-4",
            "device_type": "binaural_headset",
            "recording_setup": "close_talk",
            "noise_type": "home_appliance",
            "motion_state": "driving",
            "overlap_level": "high",
        },
        "provenance": {"annotation_confidence": "high"},
        "subset": {
            "id": "development",
            "split": "development",
            "selector": {
                "mode": "line_glob",
                "pattern": "development.jsonl",
                "protocol_ids": ["asr"],
                "split": "development",
                "subset": "development",
            },
        },
    }
    record.update(overrides)
    return record


def canonical_sample():
    return {
        "schema_version": "2.0",
        "sample_id": "demo-1",
        "dataset": "demo_alias",
        "task": "speech_recognition_transcription",
        "capability": "asr",
        "capability_protocol": "speech_recognition_transcription/asr@1",
        "scenario": {"primary": "unknown", "secondary": "unknown"},
        "metadata": {"language": "unknown", "speaker_count": None},
        "input": {"audio": [{"uri": "/tmp/demo.wav"}]},
        "reference": {"text": "hello"},
        "provenance": {
            "scenario": {"source": "unknown"},
            "metadata": {},
            "labels": {"reference.text": {"source": "dataset_native"}},
        },
        "use_bucket": "diagnostic_evidence",
        "split": "dev",
    }


def write_manifest(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_subset_catalog_filters_excluded_capabilities_and_enriches_v2(tmp_path):
    excluded = subset_record(
        record_id="demo__meeting_summary",
        framework={
            "task": "audio_speech_understanding",
            "task_id": "T2",
            "capability": "meeting_summary",
            "mapping_status": "direct",
        },
        protocol={"id": "meeting_summary", "metrics": ["rouge_l"]},
        subset={
            "id": "development",
            "split": "development",
            "selector": {
                "mode": "line_glob",
                "pattern": "development.jsonl",
                "protocol_ids": ["meeting_summary"],
                "split": "development",
                "subset": "development",
            },
        },
    )
    path = tmp_path / "subset.jsonl"
    write_manifest(path, [subset_record(), excluded])
    catalog = SubsetManifestCatalog(
        str(path), benchmark_aliases={"demo_alias": "demo_benchmark"}
    )

    assert catalog.total_record_count == 2
    assert len(catalog.records) == 1
    assert catalog.excluded_record_count == 1
    record = catalog.find_for_sample(
        canonical_sample(), subset_id="development"
    )
    enriched = enrich_canonical_sample(canonical_sample(), record, catalog)

    assert enriched["split"] == "dev"
    assert enriched["scenario"] == {
        "primary": "meeting",
        "secondary": "unknown",
    }
    assert enriched["metadata"]["scenario_subtype"] == "small_meeting"
    assert enriched["metadata"]["speaker_count"] is None
    assert enriched["metadata"]["speaker_count_bucket"] == "2-4"
    assert enriched["metadata"]["language"] == "en"
    assert enriched["metadata"]["device_type_subtype"] == "binaural_headset"
    assert enriched["metadata"]["recording_setup_subtype"] == "close_talk"
    assert enriched["metadata"]["noise_type_subtype"] == "home_appliance"
    assert enriched["metadata"]["motion_state_subtype"] == "driving"
    assert enriched["metadata"]["overlap_level_subtype"] == "high"
    assert enriched["subset_profile"]["record_id"] == record["record_id"]
    assert validate_canonical_sample(enriched, formal=False) == []
    assert validate_json_schema(enriched, "canonical") == []

    runtime = to_runtime_sample(
        enriched,
        protocol_mode="diagnostic",
        run_context={"benchmark_id": "suite-demo"},
    )
    event = result_event_v2("eval", 1, runtime, {"wer%": 0.0})
    assert runtime["subset_manifest_record_id"] == record["record_id"]
    assert runtime["metadata__scenario_subtype"] == "small_meeting"
    assert runtime["metadata__device_type_subtype"] == "binaural_headset"
    assert runtime["benchmark_id"] == "suite-demo"
    assert event["dimensions"]["subset_profile"]["subset_id"] == "development"
    assert event["dimensions"]["mapping_status"] == "direct"
    assert validate_json_schema(runtime, "runtime") == []
    assert validate_json_schema(event, "event") == []


def test_subset_catalog_rejects_ambiguous_fallback_key(tmp_path):
    first = subset_record()
    second = deepcopy(first)
    second["record_id"] = "duplicate-key"
    path = tmp_path / "ambiguous.jsonl"
    write_manifest(path, [first, second])

    with pytest.raises(SubsetManifestError, match="ambiguous subset join key"):
        SubsetManifestCatalog(str(path))


def test_attach_subset_profiles_writes_valid_enriched_jsonl(tmp_path):
    subset_path = tmp_path / "subset.jsonl"
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_manifest(subset_path, [subset_record()])
    input_path.write_text(json.dumps(canonical_sample()) + "\n", encoding="utf-8")

    result = attach_subset_profiles(
        str(input_path),
        str(output_path),
        subset_manifest=str(subset_path),
        subset_id="development",
        benchmark_aliases={"demo_alias": "demo_benchmark"},
    )

    enriched = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["input"] == 1
    assert result["written"] == 1
    assert result["unmatched"] == 0
    assert enriched["subset_profile"]["record_id"] == "demo__development__asr"
