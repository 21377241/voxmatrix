import json
import sys
from pathlib import Path

import pytest

from mesh_eval.core.schema_v2 import validate_canonical_sample, validate_json_schema
from mesh_eval.scripts import build_manifest_from_events


def _read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_source_events(path: Path, audio_path: Path):
    events = [
        {
            "type": "prompt",
            "id": 17,
            "data": {
                "content": [
                    {
                        "role": "user",
                        "contents": [{"type": "audio", "value": str(audio_path)}],
                    }
                ]
            },
        },
        {
            "type": "inference",
            "id": 17,
            "data": {"content": "raw prediction", "runtime": {"latency_ms": 12.5}},
        },
        {
            "type": "post_process",
            "id": 17,
            "data": {"content": "normalized prediction"},
        },
        {
            "type": "eval",
            "id": 17,
            "data": {"ref": "reference text", "wer%": 0.0},
        },
    ]
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_event_builder_writes_canonical_v2_and_replay_sidecar(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not-decoded-by-the-builder")
    source_path = tmp_path / "events.jsonl"
    _write_source_events(source_path, audio_path)
    output_path = tmp_path / "manifest.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_manifest_from_events",
            "--output",
            str(output_path),
            "--source",
            f"unit-asr={source_path}",
            "--limit_per_dataset",
            "1",
        ],
    )
    build_manifest_from_events.main()

    rows = _read_jsonl(output_path)
    assert len(rows) == 1
    sample = rows[0]
    assert sample["schema_version"] == "2.0"
    assert sample["sample_id"] == "unit-asr_00017"
    assert sample["input"]["audio"] == [{"uri": str(audio_path)}]
    assert "eval_info" not in sample
    assert "WavPath" not in sample
    assert validate_canonical_sample(sample, formal=False) == []
    assert validate_json_schema(sample, "canonical") == []

    replay_path = output_path.with_suffix(".replay.jsonl")
    replay = _read_jsonl(replay_path)
    assert [(event["type"], event["id"]) for event in replay] == [
        ("inference", 0),
        ("post_process", 0),
    ]
    assert replay[0]["data"] == {
        "content": "raw prediction",
        "runtime": {"latency_ms": 12.5},
    }
    assert replay[1]["data"] == {"content": "normalized prediction"}


def test_event_builder_keeps_explicit_v1_compatibility(tmp_path, monkeypatch):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not-decoded-by-the-builder")
    source_path = tmp_path / "events.jsonl"
    _write_source_events(source_path, audio_path)
    output_path = tmp_path / "manifest-v1.jsonl"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_manifest_from_events",
            "--output",
            str(output_path),
            "--source",
            f"unit-asr={source_path}",
            "--limit_per_dataset",
            "1",
            "--schema-version",
            "1.0",
        ],
    )
    build_manifest_from_events.main()

    rows = _read_jsonl(output_path)
    assert len(rows) == 1
    assert rows[0]["schema_version"] == "1.0"
    assert rows[0]["WavPath"] == str(audio_path)
    assert rows[0]["eval_info"] == {
        "inference": {"content": "normalized prediction"},
        "post_process": {"content": "normalized prediction"},
    }
    assert not output_path.with_suffix(".replay.jsonl").exists()


def test_event_builder_requires_explicit_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_manifest_from_events",
            "--output",
            str(tmp_path / "manifest.jsonl"),
        ],
    )

    with pytest.raises(
        ValueError, match="at least one --source dataset=path is required"
    ):
        build_manifest_from_events.main()
