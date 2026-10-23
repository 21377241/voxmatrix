"""Full / Native indexes must carry subset-manifest annotation fields."""

from __future__ import annotations

from mesh_eval.data.full_index_meta import (
    REQUIRED_METADATA,
    REQUIRED_SCENARIO,
    REQUIRED_TOP,
    audit_row,
    audit_slot_map,
    enrich_row,
    slot_id_from_filename,
)


def test_slot_id_from_filename():
    assert slot_id_from_filename("qa", "qa_uro_bench.jsonl") == "qa__uro_bench"
    assert (
        slot_id_from_filename("audio_reasoning", "audio_reasoning_mmsu.jsonl")
        == "audio_reasoning__mmsu"
    )


def test_enrich_row_fills_absent_including_unknown():
    record = {
        "record_id": "demo__rec",
        "scenario": {"primary": "unknown", "secondary": "unknown"},
        "metadata": {
            "language": "zh",
            "device_type": "unknown",
            "recording_setup": "unknown",
            "noise_type": "none",
            "speaker_count": "1",
            "overlap_level": "none",
            "snr_bucket": "unknown",
            "distance": "unknown",
            "motion_state": "unknown",
            "crowd_density": "unknown",
            "weather": "unknown",
            "window_state": "unknown",
            "authorization_state": "unknown",
            "speaker_age_group": "unknown",
            "speaker_role": "unknown",
            "language_mix": "none",
        },
    }
    row = enrich_row(
        {"sample_id": "s1", "WavPath": "/x.wav", "capability": "qa", "dataset": "demo"},
        record,
    )
    assert row["language"] == "zh"
    assert row["scenario"]["primary"] == "unknown"
    assert row["metadata"]["device_type"] == "unknown"
    assert row["subset_manifest_record_id"] == "demo__rec"
    assert not audit_row(row)


def test_enrich_row_never_overwrites_richer_sample():
    record = {
        "record_id": "demo__rec",
        "scenario": {"primary": "unknown", "secondary": "unknown"},
        "metadata": {"language": "en", "device_type": "unknown"},
    }
    row = enrich_row(
        {
            "sample_id": "s1",
            "language": "zh",
            "scenario": {"primary": "vehicle", "secondary": "unknown"},
            "metadata": {"language": "zh", "device_type": "phone_mic"},
        },
        record,
    )
    assert row["language"] == "zh"
    assert row["scenario"]["primary"] == "vehicle"
    assert row["metadata"]["device_type"] == "phone_mic"


def test_audit_slot_map_indexes_have_required_fields():
    problems = audit_slot_map()
    assert not problems, "\n".join(problems[:40])


def test_required_field_constants_cover_manifest_shape():
    assert "language" in REQUIRED_TOP
    assert "primary" in REQUIRED_SCENARIO
    assert "device_type" in REQUIRED_METADATA
