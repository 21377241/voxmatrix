import json
from pathlib import Path

import yaml

from mesh_eval.dataset.native import NativeClipDataset
from mesh_eval.dataset.full_eval import FullEvalDataset


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
CATALOG = ROOT / "suites" / "native" / "catalog.yaml"
REGISTRY = ROOT / "mesh_eval" / "registry" / "dataset" / "unified_speech.yaml"

FORBIDDEN_PATH_MARKERS = ("utt_clips", "smoke_", "8p31/runs")


def _catalog_slots():
    data = yaml.safe_load(CATALOG.read_text())
    return data.get("capabilities") or data.get("slots") or []


def _path_fields(row: dict):
    for key in ("WavPath", "WavPath2", "src_audio"):
        value = row.get(key)
        if value:
            yield key, str(value)


def test_native_target_speaker_is_aishell5_not_voxceleb():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "target_speaker_aishell_5.jsonl"),
        default_task="native-target-speaker-aishell-5",
        protocol="target_speaker",
        benchmark_id="native-target-speaker-aishell-5",
    ).load(10)
    assert len(rows) == 10
    assert all(row["capability"] == "target_speaker" for row in rows)
    assert all("aishell5" in str(row["sample_id"]).lower() for row in rows)
    assert all(row.get("WavPath2") for row in rows)
    assert all("reference" not in row for row in rows)
    assert {bool(row["reference_obj"]["should_execute"]) for row in rows} == {True, False}
    assert all(all(m not in row["WavPath"] for m in FORBIDDEN_PATH_MARKERS) for row in rows)
    assert all(all(m not in row["WavPath2"] for m in FORBIDDEN_PATH_MARKERS) for row in rows)


def test_native_code_switch_is_vocalbench_zh_knowledge():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "code_switch_vocalbench_zh.jsonl"),
        default_task="native-code-switch-vocalbench-zh",
        protocol="code_switch",
        benchmark_id="native-code-switch-vocalbench-zh",
    ).load(10)
    assert len(rows) == 10
    assert all(row["capability"] == "code_switch" for row in rows)
    assert "code_switching" in rows[0]["WavPath"]
    assert "knowledge" in rows[0]["WavPath"]
    assert all(all(m not in row["WavPath"] for m in FORBIDDEN_PATH_MARKERS) for row in rows)


def test_native_code_switch_open_ended_is_vocalbench_zh():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "code_switch_vocalbench_zh_open_ended.jsonl"),
        default_task="native-code-switch-vocalbench-zh-open-ended",
        protocol="code_switch",
        benchmark_id="native-code-switch-vocalbench-zh-open-ended",
    ).load(10)
    assert len(rows) == 10
    assert all(row["capability"] == "code_switch" for row in rows)
    assert all("open_ended" in row["WavPath"] for row in rows)
    assert all(all(m not in row["WavPath"] for m in FORBIDDEN_PATH_MARKERS) for row in rows)


def test_native_dcase_closed_set_has_labels_in_question():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "acoustic_scene_dcase2013_scenes.jsonl"),
        default_task="native-acoustic-scene-dcase2013-scenes",
        protocol="acoustic_scene",
        benchmark_id="native-acoustic-scene-dcase2013-scenes",
    ).load(10)
    assert len(rows) == 10
    assert "bus" in rows[0]["question"]
    assert rows[0]["text"] == rows[0]["answer"]


def test_full_eval_has_no_target_speaker_proxy():
    assert not hasattr(FullEvalDataset, "_set_target_speaker_proxy")
    assert not hasattr(FullEvalDataset, "_set_tool_call_proxy")


def test_registry_has_single_fsc_and_ts_loader():
    text = REGISTRY.read_text()
    assert "t4-fsc-instruction-following" in text
    assert "t4-fsc-tool-call" in text
    assert "full-fsc-slu" not in text
    assert "full-fsc-tool-call-proxy" not in text
    assert "full-voxceleb1-target-speaker-proxy" not in text
    assert "native-target-speaker-aishell-5" in text
    assert "full-uro-bench-meeting-summary" not in text


def test_catalog_has_no_full_eval_dataset_slots():
    slots = _catalog_slots()
    full_slots = [
        s["id"] for s in slots if str(s.get("dataset", "")).startswith("full-")
    ]
    assert full_slots == [], full_slots


def test_registry_has_no_stale_full_capability_keys():
    text = REGISTRY.read_text()
    stale = [
        "full-aishell-1-asr",
        "full-aishell-5-asr",
        "full-librispeech-asr",
        "full-common-voice-asr",
        "full-wenetspeech-asr",
        "full-magicdata-ramc-asr",
        "full-audiocaps-caption",
        "full-clotho-caption",
        "full-spoken-squad-qa",
        "full-uro-bench-qa",
        "full-vocalbench-qa",
        "full-vocalbench-zh-qa",
        "full-mmsu-audio-reasoning",
        "full-uro-bench-audio-reasoning",
        "full-vocalbench-audio-reasoning",
        "full-vocalbench-zh-audio-reasoning",
        "full-esc50-sound-event",
        "full-mmsu-sound-event",
        "full-mmsu-acoustic-scene",
        "full-mmsu-paralinguistic",
        "full-mmsu-speaker-count",
        "full-voxceleb1-verification",
        "full-covost2-zh-en",
        "full-covost2-en-zh",
        "full-mmsu-speech-translation",
        "full-mmsu-spoof-diagnostic",
        "full-libritts-tts-source",
        "full-aishell-1-asr",
        "native-aishell5-utt-asr",
        "native-sbcsae-utt-asr",
    ]
    missing_clean = [k for k in stale if f"{k}:" in text or f"\n{k}\n" in text]
    # YAML keys look like "full-foo:\n"
    present = [k for k in stale if f"\n{k}:" in text or text.startswith(f"{k}:")]
    assert present == [], present


def test_native_asr_points_at_l0_not_full_eval_or_utt_clips():
    catalog = CATALOG.read_text()
    for name in (
        "native-asr-aishell-1",
        "native-asr-aishell-5",
        "native-asr-librispeech",
        "native-asr-magicdata-ramc",
        "native-asr-sbcsae",
        "native-asr-common-voice",
        "native-asr-wenetspeech",
    ):
        assert f"dataset: {name}" in catalog
    assert "dataset: full-aishell-1-asr" not in catalog
    assert "dataset: full-librispeech-asr" not in catalog
    assert "dataset: native-aishell5-utt-asr" not in catalog
    assert "dataset: native-sbcsae-utt-asr" not in catalog

    reg = REGISTRY.read_text()
    assert "full-aishell-1-asr" not in reg
    assert "native-aishell5-utt-asr" not in reg

    aishell5 = NativeClipDataset(
        f_name=str(NATIVE / "asr_aishell_5.jsonl"),
        default_task="native-asr-aishell-5",
        protocol="asr",
        benchmark_id="native-asr-aishell-5",
    ).load(10)
    assert len(aishell5) == 10
    assert all("utt_clips" not in row["WavPath"] for row in aishell5)
    assert all(Path(row["WavPath"]).is_file() for row in aishell5)
    assert all(row.get("src_audio") for row in aishell5)
    assert len({row["src_audio"] for row in aishell5}) >= 5
    assert all(row["end_time"] - row["start_time"] >= 1.5 for row in aishell5)

    aishell1 = NativeClipDataset(
        f_name=str(NATIVE / "asr_aishell_1.jsonl"),
        default_task="native-asr-aishell-1",
        protocol="asr",
        benchmark_id="native-asr-aishell-1",
    ).load(10)
    assert len(aishell1) == 10
    assert all("/AIShell-1/" in row["WavPath"] for row in aishell1)

    cv = NativeClipDataset(
        f_name=str(NATIVE / "asr_common_voice.jsonl"),
        default_task="native-asr-common-voice",
        protocol="asr",
        benchmark_id="native-asr-common-voice",
    ).load(10)
    assert len(cv) == 10
    assert all("/en/clips/" in row["WavPath"] for row in cv)
    assert all(row["text"].isascii() for row in cv)


def test_native_long_form_uses_windowed_ondevice_smoke():
    catalog = CATALOG.read_text()
    assert "dataset: native-aishell4-long-form-asr" in catalog
    assert "dataset: native-ami-long-form-asr" in catalog
    lf_datasets = []
    current = False
    for line in catalog.splitlines():
        if line.startswith("- id: long_form_asr__"):
            current = True
            continue
        if current and line.startswith("  dataset:"):
            lf_datasets.append(line.strip())
            current = False
        elif current and line.startswith("- id:"):
            current = False
    assert lf_datasets and all("long-form-asr" in d for d in lf_datasets), lf_datasets

    rows = NativeClipDataset(
        f_name=str(NATIVE / "aishell_4_long_form_windowed.jsonl"),
        default_task="native-aishell4-long-form-asr",
        protocol="long_form_asr",
        benchmark_id="native-aishell4-long-form-asr",
    ).load(10)
    assert len(rows) >= 3
    durs = [float(r["audio_duration_seconds"]) for r in rows]
    assert all(25.0 <= d <= 330.0 for d in durs)  # S/M/L/XL smoke buckets
    assert all("windowed_ondevice" in r["WavPath"] for r in rows)
    assert all(r.get("text") for r in rows)
    assert all(r.get("bucket") in {"S", "M", "L", "XL"} for r in rows)
    assert all(r.get("crop_family_id") for r in rows)
    assert "XL" in {r.get("bucket") for r in rows}


def test_catalog_native_clip_wavpaths_exclude_utt_smoke_8p31():
    """Active native-* slots must not point WavPath at utt_clips / smoke / 8p31 runs."""
    reg = yaml.safe_load(REGISTRY.read_text())
    slots = [s for s in _catalog_slots() if str(s.get("dataset", "")).startswith("native-")]
    assert slots, "expected native catalog slots"

    checked = 0
    for slot in slots:
        ds_name = slot["dataset"]
        entry = reg.get(ds_name) or {}
        args = entry.get("args") or {}
        cls = str(entry.get("class") or "")
        if "NativeClipDataset" not in cls:
            continue
        f_name = args.get("f_name")
        if not f_name:
            continue
        path = ROOT / f_name
        assert path.is_file(), f"missing manifest for {ds_name}: {path}"
        protocol = args.get("protocol") or "asr"
        rows = NativeClipDataset(
            f_name=str(path),
            default_task=ds_name,
            protocol=protocol,
            benchmark_id=args.get("benchmark_id") or ds_name,
        ).load(10)
        assert rows, f"empty load: {ds_name}"
        for row in rows:
            for key, value in _path_fields(row):
                for marker in FORBIDDEN_PATH_MARKERS:
                    assert marker not in value, f"{ds_name} {key} contains {marker}: {value}"
            assert Path(row["WavPath"]).is_file(), f"{ds_name} missing audio: {row['WavPath']}"
            checked += 1
    assert checked >= 50, checked


def test_diar_attr_use_windowed_families():
    for name, fname, protocol in (
        ("native-diarization-aishell-4", "diarization_aishell_4_windowed.jsonl", "diarization"),
        ("native-speaker-attribution-chime-6", "speaker_attribution_chime_6_windowed.jsonl", "speaker_attribution"),
    ):
        rows = NativeClipDataset(
            f_name=str(NATIVE / fname),
            default_task=name,
            protocol=protocol,
            benchmark_id=name,
        ).load(0)
        assert len(rows) == 80
        assert [r["window_bucket"] for r in rows[:8]] == ["S", "M", "L", "XL"] * 2
        assert len({r["crop_family_id"] for r in rows}) == 20
        assert all("windowed_ondevice" in row["WavPath"] for row in rows)
        assert all(all(m not in row["WavPath"] for m in FORBIDDEN_PATH_MARKERS) for row in rows)
        assert all(Path(row["WavPath"]).is_file() for row in rows)


def test_native_slots_are_strict_subsets_of_full():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "align_native_to_full", ROOT / "scripts" / "align_native_to_full.py"
    )
    align = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(align)
    problems = []
    for slot, capability, native_path in align.runnable_slots():
        full_path = align.full_path_for(slot, capability, native_path)
        if not full_path.is_file():
            problems.append(f"{slot}: missing {full_path.name}")
            continue
        native = align.load_jsonl(native_path)
        full = align.load_jsonl(full_path)
        for row, hit in zip(native, align.match_rows(slot, native, full)):
            if hit is None:
                problems.append(f"{slot}: {row.get('sample_id')} not in Full")
            elif align.eval_view(row) != align.eval_view(align.aligned_row(row, hit)):
                problems.append(f"{slot}: {row.get('sample_id')} differs from Full")
    assert problems == [], problems[:20]
