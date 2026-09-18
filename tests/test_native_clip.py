import json
from pathlib import Path

from mesh_eval.dataset.native import NativeClipDataset
from mesh_eval.dataset.full_eval import FullEvalDataset


ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "mesh_eval" / "data" / "native"


def test_native_target_speaker_is_aishell5_not_voxceleb():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "aishell5_target_speaker.jsonl"),
        default_task="native-ts",
        protocol="target_speaker",
        benchmark_id="native-aishell5-target-speaker",
    ).load(10)
    assert len(rows) == 10
    assert all(row["capability"] == "target_speaker" for row in rows)
    assert all("aishell5" in str(row["sample_id"]).lower() for row in rows)
    assert all(row.get("WavPath2") for row in rows)
    assert {bool(row["reference"]["should_execute"]) for row in rows} == {True, False}


def test_native_code_switch_is_vocalbench_zh_knowledge():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "vocalbench_zh_cs_knowledge.jsonl"),
        default_task="native-cs",
        protocol="code_switch",
        benchmark_id="native-vocalbench-zh-code-switch",
    ).load(10)
    assert len(rows) == 10
    assert all(row["capability"] == "code_switch" for row in rows)
    assert "code_switching" in rows[0]["WavPath"]


def test_native_dcase_closed_set_has_labels_in_question():
    rows = NativeClipDataset(
        f_name=str(NATIVE / "dcase2013_acoustic_scene.jsonl"),
        default_task="native-as",
        protocol="acoustic_scene",
        benchmark_id="native-dcase2013-acoustic-scene",
    ).load(10)
    assert len(rows) == 10
    assert "bus" in rows[0]["question"]
    assert rows[0]["text"] == rows[0]["answer"]


def test_full_eval_has_no_target_speaker_proxy():
    assert not hasattr(FullEvalDataset, "_set_target_speaker_proxy")
    assert not hasattr(FullEvalDataset, "_set_tool_call_proxy")


def test_registry_has_single_fsc_and_ts_loader():
    text = (ROOT / "mesh_eval" / "registry" / "dataset" / "unified_speech.yaml").read_text()
    assert "t4-fsc-instruction-following" in text
    assert "t4-fsc-tool-call" in text
    assert "full-fsc-slu" not in text
    assert "full-fsc-tool-call-proxy" not in text
    assert "full-voxceleb1-target-speaker-proxy" not in text
    assert "native-aishell5-target-speaker" in text
    assert "full-uro-bench-meeting-summary" not in text
