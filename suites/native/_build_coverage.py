#!/usr/bin/env python3
"""Build per-benchmark native coverage catalog. Run from repo root once."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
ANN = ROOT / "annotation" / "benchmark_subset_manifest.jsonl"
REG = ROOT / "mesh_eval" / "registry" / "dataset" / "unified_speech.yaml"
CAT = ROOT / "suites" / "native" / "catalog.yaml"
P8 = Path("/mnt/afs/users/wangyl/8p31/runs")
SMOKE_ASR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_all")
SMOKE_DIAR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_diar_all")
SMOKE_ATTR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_attr_all")

EXCLUDE = {
    ("asr", "aishell_2"): "resource_status unavailable",
    ("asr", "easycom"): "Qwen3-Omni Instruct 60s early-stop; smoke invalid until sliced",
    ("instruction_following", "massive"): "mapping_status insufficient",
    ("instruction_following", "fluent_speech_commands"): (
        "Moved to spoken_slu (intent+slots); not IF. See docs/待讨论问题0924.md §8."
    ),
    ("instruction_following", "slurp"): (
        "Moved to spoken_slu (intent+slots); not IF. See docs/待讨论问题0924.md §8."
    ),
    ("tool_call", "massive"): "Suspended 2026-09-26: tool_call not evaluated for now.",
    ("spoof_detection", "asvspoof"): "Suspended 2026-09-26: spoof_detection not evaluated for now.",
    ("speaker_attribution", "cn_celeb"): "SID mis-tag; smoke do not score",
    ("speaker_attribution", "aishell_5"): (
        "Out of windowed_ondevice track (legacy 30s windows retired); not rebuilt."
    ),
    ("diarization", "misp"): "smoke excluded this round",
}

# Meeting track: formal set from scripts/sample_windowed_ondevice_eval.py
# (20 complete S/M/L/XL families per dataset, family-major rows).
WINDOWED_MEETING = ("aishell_4", "alimeeting", "ami", "chime_6")
WINDOWED_NOTE = "Windowed on-device S/M/L/XL nested families (eval_data/windowed_ondevice; 20 families x 4 buckets)."
WINDOWED_AUDIO_NOTE = {
    "aishell_4": "Continuous crop.",
    "alimeeting": "Continuous crop.",
    "ami": "AMI SDM utterance overlay.",
    "chime_6": "CHiME-6 ref-array 4-ch mean crop.",
}
ZH_DIGIT_NOTE = (
    "CER uses zh TextNorm (angle tags stripped). Digit spoken forms (零零六) vs "
    "Arabic numerals in hyp may inflate CER; report as known bias."
)

# English ASR prompts/metrics. common_voice follows smoke #6: English CV clips,
# not the zh-HK full_eval draw that produced WER 530%.
EN_ASR = {"librispeech", "common_voice", "sbcsae", "ami", "chime_6"}
SPLIT_RANK = {"test": 0, "evaluation": 1, "eval": 1, "dev": 2, "development": 2, "train": 3}

HEAD = """# Native coverage entry: every executable (capability, benchmark) pair.
# Small-n (default 10). Eval indexes under mesh_eval/data/native/ (L0 / eval_clips).
# Annotation routing: annotation/benchmark_subset_manifest.jsonl.
# No FullEval / utt_clips / smoke / 8p31 as eval entry. Blocked pairs are not submitted.

"""


def copy_jsonl(src: Path, dest: Path, limit: int = 10) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [line for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    dest.write_text("\n".join(lines[:limit]) + "\n", encoding="utf-8")


def pick_record(records: list[dict]) -> dict:
    def key(rec: dict) -> tuple:
        split = str((rec.get("subset") or {}).get("split") or "").lower()
        mapping = str(rec["framework"]["mapping_status"])
        return (
            0 if mapping == "direct" else 1,
            SPLIT_RANK.get(split, 9),
            rec["record_id"],
        )

    return sorted(records, key=key)[0]


def asr_fields(benchmark: str, *, long_form: bool) -> dict:
    en = benchmark in EN_ASR
    agg = "wer" if en else "wer-zh"
    return {
        "prompt": "qwen3-omni-asr-en" if en else "qwen3-omni-asr-zh",
        "evaluator": "wer" if en else "cer",
        "agg": f"{agg}-windowed" if long_form else agg,
        "status": "ok",
    }


def slot(
    capability: str,
    benchmark: str,
    dataset: str,
    *,
    prompt: str,
    evaluator: str,
    agg: str,
    status: str,
    note: str = "",
    **extra,
) -> dict:
    item = {
        "id": f"{capability}__{benchmark}",
        "capability": capability,
        "benchmark": benchmark,
        "status": status,
        "suite": extra.pop("suite", "main"),
        "dataset": dataset,
        "prompt": prompt,
        "evaluator": evaluator,
        "agg": agg,
    }
    item.update(extra)
    if note:
        item["note"] = note
    return item


def blocked_slot(capability: str, benchmark: str, reason: str) -> dict:
    return {
        "id": f"{capability}__{benchmark}",
        "capability": capability,
        "benchmark": benchmark,
        "status": "blocked",
        "suite": "none",
        "reason": reason,
    }


def native_clip(name: str, f_name: str, protocol: str) -> dict:
    return {
        name: {
            "class": "mesh_eval.dataset.native.NativeClipDataset",
            "args": {
                "f_name": f_name,
                "default_task": name,
                "ref_col": "text",
                "protocol": protocol,
                "benchmark_id": name,
            },
        }
    }


def full_eval(name: str, dataset: str, view: str = "") -> dict:
    args = {"dataset": dataset, "benchmark_id": name}
    if view:
        args["view"] = view
    return {
        name: {
            "class": "mesh_eval.dataset.full_eval.FullEvalDataset",
            "args": args,
        }
    }


def native_key(capability: str, benchmark: str) -> str:
    return f"native-{capability.replace('_', '-')}-{benchmark.replace('_', '-')}"


def require_cap_manifest(capability: str, benchmark: str) -> Path:
    if capability in {"diarization", "speaker_attribution"} and benchmark in WINDOWED_MEETING:
        path = NATIVE / f"{capability}_{benchmark}_windowed.jsonl"
    else:
        path = NATIVE / f"{capability}_{benchmark}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; run scripts/sample_windowed_ondevice_eval.py "
            f"or suites/native/build_native_capability_samples.py"
        )
    return path


def windowed_slots() -> dict:
    slots = {}
    for bench in WINDOWED_MEETING:
        note = f"{WINDOWED_NOTE} {WINDOWED_AUDIO_NOTE[bench]}"
        asr_note = note if bench in EN_ASR else f"{note} {ZH_DIGIT_NOTE}"
        slots[("long_form_asr", bench)] = slot(
            "long_form_asr",
            bench,
            f"native-{bench.replace('_', '')}-long-form-asr",
            **asr_fields(bench, long_form=True),
            limit_step=4,
            note=asr_note,
        )
        slots[("diarization", bench)] = slot(
            "diarization",
            bench,
            native_key("diarization", bench),
            prompt="mesh-diarization",
            evaluator="mesh-diarization",
            agg="mesh",
            status="ok",
            post_process=["mesh-diarization-normalize"],
            limit_step=4,
            note=f"{note} prompt-v2 + normalize; window-DER.",
        )
        slots[("speaker_attribution", bench)] = slot(
            "speaker_attribution",
            bench,
            native_key("speaker_attribution", bench),
            prompt="mesh-speaker-attribution",
            evaluator="mesh-speaker-attribution",
            agg="mesh",
            status="ok",
            limit_step=4,
            note=f"{note} cpCER without timestamps.",
        )
    return slots


def main() -> None:
    NATIVE.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(line) for line in ANN.read_text().splitlines() if line.strip()]
    pairs: dict[tuple[str, str], list[dict]] = {}
    for rec in recs:
        pairs.setdefault((rec["framework"]["capability"], rec["benchmark"]["id"]), []).append(rec)

    new_reg: dict = {}
    # ASR: point at L0 / native-derived asr_*.jsonl (build_asr_native_samples.py).
    # Do not use FullEval, utt_clips indexes, or smoke copies as the eval entry.
    asr_keys = {
        "aishell_1": "native-asr-aishell-1",
        "aishell_5": "native-asr-aishell-5",
        "librispeech": "native-asr-librispeech",
        "magicdata_ramc": "native-asr-magicdata-ramc",
        "sbcsae": "native-asr-sbcsae",
        "common_voice": "native-asr-common-voice",
        "wenetspeech": "native-asr-wenetspeech",
    }
    for bench, key in asr_keys.items():
        manifest = NATIVE / f"asr_{bench}.jsonl"
        if not manifest.is_file():
            raise FileNotFoundError(
                f"missing {manifest}; run suites/native/build_asr_native_samples.py"
            )
        new_reg.update(native_clip(key, f"mesh_eval/data/native/asr_{bench}.jsonl", "asr"))
    for bench in WINDOWED_MEETING:
        new_reg.update(
            native_clip(
                f"native-{bench.replace('_', '')}-long-form-asr",
                f"mesh_eval/data/native/{bench}_long_form_windowed.jsonl",
                "long_form_asr",
            )
        )
    # misp / sbcsae: legacy 30-60 retired; not on windowed_ondevice track (catalog blocked).
    # Capability samples: built by build_native_capability_samples.py / build_asr_native_samples.py
    # Do not copy smoke_/8p31 or FullEval as the eval entry.
    CAP_NATIVE = [
        ("audio_caption", "audiocaps", "audio_caption"),
        ("audio_caption", "clotho", "audio_caption"),
        ("qa", "spoken_squad", "qa"),
        ("qa", "uro_bench", "qa"),
        ("qa", "vocalbench", "qa"),
        ("qa", "vocalbench_zh", "qa"),
        ("qa", "ear_wdyl", "qa"),
        ("qa", "voicebench", "qa"),
        ("audio_reasoning", "mmsu", "audio_reasoning"),
        ("audio_reasoning", "uro_bench", "audio_reasoning"),
        ("audio_reasoning", "vocalbench", "audio_reasoning"),
        ("audio_reasoning", "vocalbench_zh", "audio_reasoning"),
        ("audio_reasoning", "voicebench", "audio_reasoning"),
        ("sound_event", "esc50", "sound_event"),
        ("sound_event", "mmsu", "sound_event"),
        ("sound_event", "mavd_traffic", "sound_event_multilabel"),
        ("sound_event", "sonyc_ust_v23", "sound_event_multilabel"),
        ("sound_event", "tut_sound_events_2017", "sound_event"),
        ("acoustic_scene", "dcase2013_scenes", "acoustic_scene"),
        ("acoustic_scene", "mmsu", "acoustic_scene"),
        ("acoustic_scene", "tut_sound_events_2017", "acoustic_scene"),
        ("acoustic_scene", "tut_urban_acoustic_scenes_2018_mobile", "acoustic_scene"),
        ("paralinguistic_recognition", "mmsu", "paralinguistic_recognition"),
        ("speaker_count", "mmsu", "speaker_count"),
        ("speaker_verification", "voxceleb1", "speaker_verification"),
        ("speech_translation", "covost2", "speech_translation"),
        ("speech_translation", "mmsu", "speech_translation"),
        ("spoof_detection", "mmsu", "spoof_detection"),
        ("speech_generation", "libritts", "speech_generation"),
        ("code_switch", "mmsu", "code_switch"),
        ("code_switch", "vocalbench_zh", "code_switch"),
        ("code_switch", "vocalbench_zh_open_ended", "code_switch"),
        ("diarization", "aishell_4", "diarization"),
        ("diarization", "alimeeting", "diarization"),
        ("diarization", "ami", "diarization"),
        ("diarization", "chime_6", "diarization"),
        ("speaker_attribution", "aishell_4", "speaker_attribution"),
        ("speaker_attribution", "alimeeting", "speaker_attribution"),
        ("speaker_attribution", "ami", "speaker_attribution"),
        ("speaker_attribution", "chime_6", "speaker_attribution"),
        ("target_speaker", "aishell_5", "target_speaker"),
    ]
    for cap, bench, protocol in CAP_NATIVE:
        path = require_cap_manifest(cap, bench)
        new_reg.update(
            native_clip(
                native_key(cap, bench),
                f"mesh_eval/data/native/{path.name}",
                protocol,
            )
        )


    existing = yaml.safe_load(REG.read_text()) or {}
    for key, value in new_reg.items():
        existing[key] = value
    # Drop deprecated ASR loaders from the active registry surface.
    for stale in list(existing):
        if stale.startswith("full-") and (
            stale.endswith("-asr")
            or stale.endswith("-caption")
            or stale.endswith("-qa")
            or stale.endswith("-reasoning")
            or "mmsu" in stale
            or "voxceleb" in stale
            or "libritts" in stale
            or "covost" in stale
            or "esc50" in stale
            or "spoken-squad" in stale
            or "uro-bench" in stale
            or "vocalbench" in stale
        ):
            existing.pop(stale, None)
    for stale in (
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
        "native-aishell5-utt-asr",
        "native-magicdata-ramc-utt-asr",
        "native-wenetspeech-utt-asr",
        "native-common-voice-en-asr",
        "native-sbcsae-utt-asr",
        "native-aishell4-utt-asr",
        "native-alimeeting-utt-asr",
        "native-misp-utt-asr",
        "native-ami-utt-asr",
        "native-chime6-utt-asr",
        "native-aishell4-diarization",
        "native-alimeeting-diarization",
        "native-ami-diarization",
        "native-chime6-diarization",
        "native-aishell4-attribution",
        "native-aishell5-attribution",
        "native-ami-attribution",
        "native-alimeeting-attribution",
        "native-aishell5-target-speaker",
        "native-dcase2013-acoustic-scene",
        "native-tut2017-acoustic-scene",
        "native-tut2018-acoustic-scene",
        "native-mavd-sound-event",
        "native-sonyc-sound-event",
        "native-tut2017-sound-event",
        "native-mmsu-code-switch",
        "native-vocalbench-zh-code-switch",
        "native-ear-wdyl-qa",
        "native-voicebench-qa",
        "native-voicebench-audio-reasoning",
        "native-speaker-attribution-aishell-5",
    ):
        existing.pop(stale, None)
    REG.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8")

    loaders = {
        ("asr", "aishell_1"): slot(
            "asr",
            "aishell_1",
            "native-asr-aishell-1",
            **asr_fields("aishell_1", long_form=False),
            note="L0 aishell1_test_asr.jsonl short wavs.",
        ),
        ("asr", "aishell_5"): slot(
            "asr",
            "aishell_5",
            "native-asr-aishell-5",
            **{
                **asr_fields("aishell_5", long_form=False),
                "status": "diagnostic",
            },
            note="L0 AISHELL-5_eval1_msswift + timestamp crop (not utt_clips).",
        ),
        ("asr", "common_voice"): slot(
            "asr",
            "common_voice",
            "native-asr-common-voice",
            **asr_fields("common_voice", long_form=False),
            note="L0 CommonVoice_26 English test.tsv clips.",
        ),
        ("asr", "librispeech"): slot(
            "asr",
            "librispeech",
            "native-asr-librispeech",
            **asr_fields("librispeech", long_form=False),
            note="L0 LibriSpeech test-clean flac.",
        ),
        ("asr", "magicdata_ramc"): slot(
            "asr",
            "magicdata_ramc",
            "native-asr-magicdata-ramc",
            **asr_fields("magicdata_ramc", long_form=False),
            note="L0 MagicHub WAV+TXT timestamp crop (not utt_clips).",
        ),
        ("asr", "sbcsae"): slot(
            "asr",
            "sbcsae",
            "native-asr-sbcsae",
            **asr_fields("sbcsae", long_form=False),
            note="L0 SBCSAE asr.jsonl kitchen_dining timestamp crop.",
        ),
        ("asr", "wenetspeech"): slot(
            "asr",
            "wenetspeech",
            "native-asr-wenetspeech",
            **asr_fields("wenetspeech", long_form=False),
            note="TEST_NET decoded segment wavs under asr_eval_clips (aes L0 not unpacked).",
        ),
        ("long_form_asr", "misp"): blocked_slot(
            "long_form_asr",
            "misp",
            "Out of windowed_ondevice track (legacy 30-60 retired); not rebuilt.",
        ),
        ("long_form_asr", "sbcsae"): blocked_slot(
            "long_form_asr",
            "sbcsae",
            "Out of windowed_ondevice track (legacy 30-60 retired); not rebuilt.",
        ),
        ("speech_translation", "covost2"): slot("speech_translation", "covost2", "native-speech-translation-covost2", prompt="qwen3-omni-s2tt-zh2en", evaluator="bleu", agg="bleu", status="ok", note="CoVoST2 zh→en. Dataset covered; en→zh stays registered."),
        ("speech_translation", "mmsu"): slot("speech_translation", "mmsu", "native-speech-translation-mmsu", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic"),
        ("code_switch", "vocalbench_zh"): slot("code_switch", "vocalbench_zh", "native-code-switch-vocalbench-zh", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic", note="knowledge short-QA (qa-exist-match). LLM-Judge (binary) as semantic auxiliary. No MER."),
        ("code_switch", "vocalbench_zh_open_ended"): slot(
            "code_switch",
            "vocalbench_zh_open_ended",
            "native-code-switch-vocalbench-zh-open-ended",
            prompt="mesh-code-switch-open-ended",
            evaluator="mesh-rouge-l",
            agg="mean",
            status="diagnostic",
            note="open_ended long-form QA (ROUGE-L). LLM-Judge (open) as semantic auxiliary. No MER.",
            subset_record_id="vocalbench_zh__test__code_switch_open_ended__code_switch__code_switch__unknown__unknown__3983314d55",
        ),
        ("code_switch", "mmsu"): slot(
            "code_switch",
            "mmsu",
            "native-code-switch-mmsu",
            prompt="mesh-external-qa",
            evaluator="mesh-rouge-l",
            agg="mean",
            status="diagnostic",
            note="Open-answer code-switch QA (ROUGE-L); MCQ options stripped. LLM-Judge (semi-open) as semantic auxiliary.",
        ),
        ("qa", "spoken_squad"): slot("qa", "spoken_squad", "native-qa-spoken-squad", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="ok"),
        ("qa", "uro_bench"): slot("qa", "uro_bench", "native-qa-uro-bench", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic"),
        ("qa", "vocalbench"): slot("qa", "vocalbench", "native-qa-vocalbench", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic"),
        ("qa", "vocalbench_zh"): slot("qa", "vocalbench_zh", "native-qa-vocalbench-zh", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic"),
        ("qa", "ear_wdyl"): slot("qa", "ear_wdyl", "native-qa-ear-wdyl", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic", note="L0 EAR wdyl QA."),
        ("qa", "voicebench"): slot("qa", "voicebench", "native-qa-voicebench", prompt="mesh-external-qa", evaluator="qa-exist-match", agg="acc", status="diagnostic", note="Open subsets often lack gold; judge pending."),
        ("audio_reasoning", "mmsu"): blocked_slot(
            "audio_reasoning",
            "mmsu",
            "Suspended 2026-09-26: audio_reasoning not evaluated for now.",
        ),
        ("audio_reasoning", "uro_bench"): blocked_slot(
            "audio_reasoning",
            "uro_bench",
            "Suspended 2026-09-26: audio_reasoning not evaluated for now.",
        ),
        ("audio_reasoning", "vocalbench"): blocked_slot(
            "audio_reasoning",
            "vocalbench",
            "Suspended 2026-09-26: audio_reasoning not evaluated for now.",
        ),
        ("audio_reasoning", "vocalbench_zh"): blocked_slot(
            "audio_reasoning",
            "vocalbench_zh",
            "Suspended 2026-09-26: audio_reasoning not evaluated for now.",
        ),
        ("audio_reasoning", "voicebench"): blocked_slot(
            "audio_reasoning",
            "voicebench",
            "Suspended 2026-09-26: audio_reasoning not evaluated for now.",
        ),
        ("audio_caption", "audiocaps"): slot("audio_caption", "audiocaps", "native-audio-caption-audiocaps", prompt="qwen3-omni-audio-caption", evaluator="coco", agg="coco", status="ok"),
        ("audio_caption", "clotho"): slot("audio_caption", "clotho", "native-audio-caption-clotho", prompt="qwen3-omni-audio-caption", evaluator="coco", agg="coco", status="ok"),
        ("sound_event", "esc50"): slot("sound_event", "esc50", "native-sound-event-esc50", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok", note="Fold-5 closed-set; no gold leak."),
        ("sound_event", "mmsu"): slot("sound_event", "mmsu", "native-sound-event-mmsu", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok", note="Per-item 4 event options; return label."),
        ("sound_event", "mavd_traffic"): slot("sound_event", "mavd_traffic", "native-sound-event-mavd-traffic", prompt="mesh-classification", evaluator="mesh-multilabel-f1", agg="mesh", status="diagnostic", note="Event-window multilabel F1."),
        ("sound_event", "sonyc_ust_v23"): slot("sound_event", "sonyc_ust_v23", "native-sound-event-sonyc-ust-v23", prompt="mesh-classification", evaluator="mesh-multilabel-f1", agg="mesh", status="diagnostic", note="Multilabel tagging F1."),
        ("sound_event", "tut_sound_events_2017"): slot("sound_event", "tut_sound_events_2017", "native-sound-event-tut-sound-events-2017", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok", note="Evaluation event clips + 6-class closed-set."),
        ("acoustic_scene", "dcase2013_scenes"): slot("acoustic_scene", "dcase2013_scenes", "native-acoustic-scene-dcase2013-scenes", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok"),
        ("acoustic_scene", "mmsu"): slot("acoustic_scene", "mmsu", "native-acoustic-scene-mmsu", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok"),
        ("acoustic_scene", "tut_sound_events_2017"): slot("acoustic_scene", "tut_sound_events_2017", "native-acoustic-scene-tut-sound-events-2017", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="diagnostic", note="L0 TUT acoustic scene."),
        ("acoustic_scene", "tut_urban_acoustic_scenes_2018_mobile"): slot("acoustic_scene", "tut_urban_acoustic_scenes_2018_mobile", "native-acoustic-scene-tut-urban-acoustic-scenes-2018-mobile", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok"),
        ("paralinguistic_recognition", "mmsu"): slot("paralinguistic_recognition", "mmsu", "native-paralinguistic-recognition-mmsu", prompt="mesh-classification", evaluator="mesh-classification", agg="mesh", status="ok", note="Per-item 4 attribute options; return label; stratified Native."),
        ("speaker_count", "mmsu"): slot("speaker_count", "mmsu", "native-speaker-count-mmsu", prompt="mesh-speaker-count", evaluator="mesh-speaker-count", agg="mesh", status="ok", note="Open integer count; MCQ options stripped."),
        ("speaker_verification", "voxceleb1"): slot("speaker_verification", "voxceleb1", "native-speaker-verification-voxceleb1", prompt="mesh-speaker-verification", evaluator="mesh-speaker-verification", agg="mesh", status="ok"),
        ("target_speaker", "aishell_5"): slot("target_speaker", "aishell_5", "native-target-speaker-aishell-5", prompt="mesh-target-speaker", evaluator="mesh-target-speaker", agg="mesh", status="diagnostic", note="AISHELL-5 Eval1+Eval2 adapted pairs: near-field enrollment 5-10s + far-field DX01C01 command from another session; negatives = the other speaker in the same car. Full 1030 pairs (scripts/build_full_target_speaker_indexes.py); Native 10 = 5 pos + 5 neg over 10 speakers."),
        ("spoof_detection", "mmsu"): blocked_slot(
            "spoof_detection",
            "mmsu",
            "Suspended 2026-09-26: spoof_detection not evaluated for now.",
        ),
        ("instruction_following", "vocalbench"): slot(
            "instruction_following",
            "vocalbench",
            "t4-vocalbench-instruction-following",
            prompt="mesh-agent-rubric",
            evaluator="mesh-agent-rubric-audio",
            agg="mesh",
            status="diagnostic",
            suite="tts",
            model="qwen3-omni-speech-local",
            post_process=["extract_audio"],
            two_phase=True,
            note="Official VocalBench FR semantic (Correct/Incorrect criteria); Thinking judge; acoustic gates TBD; not merged with VoiceBench.",
        ),
        ("instruction_following", "vocalbench_zh"): slot(
            "instruction_following",
            "vocalbench_zh",
            "t4-vocalbench-zh-instruction-following",
            prompt="mesh-agent-rubric",
            evaluator="mesh-agent-rubric-audio",
            agg="mesh",
            status="diagnostic",
            suite="tts",
            model="qwen3-omni-speech-local",
            post_process=["extract_audio"],
            two_phase=True,
            note="Official VocalBench-zh FR semantic (正确/错误); Thinking judge; acoustic gates TBD; not merged with VoiceBench.",
        ),
        ("instruction_following", "voicebench"): slot(
            "instruction_following",
            "voicebench",
            "t4-voicebench-ifeval",
            prompt="mesh-agent-rubric",
            evaluator="mesh-ifeval",
            agg="mesh",
            status="ok",
            note="Text IFEval; not merged with VocalBench.",
        ),
        ("tool_call", "fluent_speech_commands"): blocked_slot(
            "tool_call",
            "fluent_speech_commands",
            "Suspended 2026-09-26: tool_call not evaluated for now.",
        ),
        ("tool_call", "slurp"): blocked_slot(
            "tool_call",
            "slurp",
            "Suspended 2026-09-26: tool_call not evaluated for now.",
        ),
        ("tool_call", "audioagentbench_suite"): blocked_slot(
            "tool_call",
            "audioagentbench_suite",
            "Suspended 2026-09-26: tool_call not evaluated for now.",
        ),
        ("tool_call", "stepeval_audio_toolcall"): blocked_slot(
            "tool_call",
            "stepeval_audio_toolcall",
            "Suspended 2026-09-26: tool_call not evaluated for now.",
        ),
        # Multi-turn dialogue: mesh-contextual-audio keeps messages/multi_turn history;
        # judges follow each benchmark's official protocol (mesh_eval/evaluator/dialogue.py)
        # with the local Thinking judge, run after SUT release (two_phase).
        ("multi_turn_dialogue", "vocalbench_zh"): slot(
            "multi_turn_dialogue",
            "vocalbench_zh",
            "t4-vocalbench-zh-multi-turn",
            prompt="mesh-contextual-audio",
            evaluator="mesh-dialogue-vocalbench",
            agg="mesh",
            status="ok",
            two_phase=True,
            note="Official utils/multi_round.py judge (正确/错误) -> multi_round_acc.",
        ),
        ("multi_turn_dialogue", "vocalbench"): slot(
            "multi_turn_dialogue",
            "vocalbench",
            "t4-vocalbench-multi-turn",
            prompt="mesh-contextual-audio",
            evaluator="mesh-dialogue-vocalbench",
            agg="mesh",
            status="ok",
            two_phase=True,
            note="Official utils/multi_round.py judge (Correct/Wrong) -> multi_round_acc.",
        ),
        ("multi_turn_dialogue", "audioagentbench_suite"): slot(
            "multi_turn_dialogue",
            "audioagentbench_suite",
            "audio-agent-bench-dialogue",
            prompt="mesh-contextual-audio",
            evaluator="mesh-dialogue-aab",
            agg="mesh",
            status="ok",
            two_phase=True,
            note="audio-arena-bench rehydrated mode (golden text history, target audio); official judge prompt -> turn_pass + dimension pass rates; tool turns excluded.",
        ),
        ("multi_turn_dialogue", "mtalk_bench"): slot(
            "multi_turn_dialogue",
            "mtalk_bench",
            "t4-mtalk-bench",
            prompt="mesh-contextual-audio",
            evaluator="mesh-dialogue-mtalk-rubric",
            agg="mesh",
            status="ok",
            suite="tts",
            model="qwen3-omni-speech-local",
            two_phase=True,
            note="Official rubric protocol (general + specific 0/1 criteria on concatenated dialogue audio) -> rubric_score; arena/ELO needs model pairs, not computed.",
        ),
        ("multi_turn_dialogue", "voicebench"): slot(
            "multi_turn_dialogue",
            "voicebench",
            "t4-voicebench-mtbench",
            prompt="mesh-contextual-audio",
            evaluator="mesh-dialogue-mtbench",
            agg="mesh",
            status="ok",
            two_phase=True,
            note="FastChat MT-Bench single-answer grading (1-10, per turn) -> mtbench_score.",
        ),
        ("clarification", "ear_wdyl"): blocked_slot(
            "clarification",
            "ear_wdyl",
            "Suspended 2026-09-26: clarification not evaluated for now.",
        ),
        ("clarification", "audioagentbench_suite"): blocked_slot(
            "clarification",
            "audioagentbench_suite",
            "Suspended 2026-09-26: clarification not evaluated for now.",
        ),
        ("interruption", "ihbench"): blocked_slot(
            "interruption",
            "ihbench",
            "Suspended 2026-09-26: interruption not evaluated for now.",
        ),
        ("speech_generation", "libritts"): slot(
            "speech_generation",
            "libritts",
            "native-speech-generation-libritts",
            prompt="qwen3-omni-tts-en",
            evaluator="seed-tts-eval-asr-wer-en",
            agg="mesh",
            status="ok",
            suite="tts",
            post_process=["extract_audio"],
            two_phase=True,
            model="qwen3-omni-speech-local",
        ),
    }
    loaders.update(windowed_slots())

    capabilities: list[dict] = []
    missing_map: list[tuple[str, str]] = []
    for (cap, bench), records in sorted(pairs.items()):
        chosen = pick_record(records)
        if (cap, bench) in EXCLUDE:
            item = blocked_slot(cap, bench, EXCLUDE[(cap, bench)])
            item["subset_record_id"] = chosen["record_id"]
            capabilities.append(item)
            continue
        exe = any(
            rec["framework"]["mapping_status"] in {"direct", "adaptable"}
            and rec["benchmark"]["resource_status"] == "available"
            for rec in records
        )
        if not exe:
            capabilities.append(
                blocked_slot(
                    cap,
                    bench,
                    f"mapping={chosen['framework']['mapping_status']} resource={chosen['benchmark']['resource_status']}",
                )
            )
            continue
        item = loaders.get((cap, bench))
        if item is None:
            missing_map.append((cap, bench))
            continue
        item = dict(item)
        item["subset_record_id"] = chosen["record_id"]
        capabilities.append(item)

    if missing_map:
        raise SystemExit(f"missing loader map for {missing_map}")

    capabilities.extend(
        [
            {
                "id": "spoken_slu__fluent_speech_commands",
                "capability": "spoken_slu",
                "benchmark": "fluent_speech_commands",
                "status": "diagnostic",
                "suite": "main",
                "dataset": "t4-fsc-instruction-following",
                "prompt": "mesh-intent-slots",
                "evaluator": "mesh-intent",
                "agg": "mesh",
                "note": (
                    "Spoken SLU (intent_acc + slot_f1 via mesh-slot-f1 aux). "
                    "Independent of IF and tool_call."
                ),
            },
            {
                "id": "spoken_slu__slurp",
                "capability": "spoken_slu",
                "benchmark": "slurp",
                "status": "diagnostic",
                "suite": "main",
                "dataset": "t4-slurp-instruction-following",
                "prompt": "mesh-intent-slots",
                "evaluator": "mesh-intent",
                "agg": "mesh",
                "note": (
                    "Spoken SLU (intent_acc + slot_f1 via mesh-slot-f1 aux). "
                    "Independent of IF and tool_call."
                ),
            },
            {
                "id": "meeting_summary",
                "capability": "meeting_summary",
                "status": "blocked",
                "suite": "none",
                "reason": "Annotation baseline has zero meeting_summary records. URO LCSTS/Summary are news, not meetings.",
            },
            {
                "id": "todo_extraction",
                "capability": "todo_extraction",
                "status": "blocked",
                "suite": "none",
                "reason": "Annotation baseline has zero todo_extraction records.",
            },
        ]
    )

    catalog = {
        "schema_version": "native-catalog/1.1",
        "limit": 10,
        "model": "qwen3-omni-local",
        "speech_model": "qwen3-omni-speech-local",
        "annotation_baseline": "annotation/benchmark_subset_manifest.jsonl",
        "capabilities": capabilities,
    }
    CAT.write_text(HEAD + yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True), encoding="utf-8")
    runnable = [item for item in capabilities if item.get("suite") not in {None, "none"}]
    blocked = [item for item in capabilities if item.get("status") == "blocked"]
    print(f"wrote {CAT} runnable={len(runnable)} blocked={len(blocked)}")
    print(f"updated {REG} with {len(new_reg)} candidate keys")


if __name__ == "__main__":
    main()
