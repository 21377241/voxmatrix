#!/usr/bin/env python3
"""Build full audio_reasoning indexes aligned with Native catalog slots.

Tracks (gold + readable WavPath; open QA where source is MCQ):
  - MMSU reasoning_subtasks (exclude non-reasoning prefixes; strip A/B/C/D)
  - URO MuChoEval-en (strip choice block)
  - VocalBench reasoning
  - VocalBench-zh reasoning
  - VoiceBench BBH (keep Options in prompt; short letter/yes-no gold)
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/audio_reasoning"

MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_JSONL = MMSU_ROOT / "mmsu_audioqa.jsonl"
URO_MUCHO = Path(
    "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/URO-Bench/pro/MuChoEval-en/test.jsonl"
)
VB_ROOT = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench")
VBZH_ROOT = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh")
VB_PARQUET = Path(
    "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench/bbh"
)
VB_WAV = Path(
    "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench_wav/wav/bbh"
)

# Same exclude set as Native audio_reasoning MMSU builder.
MMSU_REASONING_EXCLUDE = {
    "MMSU_accent_identification_",
    "MMSU_age_prediction_",
    "MMSU_background_scene_recognition_",
    "MMSU_code_switch_question_answering_",
    "MMSU_disfluency_detection_",
    "MMSU_emotion_recognition_",
    "MMSU_gender_prediction_",
    "MMSU_intonation_perception_",
    "MMSU_non_verbal_sound_detection_",
    "MMSU_pause_perception_",
    "MMSU_pitch_comparison_",
    "MMSU_prolonged_sound_perception_",
    "MMSU_sarcasm_detection_",
    "MMSU_speech_stress_perception_",
    "MMSU_speech_translation_",
    "MMSU_speed_comparison_",
    "MMSU_synthetic_speech_detection_",
    "MMSU_total_speaker_counting_",
    "MMSU_vocal_range_comparison_",
    "MMSU_volume_comparison_",
}


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    from mesh_eval.data.full_index_meta import slot_id_from_filename, write_index_jsonl

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    slot_id = slot_id_from_filename(OUT.name, name)
    n = write_index_jsonl(path, rows, slot_id)
    return n, path


def _strip_mcq_stem(question: str) -> str:
    text = str(question or "").strip()
    text = re.split(
        r"(?i)\s*Please select from the choices below\s*:", text, maxsplit=1
    )[0].strip()
    text = re.split(r"\nA\.\s*", text, maxsplit=1)[0].strip()
    text = re.split(r"\s+A\.\s+", text, maxsplit=1)[0].strip()
    return text


def _open_qa_instruction(stem: str, *, zh: bool = False) -> str:
    """文本题干 + 作答指令（用于题干不在音频里的子集，如 MuChoEval / MMSU）。"""
    stem = stem.strip()
    if zh:
        return f"{stem}\n\n根据音频作答，只输出答案。"
    return (
        f"{stem}\n\nAnswer the question using the audio. Output only the answer."
    )


# VocalBench reasoning：题干已在 wav 朗读，{{question}} 只放短指令
SPOKEN_AR_PROMPT_EN = (
    "Listen to the spoken question in the audio and reason to answer. "
    "Output only the answer."
)
SPOKEN_AR_PROMPT_ZH = "请听音频中的口语问题，推理后作答，只输出答案。"


def _parse_mmsu_messages(doc: dict) -> tuple[str | None, str, str]:
    audio = None
    question = ""
    answer = ""
    for msg in doc.get("messages") or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("audio_path"):
                    audio = part["audio_path"]
                if part.get("text"):
                    if role == "user":
                        question = str(part["text"]).strip()
                    elif role == "assistant":
                        answer = str(part["text"]).strip()
        elif isinstance(content, str):
            if role == "user" and "<audio>" in content:
                question = content.replace("<audio>", "").strip()
            elif role == "assistant":
                answer = content.strip()
    if not audio:
        audios = doc.get("audios") or []
        if audios:
            audio = audios[0]
    return audio, question, answer


def _answer_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return parts[0] if parts else ""
    return str(value).strip()


def build_mmsu() -> tuple[int, Path]:
    rows = []
    seen: set[str] = set()
    for line in MMSU_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        if not sid or sid in seen:
            continue
        if any(sid.startswith(p) for p in MMSU_REASONING_EXCLUDE):
            continue
        rel, question, answer = _parse_mmsu_messages(doc)
        if not rel or not answer:
            continue
        wav = MMSU_ROOT / rel
        if not wav.is_file():
            alt = MMSU_ROOT / "audio" / Path(rel).name
            wav = alt if alt.is_file() else wav
        if not wav.is_file():
            continue
        opts = re.findall(r"^([A-D])\.\s*(.+)$", question, flags=re.M)
        choice_map = {k: v.strip() for k, v in opts}
        if answer in choice_map:
            answer = choice_map[answer]
        elif answer.upper() in choice_map:
            answer = choice_map[answer.upper()]
        stem = _strip_mcq_stem(question)
        if not stem:
            continue
        seen.add(sid)
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "mmsu",
                "benchmark": "mmsu",
                "capability": "audio_reasoning",
                "protocol": "open_qa",
                "subset": "reasoning_subtasks",
                "split": "test",
                "WavPath": str(wav),
                "question": _open_qa_instruction(stem),
                "answer": answer,
                "text": answer,
                "source_index": str(MMSU_JSONL),
            }
        )
    return _write(rows, "audio_reasoning_mmsu.jsonl")


def build_uro() -> tuple[int, Path]:
    rows = []
    root = URO_MUCHO.parent
    seen: set[str] = set()
    for line in URO_MUCHO.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        answer = _answer_text(doc.get("target_text") or doc.get("answer"))
        question = str(doc.get("source_text") or doc.get("question") or "").strip()
        audio = doc.get("source_wav") or doc.get("audio") or doc.get("WavPath")
        if not answer or not question or not audio:
            continue
        wav = Path(audio) if Path(str(audio)).is_absolute() else (root / audio)
        if not wav.is_file():
            continue
        sid = str(doc.get("id") or f"uro_{len(rows):04d}")
        if sid in seen:
            continue
        seen.add(sid)
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "uro_bench",
                "benchmark": "uro_bench",
                "capability": "audio_reasoning",
                "protocol": "open_qa",
                "subset": "multimodal_reasoning",
                "split": "test",
                "WavPath": str(wav),
                "question": _open_qa_instruction(_strip_mcq_stem(question)),
                "answer": answer,
                "text": answer,
                "source_index": str(URO_MUCHO),
            }
        )
    return _write(rows, "audio_reasoning_uro_bench.jsonl")


def build_vocalbench(
    root: Path,
    json_name: str,
    dataset: str,
    output_name: str,
    *,
    zh: bool = False,
) -> tuple[int, Path]:
    source = root / "json" / json_name
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    seen: set[str] = set()
    for item in data:
        rel = str(item.get("Audio") or item.get("audio") or "").strip()
        candidates = [root / "audio" / rel, root / rel]
        wav = next((p for p in candidates if p.is_file()), None)
        answer = _answer_text(
            item.get("Answer")
            or item.get("Answer_zh")
            or item.get("Answer_en")
            or item.get("answer")
        )
        question = str(
            item.get("Question")
            or item.get("Question_zh")
            or item.get("question")
            or ""
        ).strip()
        sid = str(item.get("Qid") or item.get("id") or "").strip()
        if not sid or sid in seen or wav is None or not answer or not question:
            continue
        seen.add(sid)
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": dataset,
                "benchmark": dataset,
                "capability": "audio_reasoning",
                "protocol": "open_qa",
                "subset": "reasoning",
                "split": "test",
                "WavPath": str(wav),
                # 口语题干在音频中：勿把 stem 再写入 mesh-external-qa 的 {{question}}
                "question": SPOKEN_AR_PROMPT_ZH if zh else SPOKEN_AR_PROMPT_EN,
                "spoken_question_text": question,
                "answer": answer,
                "text": answer,
                "source_index": str(source),
            }
        )
    return _write(rows, output_name)


# VoiceBench BBH：完整题干+格式要求已在 wav 朗读；{{question}} 勿再塞题干。
# 音频要求最终答案带前缀 "The answer is:" —— gold(ref) 必须同形，不能只写 (A)/Yes。
SPOKEN_BBH_PROMPT = (
    "Listen to the spoken question and instructions in the audio. "
    "Follow the answer format required in the audio."
)


def _bbh_formatted_ref(answer: str) -> str:
    """把短标签改成与音频格式要求一致的 gold。"""
    a = str(answer or "").strip()
    if not a:
        return a
    low = a.lower()
    if low.startswith("the answer is:"):
        rest = a.split(":", 1)[1].strip()
        return f"The answer is: {rest}"
    return f"The answer is: {a}"


def build_voicebench_bbh() -> tuple[int, Path]:
    import pyarrow.parquet as pq

    wav_by_stem = {p.stem: p for p in VB_WAV.rglob("*.wav")}
    rows = []
    seen: set[str] = set()
    for path in sorted(VB_PARQUET.glob("*.parquet")):
        table = pq.read_table(path)
        for i in range(table.num_rows):
            item = {col: table.column(col)[i].as_py() for col in table.column_names}
            sid = str(item.get("id") or "").strip()
            answer = _answer_text(item.get("reference") or item.get("answer"))
            question = str(item.get("prompt") or item.get("question") or "").strip()
            if not sid or sid in seen or not answer or not question:
                continue
            wav = wav_by_stem.get(sid)
            if wav is None or not wav.is_file():
                continue
            seen.add(sid)
            ref = _bbh_formatted_ref(answer)
            rows.append(
                {
                    "schema_version": "full-index/1.0",
                    "sample_id": sid,
                    "dataset": "voicebench",
                    "benchmark": "voicebench",
                    "capability": "audio_reasoning",
                    "protocol": "open_qa",
                    "subset": "bbh",
                    "split": "test",
                    "WavPath": str(wav),
                    "question": SPOKEN_BBH_PROMPT,
                    # 与 wav 一致：完整题干 + Options + 格式套话
                    "spoken_question_text": question,
                    "answer": ref,
                    "text": ref,
                    "source_index": str(path),
                }
            )
    return _write(rows, "audio_reasoning_voicebench.jsonl")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    n, path = build_mmsu()
    summary["mmsu"] = {"records": n, "output": str(path)}
    n, path = build_uro()
    summary["uro_bench"] = {"records": n, "output": str(path)}
    n, path = build_vocalbench(
        VB_ROOT, "reasoning.json", "vocalbench", "audio_reasoning_vocalbench.jsonl"
    )
    summary["vocalbench"] = {"records": n, "output": str(path)}
    n, path = build_vocalbench(
        VBZH_ROOT,
        "reasoning.json",
        "vocalbench_zh",
        "audio_reasoning_vocalbench_zh.jsonl",
        zh=True,
    )
    summary["vocalbench_zh"] = {"records": n, "output": str(path)}
    n, path = build_voicebench_bbh()
    summary["voicebench"] = {"records": n, "output": str(path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    missing = [k for k, v in summary.items() if v["records"] <= 0]
    if missing:
        raise SystemExit(f"empty full indexes: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
