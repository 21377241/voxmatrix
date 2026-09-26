#!/usr/bin/env python3
"""Build per-(capability, benchmark) native eval samples from L0 (not FullEval/smoke).

Writes mesh_eval/data/native/<capability>_<benchmark>.jsonl (limit=10).
Long/windowed units land under mesh_eval/data/native/eval_clips/.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
CLIPS = NATIVE / "eval_clips"
ANN = Path(
    "/mnt/afs/users/wangyl/benchmark_annotation_audit/native_v2_pipeline/outputs/benchmark_subset_manifest.jsonl"
)
FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"
LIMIT = 10
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_JSONL = MMSU_ROOT / "mmsu_audioqa.jsonl"

# Existing smoke window packs: physical clips only; rewrite into eval_clips/.
SMOKE_DIAR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_diar_all")
SMOKE_ATTR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_attr_all")
SMOKE_TS = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_ts_all")
P8 = Path("/mnt/afs/users/wangyl/8p31/runs")

MMSU_PREFIX = {
    "acoustic_scene": ["MMSU_background_scene_recognition_"],
    "sound_event": ["MMSU_non_verbal_sound_detection_"],
    "code_switch": ["MMSU_code_switch_question_answering_"],
    "speech_translation": ["MMSU_speech_translation_"],
    "spoof_detection": ["MMSU_synthetic_speech_detection_"],
    "speaker_count": ["MMSU_total_speaker_counting_"],
    "paralinguistic_recognition": [
        "MMSU_accent_identification_",
        "MMSU_age_prediction_",
        "MMSU_disfluency_detection_",
        "MMSU_emotion_recognition_",
        "MMSU_gender_prediction_",
        "MMSU_intonation_perception_",
        "MMSU_pause_perception_",
        "MMSU_pitch_comparison_",
        "MMSU_prolonged_sound_perception_",
        "MMSU_sarcasm_detection_",
        "MMSU_speech_stress_perception_",
        "MMSU_speed_comparison_",
        "MMSU_vocal_range_comparison_",
        "MMSU_volume_comparison_",
    ],
}
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def out_name(capability: str, benchmark: str) -> Path:
    return NATIVE / f"{capability}_{benchmark}.jsonl"


def parse_messages(doc: dict) -> tuple[str | None, str, str]:
    """Return (rel_audio, question_or_empty, answer_text)."""
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


def iter_mmsu(prefixes: list[str] | None = None, exclude: set[str] | None = None):
    with MMSU_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            sid = str(doc.get("id") or "")
            if prefixes and not any(sid.startswith(p) for p in prefixes):
                continue
            if exclude and any(sid.startswith(p) for p in exclude):
                continue
            rel, question, answer = parse_messages(doc)
            if not rel or not answer:
                continue
            wav = MMSU_ROOT / rel
            if not wav.is_file():
                # try audio/ basename fallback
                alt = MMSU_ROOT / "audio" / Path(rel).name
                wav = alt if alt.is_file() else wav
            if not wav.is_file():
                continue
            yield {
                "sample_id": sid,
                "dataset": "mmsu",
                "WavPath": str(wav),
                "question": question,
                "answer": answer,
                "text": answer,
                "source_index": str(MMSU_JSONL),
            }


def take(gen, limit: int = LIMIT) -> list[dict]:
    rows = []
    for row in gen:
        rows.append(row)
        if len(rows) >= limit:
            break
    return rows


def _mmsu_task_key(sample_id: str) -> str:
    """MMSU_accent_identification_<uuid> -> accent_identification."""
    sid = str(sample_id or "")
    if not sid.startswith("MMSU_"):
        return "unknown"
    rest = sid[len("MMSU_") :]
    # uuid segment starts at first hyphen-containing token
    parts = rest.split("_")
    for i, part in enumerate(parts):
        if "-" in part:
            return "_".join(parts[:i]) if i else "unknown"
    return rest.rsplit("_", 1)[0] if "_" in rest else rest


def build_paralinguistic_mmsu() -> list[dict]:
    """Closed-label PR: round-robin across MMSU paralinguistic subtasks."""
    prefixes = MMSU_PREFIX["paralinguistic_recognition"]
    by_task: dict[str, list[dict]] = {}
    for row in iter_mmsu(prefixes=prefixes):
        task = _mmsu_task_key(row["sample_id"])
        by_task.setdefault(task, []).append(row)
    # round-robin for subtype diversity
    rows: list[dict] = []
    tasks = sorted(by_task)
    idx = 0
    while len(rows) < LIMIT and tasks:
        task = tasks[idx % len(tasks)]
        bucket = by_task[task]
        if bucket:
            rows.append(bucket.pop(0))
        if not bucket:
            tasks = [t for t in tasks if by_task[t]]
            if not tasks:
                break
            idx = 0
            continue
        idx += 1
    # apply closed-label rewrite via build_mmsu post-process path
    out = []
    for row in rows[:LIMIT]:
        row = dict(row)
        row["capability"] = "paralinguistic_recognition"
        row["protocol"] = "paralinguistic_recognition"
        question = str(row.get("question") or "")
        opts = re.findall(r"^([A-D])\.\s*(.+)$", question, flags=re.M)
        stem = re.split(r"\nA\.\s*", question, maxsplit=1)[0].strip()
        labels = [text.strip() for _, text in opts]
        answer = str(row.get("answer") or row.get("text") or "").strip()
        choice_map = {k: v.strip() for k, v in opts}
        if answer in choice_map:
            answer = choice_map[answer]
        if labels:
            row["closed_labels"] = labels
            row["choice_a"] = labels[0] if len(labels) > 0 else ""
            row["choice_b"] = labels[1] if len(labels) > 1 else ""
            row["choice_c"] = labels[2] if len(labels) > 2 else ""
            row["choice_d"] = labels[3] if len(labels) > 3 else ""
            row["question"] = (
                f"{stem} Choose exactly one label from: {', '.join(labels)}. "
                f"Return only the attribute label."
            )
        row["answer"] = answer
        row["text"] = answer
        row["subset"] = _mmsu_task_key(row["sample_id"])
        out.append(row)
    return out


def build_mavd() -> list[dict]:
    """Crop short event windows from L0 MAVD flac using timestamps."""
    jsonl = Path("/mnt/afs/eval_data/benchmarks/MAVD-traffic/manifests/sound_event.jsonl")
    clip_dir = CLIPS / "sound_event" / "mavd_traffic"
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            if doc.get("split") != "test":
                continue
            src = Path(doc.get("audio") or "")
            start, end = doc.get("start_seconds"), doc.get("end_seconds")
            label = doc.get("event_label")
            if not src.is_file() or start is None or end is None or not label:
                continue
            # keep windows reasonably short for Omni
            if float(end) - float(start) > 15:
                end = float(start) + 10
            sid = str(doc.get("id") or f"mavd_{len(rows):04d}")
            dst = clip_dir / f"{sid}.wav"
            if not dst.is_file():
                try:
                    from suites.native.build_asr_native_samples import ffmpeg_crop  # type: ignore
                except Exception:
                    pass
                cmd = [
                    FFMPEG, "-y", "-ss", f"{float(start):.6f}", "-to", f"{float(end):.6f}",
                    "-i", str(src), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dst),
                ]
                dst.parent.mkdir(parents=True, exist_ok=True)
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0 or not dst.is_file():
                    continue
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "mavd_traffic",
                    "capability": "sound_event",
                    "protocol": "sound_event_multilabel",
                    "WavPath": str(dst),
                    "labels": [str(label)],
                    "text": str(label),
                    "answer": [str(label)],
                    "src_audio": str(src),
                    "start_time": float(start),
                    "end_time": float(end),
                    "source_index": str(jsonl),
                }
            )
    return rows


def build_sonyc() -> list[dict]:
    jsonl = Path("/mnt/afs/eval_data/benchmarks/SONYC-UST-v2.3/manifests/sound_event_tagging.jsonl")
    ann = Path("/mnt/afs/eval_data/benchmarks/SONYC-UST-v2.3/annotations.csv")
    # coarse presence columns
    coarse = [
        ("1_engine_presence", "engine"),
        ("2_machinery-impact_presence", "machinery impact"),
        ("3_non-machinery-impact_presence", "non machinery impact"),
        ("4_powered-saw_presence", "powered saw"),
        ("5_alert-signal_presence", "alert signal"),
        ("6_music_presence", "music"),
        ("7_human-voice_presence", "human voice"),
        ("8_dog_presence", "dog"),
    ]
    by_file: dict[str, list[str]] = {}
    if ann.is_file():
        with ann.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "test":
                    continue
                fn = row.get("audio_filename") or ""
                labels = [name for col, name in coarse if str(row.get(col)) == "1"]
                if fn and labels:
                    by_file[fn] = labels
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            if doc.get("split") != "test":
                continue
            audio = Path(doc.get("audio") or "")
            fn = doc.get("audio_filename") or audio.name
            labels = by_file.get(fn) or []
            if not audio.is_file() or not labels:
                continue
            rows.append(
                {
                    "sample_id": str(doc.get("id") or f"sonyc_{len(rows):04d}"),
                    "dataset": "sonyc_ust_v23",
                    "capability": "sound_event",
                    "protocol": "sound_event_multilabel",
                    "WavPath": str(audio),
                    "labels": labels,
                    "text": ", ".join(labels),
                    "answer": labels,
                    "source_index": str(jsonl),
                }
            )
    return rows


def _strip_mcq_stem(question: str) -> str:
    """Drop A/B/C/D choice blocks (newline or inline after 'select from...')."""
    text = str(question or "").strip()
    text = re.split(
        r"(?i)\s*Please select from the choices below\s*:", text, maxsplit=1
    )[0].strip()
    text = re.split(r"\nA\.\s*", text, maxsplit=1)[0].strip()
    text = re.split(r"\s+A\.\s+", text, maxsplit=1)[0].strip()
    return text


def _open_qa_instruction(stem: str, *, zh: bool = False) -> str:
    stem = stem.strip()
    if zh:
        return f"{stem}\n\n根据音频作答，只输出答案。"
    return (
        f"{stem}\n\nAnswer the question using the audio. Output only the answer."
    )


def build_mmsu(capability: str) -> list[dict]:
    if capability == "audio_reasoning":
        rows = take(iter_mmsu(exclude=MMSU_REASONING_EXCLUDE))
    else:
        rows = take(iter_mmsu(prefixes=MMSU_PREFIX[capability]))
    for row in rows:
        row["capability"] = capability
        row["protocol"] = capability
        if capability == "code_switch":
            # MMSU source is multiple-choice, but the Native code-switch task
            # evaluates open answers.  Do not leak A/B/C/D options to the model.
            question = str(row.get("question") or "")
            question = re.split(r"\nA\.\s", question, maxsplit=1)[0].strip()
            row["question"] = (
                f"{question}\n\nAnswer the question using the audio. Output only the answer."
            )
        elif capability == "audio_reasoning":
            # Source is MCQ; Native AR is open QA (option text as gold).
            question = str(row.get("question") or "")
            opts = re.findall(r"^([A-D])\.\s*(.+)$", question, flags=re.M)
            choice_map = {k: v.strip() for k, v in opts}
            stem = _strip_mcq_stem(question)
            answer = str(row.get("answer") or row.get("text") or "").strip()
            answer_letter = ""
            if answer in choice_map:
                answer_letter = answer
                answer = choice_map[answer]
            elif answer.upper() in choice_map:
                answer_letter = answer.upper()
                answer = choice_map[answer_letter]
            else:
                for letter, text in choice_map.items():
                    if text == answer:
                        answer_letter = letter
                        break
            row["question_mcq_original"] = question
            if choice_map:
                row["choice_a"] = choice_map.get("A", "")
                row["choice_b"] = choice_map.get("B", "")
                row["choice_c"] = choice_map.get("C", "")
                row["choice_d"] = choice_map.get("D", "")
            if answer_letter:
                row["answer_letter"] = answer_letter
            row["question"] = _open_qa_instruction(stem)
            row["answer"] = answer
            row["text"] = answer
            row["protocol"] = "open_qa"
        elif capability in ("acoustic_scene", "sound_event", "paralinguistic_recognition"):
            # Keep the four options, but ask for the label text (not A/B/C/D)
            # so scoring matches other closed slots via mesh-classification.
            question = str(row.get("question") or "")
            opts = re.findall(r"^([A-D])\.\s*(.+)$", question, flags=re.M)
            stem = re.split(r"\nA\.\s*", question, maxsplit=1)[0].strip()
            labels = [text.strip() for _, text in opts]
            answer = str(row.get("answer") or row.get("text") or "").strip()
            if answer in dict(opts):
                answer = dict(opts)[answer].strip()
            kind = {
                "acoustic_scene": "scene",
                "sound_event": "event",
                "paralinguistic_recognition": "attribute",
            }[capability]
            if labels:
                row["closed_labels"] = labels
                row["choice_a"] = labels[0] if len(labels) > 0 else ""
                row["choice_b"] = labels[1] if len(labels) > 1 else ""
                row["choice_c"] = labels[2] if len(labels) > 2 else ""
                row["choice_d"] = labels[3] if len(labels) > 3 else ""
                row["question"] = (
                    f"{stem} Choose exactly one label from: {', '.join(labels)}. "
                    f"Return only the {kind} label."
                )
            row["answer"] = answer
            row["text"] = answer
        elif capability == "speaker_count":
            # MMSU official is MCQ ("N people"); Native open track asks for an
            # integer only so mesh-speaker-count can score speaker_count_acc.
            question = str(row.get("question") or "")
            opts = re.findall(r"^([A-D])\.\s*(.+)$", question, flags=re.M)
            stem = re.split(r"\nA\.\s*", question, maxsplit=1)[0].strip()
            answer = str(row.get("answer") or row.get("text") or "").strip()
            choice_map = {k: v.strip() for k, v in opts}
            if answer in choice_map:
                answer = choice_map[answer]
            match = re.search(r"\d+", answer)
            count = int(match.group(0)) if match else None
            row["question"] = (
                f"{stem or 'How many different speakers are in the audio?'} "
                "Output only a single integer."
            )
            if count is not None:
                row["answer"] = count
                row["text"] = count
                row["count"] = count
            else:
                row["answer"] = answer
                row["text"] = answer
    return rows


def build_msswift_caption(jsonl: Path, root: Path, dataset: str) -> list[dict]:
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            rel, _, caption = parse_messages(doc)
            if not rel or not caption:
                continue
            wav = root / rel
            if not wav.is_file():
                continue
            rows.append(
                {
                    "sample_id": doc.get("id") or f"{dataset}_{len(rows):04d}",
                    "dataset": dataset,
                    "capability": "audio_caption",
                    "protocol": "audio_caption",
                    "WavPath": str(wav),
                    "text": caption,
                    "answer": caption,
                    "source_index": str(jsonl),
                }
            )
    return rows


def _esc50_closed_labels(meta: Path) -> list[str]:
    cats: set[str] = set()
    with meta.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cat = str(row.get("category") or "").strip()
            if cat:
                cats.add(cat)
    return sorted(cats)


def build_esc50() -> list[dict]:
    """Fold-5 ESC-50 closed-set smoke: one sample per class when possible.

    Never put the gold label into ``question`` (old leak made accuracy fake).
    Prefer the audited 8p31 smoke via ``migrate_sound_event_p8`` when registered.
    """
    root = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/ESC-50/ESC-50-master")
    meta = root / "meta" / "esc50.csv"
    audio = root / "audio"
    closed = _esc50_closed_labels(meta)
    prompt = (
        "Return exactly one ESC-50 label from this list: "
        + ", ".join(closed)
        + ". Return only the event label."
    )
    by_cat: dict[str, list[dict]] = {}
    with meta.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("fold")) != "5":
                continue
            wav = audio / row["filename"]
            if not wav.is_file():
                continue
            label = str(row["category"]).strip()
            by_cat.setdefault(label, []).append(
                {
                    "sample_id": f"esc50_{Path(row['filename']).stem}",
                    "dataset": "esc50",
                    "capability": "sound_event",
                    "protocol": "sound_event",
                    "WavPath": str(wav),
                    "text": label,
                    "answer": label,
                    "closed_labels": closed,
                    "question": prompt,
                    "source_index": str(meta),
                }
            )
    # round-robin across classes for diversity
    rows: list[dict] = []
    cats = sorted(by_cat)
    idx = 0
    while len(rows) < LIMIT and cats:
        cat = cats[idx % len(cats)]
        bucket = by_cat[cat]
        if bucket:
            rows.append(bucket.pop(0))
        if not bucket:
            cats = [c for c in cats if by_cat[c]]
            if not cats:
                break
            idx = 0
            continue
        idx += 1
    return rows[:LIMIT]


def migrate_sound_event_p8(bench: str) -> list[dict]:
    """Import audited 8p31 sound_event smoke (闭集 / 切片 / 多标签)."""
    src = P8 / f"sound_event_{bench}_smoke" / f"sound_event_{bench}_10.jsonl"
    rows = migrate_smoke_manifest(src, "sound_event", bench, f"sound_event_{bench}")
    for row in rows:
        row["capability"] = "sound_event"
        if bench == "mmsu":
            labels = [
                str(row.get(k) or "").strip()
                for k in ("choice_a", "choice_b", "choice_c", "choice_d")
                if str(row.get(k) or "").strip()
            ]
            stem = re.split(
                r"\nA\.\s*", str(row.get("question") or ""), maxsplit=1
            )[0].strip()
            answer = str(row.get("answer") or row.get("text") or "").strip()
            letter_map = {
                "A": row.get("choice_a"),
                "B": row.get("choice_b"),
                "C": row.get("choice_c"),
                "D": row.get("choice_d"),
            }
            if answer in letter_map and letter_map[answer]:
                answer = str(letter_map[answer]).strip()
            if labels:
                row["closed_labels"] = labels
                row["question"] = (
                    f"{stem} Choose exactly one label from: {', '.join(labels)}. "
                    f"Return only the event label."
                )
            row["answer"] = answer
            row["text"] = answer
            row["protocol"] = "sound_event"
        elif bench in ("mavd_traffic", "sonyc_ust_v23"):
            labels = row.get("labels") or row.get("answer")
            if isinstance(labels, str):
                labels = [x.strip() for x in labels.split(",") if x.strip()]
            if not isinstance(labels, list):
                labels = [str(labels)] if labels else []
            labels = [str(x).strip() for x in labels if str(x).strip()]
            row["labels"] = labels
            row["answer"] = labels
            row["text"] = ", ".join(labels)
            row["protocol"] = row.get("protocol") or "sound_event_multilabel"
            closed = (
                MAVD_CLOSED_LABELS if bench == "mavd_traffic" else SONYC_CLOSED_LABELS
            )
            row["closed_labels"] = list(closed)
            if bench == "mavd_traffic":
                row["question"] = (
                    "Identify all traffic sound events present in the audio. "
                    "Choose zero or more labels from this closed list only: "
                    + ", ".join(closed)
                    + ". Return a comma-separated list of labels from the list, "
                    "with no explanations."
                )
            else:
                row["question"] = (
                    "Name the prominent urban sound event(s) you hear. "
                    "Choose zero or more labels from this closed list only: "
                    + ", ".join(closed)
                    + ". Return a comma-separated list of labels from the list, "
                    "with no explanations."
                )
        else:
            # esc50 / tut: keep closed-set question from 8p31
            answer = str(row.get("answer") or row.get("text") or "").strip()
            row["answer"] = answer
            row["text"] = answer
            row["protocol"] = row.get("protocol") or "sound_event"
            if bench == "tut_sound_events_2017" and not str(row.get("question") or "").strip():
                closed = row.get("closed_labels") or [
                    "brakes squeaking",
                    "car",
                    "children",
                    "large vehicle",
                    "people speaking",
                    "people walking",
                ]
                row["closed_labels"] = closed
                row["question"] = (
                    "Identify the labelled sound event. Choose exactly one label from: "
                    + ", ".join(closed)
                    + ". Return only the event label."
                )
    return rows


MAVD_CLOSED_LABELS = [
    "bus",
    "bus/brakes",
    "bus/compressor",
    "bus/engine_accelerating",
    "bus/engine_idling",
    "bus/wheel_rolling",
    "car",
    "car/brakes",
    "car/engine_accelerating",
    "car/engine_idling",
    "car/wheel_rolling",
    "chatter",
    "motorcycle",
    "motorcycle/brakes",
    "motorcycle/engine_accelerating",
    "motorcycle/engine_idling",
    "music",
    "other",
    "truck",
    "truck/brakes",
    "truck/compressor",
    "truck/engine_accelerating",
    "truck/engine_idling",
]

SONYC_CLOSED_LABELS = [
    "alert signal",
    "car alarm",
    "car horn",
    "chainsaw",
    "dog",
    "dog barking whining",
    "engine",
    "engine of uncertain size",
    "hoe ram",
    "human voice",
    "jackhammer",
    "large crowd",
    "large sounding engine",
    "machinery impact",
    "medium sounding engine",
    "mobile music",
    "music",
    "music from uncertain source",
    "non machinery impact",
    "other unknown alert signal",
    "other unknown human voice",
    "other unknown impact machinery",
    "other unknown powered saw",
    "person or small group shouting",
    "person or small group talking",
    "pile driver",
    "powered saw",
    "reverse beeper",
    "rock drill",
    "siren",
    "small medium rotating saw",
    "small sounding engine",
    "stationary music",
]


def build_spoken_squad() -> list[dict]:
    raw = Path("/mnt/afs/oss_data/datasets/04_dialogue_slu/Spoken-SQuAD/raw")
    jsonl = raw / "test_WER44.json"
    wav_dir = raw / "merged_wav" / "test_wer44"
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            wav = wav_dir / f"{i}_0.wav"
            if not wav.is_file():
                # try alternate naming from full_eval
                alt = list(wav_dir.glob(f"{i}_*.wav"))
                wav = alt[0] if alt else wav
            if not wav.is_file():
                continue
            ans = doc.get("answers") or {}
            texts = ans.get("text") or []
            answer = texts[0] if texts else ""
            if not answer:
                continue
            rows.append(
                {
                    "sample_id": f"ssq_{i}_0",
                    "dataset": "spoken_squad",
                    "capability": "qa",
                    "protocol": "qa",
                    "WavPath": str(wav),
                    "question": str(doc.get("question") or "").strip(),
                    "answer": answer,
                    "text": answer,
                    "source_index": str(jsonl),
                }
            )
    return rows


def build_vocalbench_json(path: Path, dataset: str, capability: str, root: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    zh = "zh" in dataset.lower() or "zh" in path.name.lower()
    for item in data:
        if len(rows) >= LIMIT:
            break
        rel = item.get("Audio") or item.get("audio") or ""
        candidates = [root / "audio" / rel, root / rel]
        wav = next((p for p in candidates if p.is_file()), None)
        if wav is None:
            continue
        answer = str(
            item.get("Answer")
            or item.get("Answer_zh")
            or item.get("Answer_en")
            or item.get("answer")
            or ""
        ).strip()
        question = str(
            item.get("Question") or item.get("Question_zh") or item.get("question") or ""
        ).strip()
        if not answer:
            continue
        if capability == "audio_reasoning" and question:
            question = _open_qa_instruction(question, zh=zh)
        rows.append(
            {
                "sample_id": str(item.get("Qid") or item.get("id") or f"{dataset}_{len(rows):04d}"),
                "dataset": dataset,
                "capability": capability,
                "protocol": "open_qa" if capability == "audio_reasoning" else capability,
                "WavPath": str(wav),
                "question": question,
                "answer": answer,
                "text": answer,
                "source_index": str(path),
            }
        )
    return rows


def build_uro(path: Path, capability: str) -> list[dict]:
    rows = []
    root = path.parent
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            audio = (
                doc.get("source_wav")
                or doc.get("audio")
                or doc.get("WavPath")
                or doc.get("audio_path")
            )
            if not audio:
                continue
            wav = Path(audio) if Path(str(audio)).is_absolute() else (root / audio)
            if not wav.is_file():
                continue
            answer = str(
                doc.get("target_text") or doc.get("answer") or doc.get("output") or doc.get("text") or ""
            ).strip()
            question = str(
                doc.get("source_text") or doc.get("question") or doc.get("instruction") or ""
            ).strip()
            if not answer:
                continue
            protocol = capability
            if capability == "audio_reasoning" and question:
                # MuChoEval (MuChoMusic) MCQ has intentional IR/CU/IU distractors;
                # Native AR uses open QA (short-label gold), not the MCQ protocol.
                stem = _strip_mcq_stem(question)
                question = _open_qa_instruction(stem)
                protocol = "open_qa"
            rows.append(
                {
                    "sample_id": str(doc.get("id") or f"uro_{len(rows):04d}"),
                    "dataset": "uro_bench",
                    "capability": capability,
                    "protocol": protocol,
                    "WavPath": str(wav),
                    "question": question,
                    "answer": answer,
                    "text": answer,
                    "source_index": str(path),
                }
            )
    return rows


def build_covost() -> list[dict]:
    root = Path("/mnt/afs/oss_data/datasets/07_speech_translation/CoVoST_2_zh_en_bidirectional")
    # Catalog uses zh→en historically; annotation prefers en→zh-CN test. Use zh→en msswift for prompt alignment.
    jsonl = root / "CoVoST_zh-CN_en_test_msswift_norm.jsonl"
    if not jsonl.is_file():
        jsonl = root / "CoVoST_en_zh-CN_test_msswift_norm.jsonl"
    rows = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            rel, question, text = parse_messages(doc)
            if not rel or not text:
                continue
            wav = root / rel
            if not wav.is_file():
                continue
            rows.append(
                {
                    "sample_id": doc.get("id") or f"covost_{len(rows):04d}",
                    "dataset": "covost2",
                    "capability": "speech_translation",
                    "protocol": "speech_translation",
                    "WavPath": str(wav),
                    "question": question or "Translate the speech.",
                    "text": text,
                    "answer": text,
                    "source_index": str(jsonl),
                }
            )
    return rows


def build_libritts() -> list[dict]:
    root = Path("/mnt/afs/eval_data/01_basic_transcription/LibriTTS/test-clean")
    rows = []
    for tsv in sorted(root.glob("*/*/*.trans.tsv")):
        with tsv.open(encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= LIMIT:
                    return rows
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                utt_id, text = parts[0], parts[1]
                wav = tsv.parent / f"{utt_id}.wav"
                if not wav.is_file():
                    continue
                rows.append(
                    {
                        "sample_id": utt_id,
                        "dataset": "libritts",
                        "capability": "speech_generation",
                        "protocol": "speech_generation",
                        "WavPath": str(wav),
                        "text": text,
                        "answer": text,
                        "source_index": str(tsv),
                    }
                )
    return rows


def build_voxceleb1() -> list[dict]:
    """Native smoke: 5 same + 5 diff trials, stride across veri_test (not first-10)."""
    veri = Path(
        "/mnt/afs/users/wangyl/benchmark_annotation_audit/native_v2_pipeline/references/voxceleb1/veri_test.txt"
    )
    roots = [
        Path("/mnt/afs/oss_data/datasets/03_multi_speaker/VoxCeleb_1_2_multi_speaker_multi_speaker/juliuscn/voxceleb/vox1/wav"),
        Path("/mnt/afs/oss_data/datasets/03_multi_speaker/VoxCeleb1/wav"),
        Path("/mnt/afs/eval_data/03_multi_speaker/VoxCeleb1/wav"),
    ]
    root = next((r for r in roots if r.is_dir()), None)
    if root is None or not veri.is_file():
        return []

    same: list[dict] = []
    diff: list[dict] = []
    with veri.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label_s, p1, p2 = parts
            try:
                label = int(label_s)
            except ValueError:
                continue
            w1, w2 = root / p1, root / p2
            if not w1.is_file() or not w2.is_file():
                continue
            row = {
                "sample_id": f"voxceleb1_veri_{idx:05d}_{'same' if label == 1 else 'diff'}",
                "dataset": "voxceleb1",
                "capability": "speaker_verification",
                "protocol": "speaker_verification",
                "WavPath": str(w1),
                "WavPath2": str(w2),
                "label": label,
                "text": label,
                "same_speaker": bool(label == 1),
                "enroll_relpath": p1,
                "test_relpath": p2,
                "source_index": str(veri),
                "source_line": idx,
            }
            (same if label == 1 else diff).append(row)

    def _stride_take(pool: list[dict], n: int) -> list[dict]:
        if not pool:
            return []
        if len(pool) <= n:
            return pool[:n]
        step = max(1, len(pool) // n)
        picked = [pool[i * step] for i in range(n)]
        # de-dup if step collapses
        seen = set()
        out = []
        for row in picked:
            sid = row["sample_id"]
            if sid in seen:
                continue
            seen.add(sid)
            out.append(row)
        for row in pool:
            if len(out) >= n:
                break
            if row["sample_id"] not in seen:
                seen.add(row["sample_id"])
                out.append(row)
        return out[:n]

    half = max(1, LIMIT // 2)
    rows = _stride_take(same, half) + _stride_take(diff, LIMIT - half)
    rows.sort(key=lambda r: int(r["source_line"]))
    return rows


def build_from_l0_manifest(jsonl: Path, capability: str, dataset: str, *, multilabel: bool = False) -> list[dict]:
    rows = []
    labels_set: set[str] = set()
    with jsonl.open(encoding="utf-8") as handle:
        preview = []
        for i, line in enumerate(handle):
            doc = json.loads(line)
            preview.append(doc)
            lab = doc.get("scene_label") or doc.get("scene_class") or doc.get("label") or doc.get("event_label")
            if lab:
                labels_set.add(str(lab))
            if i >= 200:
                break
    closed = sorted(labels_set) if labels_set else []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            audio = doc.get("audio") or doc.get("WavPath") or doc.get("audio_path")
            if not audio or not Path(audio).is_file():
                continue
            if multilabel:
                labels = doc.get("labels") or doc.get("events") or []
                if isinstance(labels, str):
                    labels = [x.strip() for x in labels.split(",") if x.strip()]
                text = ", ".join(str(x) for x in labels)
                if not text:
                    continue
                rows.append(
                    {
                        "sample_id": str(doc.get("id") or f"{dataset}_{len(rows):04d}"),
                        "dataset": dataset,
                        "capability": capability,
                        "protocol": "sound_event_multilabel",
                        "WavPath": str(audio),
                        "labels": list(labels),
                        "text": text,
                        "answer": text,
                        "source_index": str(jsonl),
                    }
                )
            else:
                label = (
                    doc.get("scene_label")
                    or doc.get("scene_class")
                    or doc.get("acoustic_scene_label")
                    or doc.get("official_scene_label")
                    or doc.get("label")
                    or doc.get("event_label")
                    or doc.get("masked_word")
                    or doc.get("answer")
                )
                if label is None:
                    continue
                label = str(label)
                question = ""
                if capability == "acoustic_scene" and closed:
                    question = (
                        "Identify the acoustic scene. Choose exactly one label from: "
                        + ", ".join(closed)
                        + "."
                    )
                elif capability == "qa":
                    question = str(
                        doc.get("question")
                        or "What is the word obscured by the white noise in this audio? Answer with one word only."
                    )
                rows.append(
                    {
                        "sample_id": str(doc.get("id") or f"{dataset}_{len(rows):04d}"),
                        "dataset": dataset,
                        "capability": capability,
                        "protocol": capability,
                        "WavPath": str(audio),
                        "text": label,
                        "answer": label,
                        "question": question,
                        "closed_labels": closed or None,
                        "source_index": str(jsonl),
                    }
                )
    return rows


def build_tut2018() -> list[dict]:
    root = Path("/mnt/afs/eval_data/benchmarks/TUT_Urban_Acoustic_Scenes_2018_Mobile")
    # Prefer evaluation meta if present
    metas = list(root.rglob("*evaluation*.csv")) + list(root.rglob("*train*.csv"))
    rows = []
    for meta in metas:
        with meta.open(encoding="utf-8") as handle:
            # fold-style: filename\tscene or header
            header = handle.readline()
            handle.seek(0)
            if "filename" in header.lower() or "\t" in header:
                reader = csv.DictReader(handle, delimiter="\t" if "\t" in header else ",")
                for row in reader:
                    if len(rows) >= LIMIT:
                        return rows
                    fn = row.get("filename") or row.get("File") or ""
                    scene = row.get("scene_label") or row.get("label") or row.get("Scene") or ""
                    if not fn or not scene:
                        continue
                    # search audio
                    hits = list(root.rglob(Path(fn).name))
                    wav = hits[0] if hits else None
                    if wav is None or not wav.is_file():
                        continue
                    rows.append(
                        {
                            "sample_id": f"tut2018_{wav.stem}",
                            "dataset": "tut_urban_acoustic_scenes_2018_mobile",
                            "capability": "acoustic_scene",
                            "protocol": "acoustic_scene",
                            "WavPath": str(wav),
                            "text": scene,
                            "answer": scene,
                            "source_index": str(meta),
                        }
                    )
        if rows:
            break
    # fallback: first 10 wavs under raw with label from filename prefix
    if not rows:
        for wav in sorted(root.rglob("*.wav"))[: LIMIT * 3]:
            scene = wav.name.split("-")[0]
            rows.append(
                {
                    "sample_id": f"tut2018_{wav.stem}",
                    "dataset": "tut_urban_acoustic_scenes_2018_mobile",
                    "capability": "acoustic_scene",
                    "protocol": "acoustic_scene",
                    "WavPath": str(wav),
                    "text": scene,
                    "answer": scene,
                    "source_index": str(wav.parent),
                }
            )
            if len(rows) >= LIMIT:
                break
    return rows[:LIMIT]


def migrate_smoke_manifest(src: Path, capability: str, dataset: str, dest_name: str) -> list[dict]:
    """Copy smoke/P8 clip wavs into eval_clips and rewrite WavPath(/WavPath2)."""
    rows = []
    clip_dir = CLIPS / capability / dataset
    clip_dir.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= LIMIT:
                break
            doc = json.loads(line)
            wav = Path(doc.get("WavPath") or doc.get("audio_path") or "")
            if not wav.is_file():
                continue
            dst = clip_dir / wav.name
            if not dst.is_file():
                shutil.copy2(wav, dst)
            out = dict(doc)
            out["WavPath"] = str(dst)
            wav2 = Path(doc.get("WavPath2") or "")
            if wav2.is_file():
                dst2 = clip_dir / wav2.name
                if not dst2.is_file():
                    shutil.copy2(wav2, dst2)
                out["WavPath2"] = str(dst2)
            out["dataset"] = dataset
            out["capability"] = capability
            out["protocol"] = out.get("protocol") or capability
            out["sample_id"] = out.get("sample_id") or f"{dataset}_{len(rows):04d}"
            out["source_index"] = str(src)
            rows.append(out)
    return rows


def _qa_answer_text(answer) -> str:
    if isinstance(answer, list):
        parts = [str(x).strip() for x in answer if x is not None and str(x).strip()]
        return parts[0] if parts else ""
    if answer is None:
        return ""
    return str(answer).strip()


def migrate_qa_p8_smoke(
    src: Path,
    dataset: str,
    *,
    require_answer: bool = True,
    short_track_only: bool = False,
) -> list[dict]:
    """Import 8p31 QA smoke metadata (乙组抽样) into Native indexes."""
    if not src.is_file():
        return []
    rows: list[dict] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if short_track_only:
            track = doc.get("qa_track")
            sid = str(doc.get("sample_id") or "")
            if track == "open":
                continue
            if track not in ("short", None) and not (
                sid.startswith("knowledge-") or sid.startswith("chinese_knowledge-")
            ):
                continue
        wav = Path(doc.get("WavPath") or doc.get("audio_locator") or "")
        if not wav.is_file():
            continue
        answer_text = _qa_answer_text(doc.get("answer"))
        if require_answer and not answer_text:
            continue
        question = str(doc.get("question") or "").strip()
        if not question:
            continue
        rows.append(
            {
                "sample_id": str(doc.get("sample_id") or f"{dataset}_{len(rows):04d}"),
                "dataset": dataset,
                "capability": "qa",
                "protocol": "qa",
                "subset": doc.get("subset_id") or doc.get("subset"),
                "WavPath": str(wav),
                "question": question,
                "answer": answer_text,
                "text": answer_text,
                "answers": doc.get("answer") if isinstance(doc.get("answer"), list) else None,
                "qa_track": doc.get("qa_track") or "short",
                "source_index": str(src),
                "record_id": doc.get("record_id"),
            }
        )
        if len(rows) >= LIMIT:
            break
    return rows


def build_qa_vocalbench_from_p8() -> list[dict]:
    """Catalog knowledge short-track: use 8p31 knowledge smoke, pad to 10."""
    short = migrate_qa_p8_smoke(
        P8 / "qa_vocalbench_smoke" / "qa_vocalbench_knowledge_10.jsonl",
        "vocalbench",
    )
    if len(short) < LIMIT:
        short = migrate_qa_p8_smoke(
            P8 / "qa_vocalbench_smoke" / "qa_vocalbench_10.jsonl",
            "vocalbench",
            short_track_only=True,
        )
    if len(short) >= LIMIT:
        return short[:LIMIT]
    have = {r["sample_id"] for r in short}
    data = json.loads(
        Path(
            "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench/json/knowledge.json"
        ).read_text(encoding="utf-8")
    )
    root = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench")
    step = max(1, len(data) // 40)
    for idx in range(0, len(data), step):
        if len(short) >= LIMIT:
            break
        item = data[idx]
        sid = str(item.get("Qid") or item.get("id") or "")
        if not sid or sid in have:
            continue
        rel = item.get("Audio") or item.get("audio") or ""
        wav = root / "audio" / rel
        if not wav.is_file():
            wav = root / rel
        if not wav.is_file():
            continue
        answer = str(item.get("Answer") or item.get("answer") or "").strip()
        question = str(item.get("Question") or item.get("question") or "").strip()
        if not answer or not question:
            continue
        short.append(
            {
                "sample_id": sid,
                "dataset": "vocalbench",
                "capability": "qa",
                "protocol": "qa",
                "subset": "knowledge",
                "WavPath": str(wav),
                "question": question,
                "answer": answer,
                "text": answer,
                "qa_track": "short",
                "source_index": str(root / "json" / "knowledge.json"),
            }
        )
        have.add(sid)
    return short[:LIMIT]


def build_qa_uro_from_p8() -> list[dict]:
    """Keep 8p31 rows with gold; pad with APE-zh target_text items."""
    rows = migrate_qa_p8_smoke(
        P8 / "qa_uro_bench_smoke" / "qa_uro_bench_10.jsonl",
        "uro_bench",
        require_answer=True,
    )
    if len(rows) >= LIMIT:
        return rows[:LIMIT]
    have = {r["sample_id"] for r in rows}
    for extra in build_uro(
        Path(
            "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/URO-Bench/basic/APE-zh/test.jsonl"
        ),
        "qa",
    ):
        if len(rows) >= LIMIT:
            break
        if extra["sample_id"] in have:
            continue
        rows.append(extra)
        have.add(extra["sample_id"])
    return rows[:LIMIT]


def build_qa_voicebench_from_p8() -> list[dict]:
    """8p31 MMSU-open gold (2) + openbookqa pad (has reference + wav)."""
    rows = migrate_qa_p8_smoke(
        P8 / "qa_voicebench_smoke" / "qa_voicebench_mmsu_open_2.jsonl",
        "voicebench",
        require_answer=True,
    )
    if len(rows) >= LIMIT:
        return rows[:LIMIT]
    have = {r["sample_id"] for r in rows}
    for extra in build_voicebench_parquet("qa"):
        if len(rows) >= LIMIT:
            break
        if extra["sample_id"] in have:
            continue
        rows.append(extra)
        have.add(extra["sample_id"])
    return rows[:LIMIT]


def build_voicebench_parquet(capability: str) -> list[dict]:
    """VoiceBench: prefer subsets with gold + exported wav under VoiceBench_wav."""
    import pyarrow.parquet as pq

    root = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench")
    wav_root = Path(
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench_wav/wav"
    )
    if capability == "qa":
        # openbookqa has reference letters; alpacaeval_full is open-ended (no gold).
        subset = "openbookqa"
        path = root / "openbookqa" / "test-00000-of-00001.parquet"
        shard = "test-00000-of-00001"
    else:
        subset = "bbh"
        path = root / "bbh" / "test-00000-of-00023.parquet"
        shard = "test-00000-of-00023"
    if not path.is_file():
        return []
    table = pq.read_table(path)
    rows = []
    for i in range(table.num_rows):
        if len(rows) >= LIMIT:
            break
        item = {col: table.column(col)[i].as_py() for col in table.column_names}
        answer = item.get("reference") or item.get("answer") or item.get("output")
        question = item.get("prompt") or item.get("question") or ""
        if answer in (None, ""):
            continue
        wav = wav_root / subset / shard / f"{shard}_row{i}.wav"
        if not wav.is_file():
            continue
        rows.append(
            {
                "sample_id": f"voicebench_{subset}_{i}",
                "dataset": "voicebench",
                "capability": capability,
                "protocol": capability,
                "subset": subset,
                "WavPath": str(wav),
                "question": str(question),
                "answer": str(answer),
                "text": str(answer),
                "source_index": str(path),
            }
        )
    return rows


def build_code_switch_vocalbench() -> list[dict]:
    root = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh")
    path = root / "json" / "cs_knowledge.json"
    return build_vocalbench_json(path, "vocalbench_zh", "code_switch", root)


def build_code_switch_vocalbench_open_ended() -> list[dict]:
    root = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh")
    path = root / "json" / "cs_open_ended.json"
    rows = build_vocalbench_json(path, "vocalbench_zh", "code_switch", root)
    for row in rows:
        row["subset"] = "code_switch_open_ended"
    return rows


BUILDERS: dict[tuple[str, str], callable] = {}


def register():
    BUILDERS[("audio_caption", "audiocaps")] = lambda: build_msswift_caption(
        Path("/mnt/afs/oss_data/datasets/99_auxiliary_or_out_of_scope/AudioCaps/AudioCaps_test_AudioCaptioning.jsonl"),
        Path("/mnt/afs/oss_data/datasets/99_auxiliary_or_out_of_scope/AudioCaps"),
        "audiocaps",
    )
    BUILDERS[("audio_caption", "clotho")] = lambda: build_msswift_caption(
        Path("/mnt/afs/oss_data/datasets/99_auxiliary_or_out_of_scope/Clotho/Clotho_test_AudioCaptioning.jsonl"),
        Path("/mnt/afs/oss_data/datasets/99_auxiliary_or_out_of_scope/Clotho"),
        "clotho",
    )
    BUILDERS[("sound_event", "esc50")] = lambda: migrate_sound_event_p8("esc50") or build_esc50()
    BUILDERS[("qa", "spoken_squad")] = lambda: migrate_qa_p8_smoke(
        P8 / "qa_spoken_squad_smoke" / "qa_spoken_squad_10.jsonl",
        "spoken_squad",
    )
    BUILDERS[("speech_translation", "covost2")] = build_covost
    BUILDERS[("speech_generation", "libritts")] = build_libritts
    BUILDERS[("speaker_verification", "voxceleb1")] = build_voxceleb1

    for cap in (
        "acoustic_scene",
        "code_switch",
        "speech_translation",
        "spoof_detection",
        "speaker_count",
        "audio_reasoning",
    ):
        BUILDERS[(cap, "mmsu")] = lambda c=cap: build_mmsu(c)
    # Prefer audited 8p31 MMSU non-verbal smoke (closed label text + mesh-classification).
    BUILDERS[("sound_event", "mmsu")] = (
        lambda: migrate_sound_event_p8("mmsu") or build_mmsu("sound_event")
    )
    BUILDERS[("paralinguistic_recognition", "mmsu")] = build_paralinguistic_mmsu

    BUILDERS[("qa", "vocalbench")] = build_qa_vocalbench_from_p8
    BUILDERS[("qa", "vocalbench_zh")] = lambda: migrate_qa_p8_smoke(
        P8 / "qa_vocalbench_zh_smoke" / "qa_vocalbench_zh_10.jsonl",
        "vocalbench_zh",
    )
    BUILDERS[("qa", "uro_bench")] = build_qa_uro_from_p8
    BUILDERS[("qa", "ear_wdyl")] = lambda: migrate_qa_p8_smoke(
        P8 / "qa_ear_wdyl_smoke" / "qa_ear_wdyl_10.jsonl",
        "ear_wdyl",
    )
    BUILDERS[("audio_reasoning", "vocalbench")] = lambda: build_vocalbench_json(
        Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench/json/reasoning.json"),
        "vocalbench",
        "audio_reasoning",
        Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench"),
    )
    BUILDERS[("audio_reasoning", "vocalbench_zh")] = lambda: build_vocalbench_json(
        Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh/json/reasoning.json"),
        "vocalbench_zh",
        "audio_reasoning",
        Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh"),
    )
    BUILDERS[("code_switch", "vocalbench_zh")] = build_code_switch_vocalbench
    BUILDERS[("code_switch", "vocalbench_zh_open_ended")] = (
        build_code_switch_vocalbench_open_ended
    )

    BUILDERS[("audio_reasoning", "uro_bench")] = lambda: build_uro(
        Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/URO-Bench/pro/MuChoEval-en/test.jsonl"),
        "audio_reasoning",
    )

    BUILDERS[("acoustic_scene", "dcase2013_scenes")] = lambda: build_from_l0_manifest(
        Path("/mnt/afs/eval_data/benchmarks/DCASE2013_Scenes_Public/manifests/acoustic_scene.jsonl"),
        "acoustic_scene",
        "dcase2013_scenes",
    )
    BUILDERS[("acoustic_scene", "tut_sound_events_2017")] = lambda: build_from_l0_manifest(
        Path("/mnt/afs/eval_data/benchmarks/TUT-Sound-Events-2017/manifests/acoustic_scene.jsonl"),
        "acoustic_scene",
        "tut_sound_events_2017",
    )
    BUILDERS[("acoustic_scene", "tut_urban_acoustic_scenes_2018_mobile")] = build_tut2018
    # Prefer audited 8p31 smoke (MAVD/TUT event clips + multilabel / closed-set).
    # Fallback builders still exist for offline rebuild without 8p31.
    BUILDERS[("sound_event", "mavd_traffic")] = (
        lambda: migrate_sound_event_p8("mavd_traffic") or build_mavd()
    )
    BUILDERS[("sound_event", "sonyc_ust_v23")] = (
        lambda: migrate_sound_event_p8("sonyc_ust_v23") or build_sonyc()
    )
    BUILDERS[("sound_event", "tut_sound_events_2017")] = (
        lambda: migrate_sound_event_p8("tut_sound_events_2017")
        or build_from_l0_manifest(
            Path(
                "/mnt/afs/eval_data/benchmarks/TUT-Sound-Events-2017/manifests/sound_event.jsonl"
            ),
            "sound_event",
            "tut_sound_events_2017",
        )
    )

    # diar / attr / target_speaker: migrate physical clips into eval_clips
    BUILDERS[("diarization", "alimeeting")] = lambda: migrate_smoke_manifest(
        SMOKE_DIAR / "alimeeting" / "manifest.jsonl", "diarization", "alimeeting", "diarization_alimeeting"
    )
    BUILDERS[("diarization", "ami")] = lambda: migrate_smoke_manifest(
        SMOKE_DIAR / "ami" / "manifest.jsonl", "diarization", "ami", "diarization_ami"
    )
    BUILDERS[("diarization", "chime_6")] = lambda: migrate_smoke_manifest(
        SMOKE_DIAR / "chime_6" / "manifest.jsonl", "diarization", "chime_6", "diarization_chime_6"
    )
    BUILDERS[("diarization", "aishell_4")] = lambda: migrate_smoke_manifest(
        SMOKE_DIAR / "aishell_4" / "manifest.jsonl",
        "diarization",
        "aishell_4",
        "diarization_aishell_4",
    )
    BUILDERS[("speaker_attribution", "aishell_4")] = lambda: migrate_smoke_manifest(
        SMOKE_ATTR / "aishell_4" / "manifest.jsonl", "speaker_attribution", "aishell_4", "attr"
    )
    BUILDERS[("speaker_attribution", "aishell_5")] = lambda: migrate_smoke_manifest(
        SMOKE_ATTR / "aishell_5" / "manifest.jsonl", "speaker_attribution", "aishell_5", "attr"
    )
    BUILDERS[("speaker_attribution", "ami")] = lambda: migrate_smoke_manifest(
        SMOKE_ATTR / "ami" / "manifest.jsonl", "speaker_attribution", "ami", "attr"
    )
    BUILDERS[("speaker_attribution", "alimeeting")] = lambda: migrate_smoke_manifest(
        SMOKE_ATTR / "alimeeting" / "manifest.jsonl",
        "speaker_attribution",
        "alimeeting",
        "attr",
    )
    BUILDERS[("target_speaker", "aishell_5")] = lambda: migrate_smoke_manifest(
        SMOKE_TS / "aishell_5" / "manifest.jsonl",
        "target_speaker",
        "aishell_5",
        "ts",
    )

    def voicebench_qa():
        return build_qa_voicebench_from_p8()

    def voicebench_ar():
        # Prefer 8p31 BBH smoke (exported wav + gold); parquet VoiceBench_wav
        # layout is incomplete for bbh shards.
        rows = migrate_qa_p8_smoke(
            P8 / "audio_reasoning_voicebench_smoke" / "audio_reasoning_voicebench_10.jsonl",
            "voicebench",
        )
        for row in rows:
            row["capability"] = "audio_reasoning"
            row["protocol"] = "open_qa"
        if len(rows) >= LIMIT:
            return rows[:LIMIT]
        return build_voicebench_parquet("audio_reasoning")

    BUILDERS[("qa", "voicebench")] = voicebench_qa
    BUILDERS[("audio_reasoning", "voicebench")] = voicebench_ar


def main(only: set[str] | None = None) -> None:
    register()
    CLIPS.mkdir(parents=True, exist_ok=True)
    summary = {}
    missing = []
    for (cap, bench), builder in sorted(BUILDERS.items()):
        key = f"{cap}/{bench}"
        if only and key not in only and f"{cap}_{bench}" not in only:
            continue
        rows = builder()
        path = out_name(cap, bench)
        write_jsonl(path, rows)
        summary[key] = {"n": len(rows), "manifest": str(path)}
        print(f"[done] {key} n={len(rows)} -> {path}")
        if len(rows) < LIMIT:
            missing.append(key)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(f"insufficient samples: {missing}")


if __name__ == "__main__":
    import sys

    only = {a for a in sys.argv[1:] if not a.startswith("-")} or None
    main(only)
