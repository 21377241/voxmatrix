#!/usr/bin/env python3
"""Build full paralinguistic_recognition indexes (MMSU closed-set labels).

Covers all Native catalog prefixes:
  accent / age / disfluency / emotion / gender / intonation / pause /
  pitch / prolonged / sarcasm / stress / speed / vocal_range / volume.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/paralinguistic_recognition"
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_JSONL = MMSU_ROOT / "mmsu_audioqa.jsonl"

PREFIXES = [
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
]


def _task_key(sample_id: str) -> str:
    sid = str(sample_id or "")
    if not sid.startswith("MMSU_"):
        return "unknown"
    rest = sid[len("MMSU_") :]
    parts = rest.split("_")
    for i, part in enumerate(parts):
        if "-" in part:
            return "_".join(parts[:i]) if i else "unknown"
    return rest.rsplit("_", 1)[0] if "_" in rest else rest


def _parse(doc: dict) -> tuple[str | None, str, list[str], str]:
    messages = doc.get("messages") or []
    if len(messages) < 2:
        return None, "", [], ""
    user = (messages[0].get("content") or [{}])[0]
    asst = (messages[1].get("content") or [{}])[0]
    rel = str(user.get("audio_path") or "").strip() or None
    raw_q = str(user.get("text") or "").strip()
    answer = str(asst.get("text") or "").strip()
    opts = re.findall(r"^([A-D])\.\s*(.+)$", raw_q, flags=re.M)
    stem = re.split(r"\nA\.\s*", raw_q, maxsplit=1)[0].strip()
    labels = [t.strip() for _, t in opts]
    choice_map = {k: v.strip() for k, v in opts}
    if answer in choice_map:
        answer = choice_map[answer]
    return rel, stem, labels, answer


def build() -> tuple[int, Path, dict]:
    rows = []
    missing = 0
    by_task: Counter[str] = Counter()
    for line in MMSU_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        if not any(sid.startswith(p) for p in PREFIXES):
            continue
        rel, stem, labels, answer = _parse(doc)
        if not rel or not answer or len(labels) < 2:
            continue
        wav = MMSU_ROOT / rel
        if not wav.is_file():
            alt = MMSU_ROOT / "audio" / Path(rel).name
            wav = alt if alt.is_file() else wav
        if not wav.is_file():
            missing += 1
            continue
        task = _task_key(sid)
        by_task[task] += 1
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "mmsu",
                "benchmark": "mmsu",
                "capability": "paralinguistic_recognition",
                "protocol": "paralinguistic_recognition",
                "subset": task,
                "split": "test",
                "WavPath": str(wav),
                "text": answer,
                "answer": answer,
                "closed_labels": labels,
                "choice_a": labels[0] if len(labels) > 0 else "",
                "choice_b": labels[1] if len(labels) > 1 else "",
                "choice_c": labels[2] if len(labels) > 2 else "",
                "choice_d": labels[3] if len(labels) > 3 else "",
                "question": (
                    f"{stem} Choose exactly one label from: {', '.join(labels)}. "
                    f"Return only the attribute label."
                ),
                "source_index": str(MMSU_JSONL),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "paralinguistic_recognition_mmsu.jsonl"
    from mesh_eval.data.full_index_meta import write_index_jsonl

    n = write_index_jsonl(path, rows, "paralinguistic_recognition__mmsu")
    return n, path, {"missing_audio": missing, "by_task": dict(by_task)}


def main() -> None:
    n, path, meta = build()
    print(f"[full-paralinguistic] n={n} -> {path} meta={meta}")
    print(json.dumps({"n": n, "path": str(path), **meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
