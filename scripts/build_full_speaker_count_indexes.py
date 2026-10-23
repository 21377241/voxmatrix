#!/usr/bin/env python3
"""Build full speaker_count indexes (open integer track).

Native catalog currently covers MMSU total_speaker_counting only. Official
MMSU is MCQ ("N people"); Full Index stores integer count refs and an open
prompt so mesh-speaker-count can score speaker_count_acc.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/speaker_count"

MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_JSONL = MMSU_ROOT / "mmsu_audioqa.jsonl"
PREFIX = "MMSU_total_speaker_counting_"


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    from mesh_eval.data.full_index_meta import slot_id_from_filename, write_index_jsonl

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    slot_id = slot_id_from_filename(OUT.name, name)
    n = write_index_jsonl(path, rows, slot_id)
    return n, path


def _parse_count(answer: str, opts: list[tuple[str, str]]) -> Optional[int]:
    choice_map = {k: v.strip() for k, v in opts}
    text = answer.strip()
    if text in choice_map:
        text = choice_map[text]
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def build_mmsu() -> tuple[int, Path, dict]:
    rows: list[dict] = []
    missing = 0
    skipped = 0
    for line in MMSU_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        if not sid.startswith(PREFIX):
            continue
        messages = doc.get("messages") or []
        if len(messages) < 2:
            skipped += 1
            continue
        user = (messages[0].get("content") or [{}])[0]
        asst = (messages[1].get("content") or [{}])[0]
        rel = str(user.get("audio_path") or "").strip()
        raw_q = str(user.get("text") or "").strip()
        answer = str(asst.get("text") or "").strip()
        opts = re.findall(r"^([A-D])\.\s*(.+)$", raw_q, flags=re.M)
        stem = re.split(r"\nA\.\s*", raw_q, maxsplit=1)[0].strip()
        count = _parse_count(answer, opts)
        if not rel or count is None:
            skipped += 1
            continue
        wav = MMSU_ROOT / rel
        if not wav.is_file():
            alt = MMSU_ROOT / "audio" / Path(rel).name
            wav = alt if alt.is_file() else wav
        if not wav.is_file():
            missing += 1
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "mmsu",
                "benchmark": "mmsu",
                "capability": "speaker_count",
                "protocol": "speaker_count",
                "subset": "total_speaker_counting",
                "split": "test",
                "WavPath": str(wav),
                "text": count,
                "answer": count,
                "count": count,
                "question": (
                    f"{stem or 'How many different speakers are in the audio?'} "
                    "Output only a single integer."
                ),
                "source_index": str(MMSU_JSONL),
            }
        )
    n, path = _write(rows, "speaker_count_mmsu.jsonl")
    return n, path, {"missing_audio": missing, "skipped": skipped}


def main() -> None:
    n, path, meta = build_mmsu()
    print(json.dumps({"mmsu": {"n": n, "path": str(path), **meta}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
