#!/usr/bin/env python3
"""Build full code-switch indexes (adapted mixed-language QA).

Tracks align with Native catalog:
  - MMSU code_switch_question_answering (MCQ options stripped)
  - VocalBench-zh knowledge (cs_knowledge.json)
  - VocalBench-zh open_ended (cs_open_ended.json)
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/code_switch"
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_JSONL = MMSU_ROOT / "mmsu_audioqa.jsonl"
VBZH_ROOT = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh")


def _strip_mcq_options(question: str) -> str:
    question = re.split(r"\nA\.\s", question, maxsplit=1)[0].strip()
    return (
        f"{question}\n\nAnswer the question using the audio. Output only the answer."
    )


def _parse_mmsu_messages(doc: dict) -> tuple[str | None, str, str]:
    messages = doc.get("messages") or []
    if len(messages) < 2:
        return None, "", ""
    user = messages[0].get("content") or []
    assistant = messages[1].get("content") or []
    user0 = user[0] if user else {}
    asst0 = assistant[0] if assistant else {}
    rel = str(user0.get("audio_path") or "").strip() or None
    question = str(user0.get("text") or "").strip()
    answer = str(asst0.get("text") or "").strip()
    return rel, question, answer


def build_mmsu() -> tuple[int, Path]:
    rows = []
    seen: set[str] = set()
    for line in MMSU_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        if not sid.startswith("MMSU_code_switch_question_answering_"):
            continue
        if sid in seen:
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
        seen.add(sid)
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "mmsu",
                "benchmark": "mmsu",
                "capability": "code_switch",
                "protocol": "code_switch",
                "subset": "code_switch_question_answering",
                "split": "test",
                "WavPath": str(wav),
                "question": _strip_mcq_options(question),
                "text": answer,
                "answer": answer,
                "source_index": str(MMSU_JSONL),
            }
        )
    path = OUT / "code_switch_mmsu.jsonl"
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    return len(rows), path


def build_vocalbench(
    json_name: str,
    subset: str,
    output_name: str,
) -> tuple[int, Path]:
    source = VBZH_ROOT / "json" / json_name
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = []
    seen: set[str] = set()
    for item in data:
        rel = str(item.get("Audio") or item.get("audio") or "").strip()
        candidates = [VBZH_ROOT / "audio" / rel, VBZH_ROOT / rel]
        wav = next((p for p in candidates if p.is_file()), None)
        answer = str(
            item.get("Answer")
            or item.get("Answer_zh")
            or item.get("Answer_en")
            or item.get("answer")
            or ""
        ).strip()
        question = str(
            item.get("Question")
            or item.get("Question_zh")
            or item.get("question")
            or ""
        ).strip()
        sid = str(item.get("Qid") or item.get("id") or "").strip()
        if not sid or sid in seen or wav is None or not answer:
            continue
        seen.add(sid)
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "vocalbench_zh",
                "benchmark": "vocalbench_zh",
                "capability": "code_switch",
                "protocol": "code_switch",
                "subset": subset,
                "split": "test",
                "WavPath": str(wav),
                "question": question,
                "text": answer,
                "answer": answer,
                "source_index": str(source),
            }
        )
    path = OUT / output_name
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    return len(rows), path


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    n, path = build_mmsu()
    summary["mmsu"] = {"records": n, "output": str(path)}
    n, path = build_vocalbench(
        "cs_knowledge.json",
        "code_switch_knowledge",
        "code_switch_vocalbench_zh.jsonl",
    )
    summary["vocalbench_zh_knowledge"] = {"records": n, "output": str(path)}
    n, path = build_vocalbench(
        "cs_open_ended.json",
        "code_switch_open_ended",
        "code_switch_vocalbench_zh_open_ended.jsonl",
    )
    summary["vocalbench_zh_open_ended"] = {"records": n, "output": str(path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    missing = [k for k, v in summary.items() if v["records"] <= 0]
    if missing:
        raise SystemExit(f"empty full indexes: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
