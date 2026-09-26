#!/usr/bin/env python3
"""Build full QA indexes aligned with Native catalog slots.

Tracks (all require gold + readable WavPath):
  - EAR wdyl nokw competence
  - Spoken-SQuAD test_wer44
  - URO-Bench basic/* with target_text
  - VocalBench knowledge
  - VocalBench-zh chinese_knowledge
  - VoiceBench openbookqa (has reference + exported wav)
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/qa"

EAR_MANIFEST = Path("/mnt/afs/eval_data/benchmarks/EAR/manifests/ear_wdyl.jsonl")
SS_META = Path(
    "/mnt/afs/oss_data/datasets/04_dialogue_slu/Spoken-SQuAD/output/metadata_test_wer44.jsonl"
)
SS_ROOT = Path("/mnt/afs/oss_data/datasets/04_dialogue_slu/Spoken-SQuAD")
URO_BASIC = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/URO-Bench/basic")
VB_ROOT = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench")
VBZH_ROOT = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh")
VB_PARQUET = Path(
    "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench"
)
VB_WAV = Path(
    "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench_wav/wav"
)

EAR_QUESTION = (
    "What is the word obscured by the white noise in this audio? "
    "Answer with one word only."
)


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    return len(rows), path


def _answer_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if x is not None and str(x).strip()]
        return parts[0] if parts else ""
    if isinstance(value, dict):
        texts = value.get("text")
        if isinstance(texts, list) and texts:
            return str(texts[0]).strip()
        if texts not in (None, ""):
            return str(texts).strip()
    text = str(value).strip()
    if text.startswith("{") and "text" in text:
        try:
            parsed = ast.literal_eval(text)
            return _answer_text(parsed)
        except (SyntaxError, ValueError):
            pass
    return text


def build_ear() -> tuple[int, Path]:
    rows = []
    for line in EAR_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if doc.get("condition") != "nokw":
            continue
        if doc.get("capability_nokw") not in (None, "qa"):
            continue
        wav = Path(str(doc.get("audio") or doc.get("WavPath") or ""))
        answer = str(doc.get("masked_word") or doc.get("answer") or "").strip()
        sid = str(doc.get("id") or doc.get("sample_id") or "").strip()
        if not sid or not answer or not wav.is_file():
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "ear_wdyl",
                "benchmark": "ear_wdyl",
                "capability": "qa",
                "protocol": "qa",
                "subset": "wdyl_nokw_competence",
                "split": "test",
                "qa_track": "short",
                "WavPath": str(wav),
                "question": EAR_QUESTION,
                "answer": answer,
                "text": answer,
                "source_index": str(EAR_MANIFEST),
            }
        )
    return _write(rows, "qa_ear_wdyl.jsonl")


def build_spoken_squad() -> tuple[int, Path]:
    rows = []
    for line in SS_META.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        rel = str(doc.get("audio") or "").strip()
        wav = SS_ROOT / rel if rel else Path()
        if not wav.is_file():
            continue
        answer = _answer_text(doc.get("answer"))
        question = str(doc.get("question") or "").strip()
        sid = str(doc.get("source_id") or doc.get("id") or "").strip()
        if not sid or not answer or not question:
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "spoken_squad",
                "benchmark": "spoken_squad",
                "capability": "qa",
                "protocol": "qa",
                "subset": "test_wer44",
                "split": "test",
                "qa_track": "short",
                "WavPath": str(wav),
                "question": question,
                "answer": answer,
                "text": answer,
                "source_index": str(SS_META),
            }
        )
    return _write(rows, "qa_spoken_squad.jsonl")


def build_uro() -> tuple[int, Path]:
    rows = []
    seen: set[str] = set()
    for subset_dir in sorted(URO_BASIC.iterdir()):
        if not subset_dir.is_dir():
            continue
        for path in sorted(subset_dir.glob("test*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                doc = json.loads(line)
                answer = _answer_text(
                    doc.get("target_text")
                    or doc.get("answer")
                    or doc.get("output")
                    or doc.get("text")
                )
                if not answer:
                    continue
                audio = (
                    doc.get("source_wav")
                    or doc.get("audio")
                    or doc.get("WavPath")
                    or doc.get("audio_path")
                )
                if not audio:
                    continue
                wav = Path(audio) if Path(str(audio)).is_absolute() else (subset_dir / audio)
                if not wav.is_file():
                    continue
                question = str(
                    doc.get("source_text")
                    or doc.get("question")
                    or doc.get("instruction")
                    or ""
                ).strip()
                if not question:
                    continue
                sid = str(doc.get("id") or f"{len(rows):05d}")
                sid = f"{subset_dir.name}_{sid}"
                if sid in seen:
                    continue
                seen.add(sid)
                rows.append(
                    {
                        "schema_version": "full-index/1.0",
                        "sample_id": sid,
                        "dataset": "uro_bench",
                        "benchmark": "uro_bench",
                        "capability": "qa",
                        "protocol": "qa",
                        "subset": f"basic/{subset_dir.name}",
                        "split": "test",
                        "qa_track": "short",
                        "WavPath": str(wav),
                        "question": question,
                        "answer": answer,
                        "text": answer,
                        "source_index": str(path),
                    }
                )
    return _write(rows, "qa_uro_bench.jsonl")


def build_vocalbench_json(
    root: Path,
    json_name: str,
    dataset: str,
    subset: str,
    output_name: str,
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
                "capability": "qa",
                "protocol": "qa",
                "subset": subset,
                "split": "test",
                "qa_track": "short",
                "WavPath": str(wav),
                "question": question,
                "answer": answer,
                "text": answer,
                "source_index": str(source),
            }
        )
    return _write(rows, output_name)


def build_voicebench_openbookqa() -> tuple[int, Path]:
    import pyarrow.parquet as pq

    subset = "openbookqa"
    shard = "test-00000-of-00001"
    path = VB_PARQUET / subset / f"{shard}.parquet"
    table = pq.read_table(path)
    rows = []
    for i in range(table.num_rows):
        item = {col: table.column(col)[i].as_py() for col in table.column_names}
        answer = _answer_text(item.get("reference") or item.get("answer"))
        question = str(item.get("prompt") or item.get("question") or "").strip()
        if not answer or not question:
            continue
        wav = VB_WAV / subset / shard / f"{shard}_row{i}.wav"
        if not wav.is_file():
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": f"voicebench_{subset}_{i}",
                "dataset": "voicebench",
                "benchmark": "voicebench",
                "capability": "qa",
                "protocol": "qa",
                "subset": subset,
                "split": "test",
                "qa_track": "short",
                "WavPath": str(wav),
                "question": question,
                "answer": answer,
                "text": answer,
                "source_index": str(path),
            }
        )
    return _write(rows, "qa_voicebench.jsonl")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    n, path = build_ear()
    summary["ear_wdyl"] = {"records": n, "output": str(path)}
    n, path = build_spoken_squad()
    summary["spoken_squad"] = {"records": n, "output": str(path)}
    n, path = build_uro()
    summary["uro_bench"] = {"records": n, "output": str(path)}
    n, path = build_vocalbench_json(
        VB_ROOT, "knowledge.json", "vocalbench", "knowledge", "qa_vocalbench.jsonl"
    )
    summary["vocalbench"] = {"records": n, "output": str(path)}
    n, path = build_vocalbench_json(
        VBZH_ROOT,
        "chinese_knowledge.json",
        "vocalbench_zh",
        "chinese_knowledge",
        "qa_vocalbench_zh.jsonl",
    )
    summary["vocalbench_zh"] = {"records": n, "output": str(path)}
    n, path = build_voicebench_openbookqa()
    summary["voicebench"] = {"records": n, "output": str(path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    missing = [k for k, v in summary.items() if v["records"] <= 0]
    if missing:
        raise SystemExit(f"empty full indexes: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
