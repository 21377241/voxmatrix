#!/usr/bin/env python3
"""Build 10-sample speech_translation smoke manifests (covost2 + mmsu)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = 10

COVOST_ROOT = Path(
    "/mnt/afs/oss_data/datasets/07_speech_translation/CoVoST_2_zh_en_bidirectional"
)
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_INDEX = MMSU_ROOT / "mmsu_audioqa.jsonl"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_stride(rows: list[dict], n: int = N) -> list[dict]:
    if len(rows) <= n:
        return rows
    step = max(1, len(rows) // n)
    picked = [rows[i * step] for i in range(n)]
    return picked[:n]


def build_covost_zh_en() -> list[dict]:
    jsonl = COVOST_ROOT / "CoVoST_zh-CN_en_test_msswift_norm.jsonl"
    rows: list[dict] = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            rel = row["audios"][0]
            wav = COVOST_ROOT / rel
            if not wav.is_file():
                continue
            ref = row["messages"][1]["content"].strip()
            rows.append(
                {
                    "WavPath": str(wav),
                    "text": ref,
                    "sample_id": row["id"],
                    "dataset": "covost2",
                    "subset": "zh-CN_en_test",
                    "source_lang": "zh-CN",
                    "target_lang": "en",
                }
            )
    picked = sample_stride(rows)
    if len(picked) < N:
        raise RuntimeError(f"covost2 zh-en: only {len(picked)} valid samples")
    return picked


def build_covost_en_zh() -> list[dict]:
    jsonl = COVOST_ROOT / "CoVoST_en_zh-CN_test_msswift_norm.jsonl"
    rows: list[dict] = []
    with jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("split") != "test":
                continue
            rel = row["audios"][0]
            wav = COVOST_ROOT / rel
            if not wav.is_file():
                continue
            ref = row["messages"][1]["content"].strip()
            rows.append(
                {
                    "WavPath": str(wav),
                    "text": ref,
                    "sample_id": row["id"],
                    "dataset": "covost2",
                    "subset": "en_zh-CN_test",
                    "source_lang": "en",
                    "target_lang": "zh-CN",
                }
            )
    picked = sample_stride(rows)
    if len(picked) < N:
        raise RuntimeError(f"covost2 en-zh: only {len(picked)} valid samples")
    return picked


def _parse_mmsu_question(text: str) -> tuple[str, dict[str, str]]:
    parts = re.split(r"\n(?=[A-D]\.\s)", text.strip())
    question = parts[0].strip()
    choices: dict[str, str] = {}
    for part in parts[1:]:
        match = re.match(r"^([A-D])\.\s*(.*)$", part.strip(), flags=re.S)
        if not match:
            continue
        letter, body = match.group(1), match.group(2).strip()
        choices[letter] = " ".join(body.split())
    return question, choices


def _mmsu_source_lang(question: str) -> str:
    match = re.search(r"translates the (\w+) audio", question, flags=re.I)
    return match.group(1).lower() if match else "unknown"


def _iter_mmsu_speech_translation_rows() -> list[dict]:
    rows: list[dict] = []
    with MMSU_INDEX.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["id"].startswith("MMSU_speech_translation_"):
                continue
            user = row["messages"][0]["content"][0]
            assistant = row["messages"][1]["content"][0]
            rel = user["audio_path"]
            wav = MMSU_ROOT / rel
            if not wav.is_file():
                continue
            question, choices = _parse_mmsu_question(user["text"])
            if len(choices) < 4:
                continue
            answer = assistant["text"].strip()
            rows.append(
                {
                    "WavPath": str(wav),
                    "text": answer,
                    "question": question,
                    "choice_a": choices["A"],
                    "choice_b": choices["B"],
                    "choice_c": choices["C"],
                    "choice_d": choices["D"],
                    "sample_id": row["id"],
                    "dataset": "mmsu",
                    "subset": "speech_translation",
                    "source_lang": _mmsu_source_lang(question),
                    "target_lang": "en",
                    "protocol": "mcq",
                }
            )
    return rows


def _pick_mmsu_rows(rows: list[dict], sample_ids: list[str] | None = None) -> list[dict]:
    if sample_ids:
        by_id = {row["sample_id"]: row for row in rows}
        missing = [sid for sid in sample_ids if sid not in by_id]
        if missing:
            raise RuntimeError(f"mmsu missing sample_ids: {missing[:3]}")
        return [by_id[sid] for sid in sample_ids]
    picked = sample_stride(rows)
    if len(picked) < N:
        raise RuntimeError(f"mmsu speech_translation: only {len(picked)} valid samples")
    return picked


def _existing_mmsu_sample_ids() -> list[str] | None:
    manifest = ROOT / "mmsu" / "manifest.jsonl"
    if not manifest.is_file():
        return None
    ids: list[str] = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            ids.append(json.loads(line)["sample_id"])
    return ids or None


def build_mmsu_speech_translation() -> list[dict]:
    return _pick_mmsu_rows(_iter_mmsu_speech_translation_rows(), _existing_mmsu_sample_ids())


def build_mmsu_speech_translation_open() -> list[dict]:
    base = _pick_mmsu_rows(
        _iter_mmsu_speech_translation_rows(), _existing_mmsu_sample_ids()
    )
    return [
        {
            "WavPath": row["WavPath"],
            "text": row["text"],
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "subset": row["subset"],
            "source_lang": row["source_lang"],
            "target_lang": row["target_lang"],
            "protocol": "open_s2tt",
        }
        for row in base
    ]


def main() -> None:
    datasets = {
        "covost2_zh_en": build_covost_zh_en(),
        "covost2_en_zh": build_covost_en_zh(),
        "mmsu": build_mmsu_speech_translation(),
        "mmsu_open": build_mmsu_speech_translation_open(),
    }
    summary = {
        "capability": "speech_translation",
        "count": len(datasets),
        "samples_per_benchmark": N,
        "datasets": [],
    }
    for name, rows in datasets.items():
        manifest = ROOT / name / "manifest.jsonl"
        write_jsonl(manifest, rows)
        summary["datasets"].append(
            {"name": name, "manifest": str(manifest), "n": len(rows)}
        )
        print(f"[ok] {name}: {len(rows)} -> {manifest}")
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
