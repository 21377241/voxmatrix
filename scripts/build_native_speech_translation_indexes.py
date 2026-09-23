#!/usr/bin/env python3
"""Build corrected Native speech-translation indexes."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVOST = Path("/mnt/afs/oss_data/datasets/07_speech_translation/CoVoST_2_zh_en_bidirectional")
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
NATIVE = ROOT / "mesh_eval/data/native"


def spread(rows, n=10):
    if len(rows) <= n:
        return rows
    return [rows[round(i * (len(rows) - 1) / (n - 1))] for i in range(n)]


def build_covost(source_lang, target_lang, filename, output_name):
    source = COVOST / filename
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        if doc.get("split") != "test":
            continue
        rel = (doc.get("audios") or [None])[0]
        wav = COVOST / rel if rel else None
        ref = str((doc.get("messages") or [{}, {}])[1].get("content") or "").strip()
        if not wav or not wav.is_file() or not ref:
            continue
        rows.append({
            "sample_id": doc.get("id"),
            "dataset": "covost2",
            "capability": "speech_translation",
            "protocol": "speech_translation",
            "subset": f"{source_lang}_{target_lang}_test",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "WavPath": str(wav),
            "question": f"Translate the {source_lang} speech into {target_lang}.",
            "text": ref,
            "answer": ref,
            "source_index": str(source),
        })
    out = NATIVE / output_name
    picked = spread(rows)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in picked), encoding="utf-8")
    return len(rows), len(picked), out


def build_mmsu():
    source = MMSU_ROOT / "mmsu_audioqa.jsonl"
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        if not str(doc.get("id", "")).startswith("MMSU_speech_translation_"):
            continue
        user = (doc.get("messages") or [{}, {}])[0].get("content") or [{}]
        assistant = (doc.get("messages") or [{}, {}])[1].get("content") or [{}]
        text = str(user[0].get("text") or "")
        choices = {m.group(1): m.group(2).strip() for m in re.finditer(r"\n([A-D])\.\s*(.*?)(?=\n[A-D]\.\s|$)", text, re.S)}
        wav = MMSU_ROOT / str(user[0].get("audio_path") or "")
        if len(choices) != 4 or not wav.is_file():
            continue
        answer = str(assistant[0].get("text") or "").strip()
        gold = next((k for k, v in choices.items() if v == answer), "")
        source_lang_match = re.search(r"translates the (\w+) audio", text, flags=re.I)
        source_lang = source_lang_match.group(1).lower() if source_lang_match else "unknown"
        rows.append({
            "sample_id": doc.get("id"),
            "dataset": "mmsu",
            "capability": "speech_translation",
            "protocol": "speech_translation",
            "source_lang": source_lang,
            "target_lang": "en",
            "WavPath": str(wav),
            "question": f"Listen to the {source_lang} audio and output its translation in English. Output only the translation.",
            "text": answer,
            "answer": answer,
            "source_index": str(source),
        })
    out = NATIVE / "speech_translation_mmsu.jsonl"
    picked = spread(rows)
    out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in picked), encoding="utf-8")
    return len(rows), len(picked), out


def main():
    print("covost zh-en:", build_covost("zh-CN", "en", "CoVoST_zh-CN_en_test_msswift_norm.jsonl", "speech_translation_covost2_zh_en.jsonl"))
    print("covost en-zh:", build_covost("en", "zh-CN", "CoVoST_en_zh-CN_test_msswift_norm.jsonl", "speech_translation_covost2_en_zh.jsonl"))
    print("mmsu mcq:", build_mmsu())


if __name__ == "__main__":
    main()
