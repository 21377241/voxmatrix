#!/usr/bin/env python3
"""Build full speech-translation indexes for CoVoST2 and open MMSU."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COVOST = Path("/mnt/afs/oss_data/datasets/07_speech_translation/CoVoST_2_zh_en_bidirectional")
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
OUT = ROOT / "mesh_eval/data/full/speech_translation"


def build_covost(source_lang, target_lang, filename, output_name):
    source = COVOST / filename
    rows = []
    seen = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        if doc.get("split") != "test":
            continue
        rel = (doc.get("audios") or [None])[0]
        wav = COVOST / rel if rel else None
        messages = doc.get("messages") or []
        ref = str(messages[1].get("content") if len(messages) > 1 else "").strip()
        sid = str(doc.get("id") or "")
        if not sid or sid in seen or not wav or not wav.is_file() or not ref:
            continue
        seen.add(sid)
        rows.append({
            "schema_version": "full-index/1.0",
            "sample_id": sid,
            "dataset": "covost2",
            "benchmark": "covost2",
            "capability": "speech_translation",
            "protocol": "speech_translation",
            "subset": f"{source_lang}_{target_lang}_test",
            "split": "test",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "WavPath": str(wav),
            "question": f"Translate the {source_lang} speech into {target_lang}. Output only the translation.",
            "text": ref,
            "answer": ref,
            "source_index": str(source),
        })
    path = OUT / output_name
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return len(rows), path


def build_mmsu():
    source = MMSU_ROOT / "mmsu_audioqa.jsonl"
    rows = []
    seen = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        if not str(doc.get("id", "")).startswith("MMSU_speech_translation_"):
            continue
        messages = doc.get("messages") or []
        user = messages[0].get("content") if messages else []
        assistant = messages[1].get("content") if len(messages) > 1 else []
        user = user[0] if user else {}
        assistant = assistant[0] if assistant else {}
        raw_question = str(user.get("text") or "")
        match = re.search(r"translates the (\w+) audio", raw_question, flags=re.I)
        source_lang = match.group(1).lower() if match else "unknown"
        wav = MMSU_ROOT / str(user.get("audio_path") or "")
        sid = str(doc.get("id") or "")
        ref = str(assistant.get("text") or "").strip()
        if not sid or sid in seen or not wav.is_file() or not ref:
            continue
        seen.add(sid)
        rows.append({
            "schema_version": "full-index/1.0",
            "sample_id": sid,
            "dataset": "mmsu",
            "benchmark": "mmsu",
            "capability": "speech_translation",
            "protocol": "speech_translation",
            "subset": "speech_translation",
            "split": "test",
            "source_lang": source_lang,
            "target_lang": "en",
            "WavPath": str(wav),
            "question": f"Listen to the {source_lang} audio and output its translation in English. Output only the translation.",
            "text": ref,
            "answer": ref,
            "source_index": str(source),
        })
    path = OUT / "speech_translation_mmsu.jsonl"
    path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return len(rows), path


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for key, args in {
        "covost2_zh_en": ("zh-CN", "en", "CoVoST_zh-CN_en_test_msswift_norm.jsonl", "speech_translation_covost2_zh_en.jsonl"),
        "covost2_en_zh": ("en", "zh-CN", "CoVoST_en_zh-CN_test_msswift_norm.jsonl", "speech_translation_covost2_en_zh.jsonl"),
    }.items():
        n, path = build_covost(*args)
        summary[key] = {"records": n, "output": str(path)}
    n, path = build_mmsu()
    summary["mmsu"] = {"records": n, "output": str(path)}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
