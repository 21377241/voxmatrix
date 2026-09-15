#!/usr/bin/env python3
"""Build a stratified 9-cell semantic-judge pack (target n>=30 per cell).

Cells:
  qa × {open, semi-open, binary}
  audio_reasoning × {open, semi-open}
  speech_translation × {translation}
  code_switch × {binary, semi-open, open}

Writes JSONL with WavPath/question/reference/rubric (pred may be empty until undertest).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

URO = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/URO-Bench/basic")
VB = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench")
VBZH = Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh")
MMSU = Path("/mnt/afs/eval_data/benchmarks/MMSU/mmsu_audioqa.jsonl")
COVOST = Path(
    "/mnt/afs/oss_data/datasets/07_speech_translation/CoVoST_2_zh_en_bidirectional"
)

CELLS = [
    ("qa", "open"),
    ("qa", "semi-open"),
    ("qa", "binary"),
    ("audio_reasoning", "open"),
    ("audio_reasoning", "semi-open"),
    ("speech_translation", "translation"),
    ("code_switch", "binary"),
    ("code_switch", "semi-open"),
    ("code_switch", "open"),
]


def _exists(path: Path) -> bool:
    return path.is_file()


def _uro_rows(subset: str):
    for name in ("test.jsonl", "test_mini.jsonl"):
        path = URO / subset / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line), path, subset
        return
    return
    yield  # pragma: no cover — make this a generator even when empty


def _resolve_uro_wav(subset: str, source_wav: str) -> Optional[str]:
    rel = source_wav.lstrip("./")
    cand = URO / subset / rel
    if cand.is_file():
        return str(cand)
    # some rows store audio/xxx.wav already under subset
    if Path(source_wav).is_file():
        return source_wav
    return None


def _vb_json(root: Path, name: str) -> List[Dict[str, Any]]:
    path = root / "json" / name
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def _vb_wav(root: Path, audio_rel: str) -> Optional[str]:
    cand = root / "audio" / audio_rel
    if cand.is_file():
        return str(cand)
    if Path(audio_rel).is_file():
        return audio_rel
    return None


def _mmsu_rows(prefix: str) -> Iterable[Dict[str, Any]]:
    if not MMSU.is_file():
        return
    with MMSU.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            sid = str(obj.get("id") or "")
            if not sid.startswith(prefix):
                continue
            yield obj


def _mmsu_fields(obj: Dict[str, Any]) -> Tuple[str, str, str, str]:
    sid = str(obj.get("id") or "")
    question = ""
    answer = ""
    wav = ""
    for msg in obj.get("messages") or []:
        role = msg.get("role")
        for part in msg.get("content") or []:
            if not isinstance(part, dict):
                continue
            if role == "user":
                if part.get("text"):
                    question = str(part["text"])
                if part.get("audio_path"):
                    wav = str(part["audio_path"])
            elif role == "assistant" and part.get("text"):
                answer = str(part["text"])
    # resolve wav under MMSU root
    root = MMSU.parent
    for cand in (root / wav, root / "audio" / Path(wav).name, Path(wav)):
        if cand.is_file():
            return sid, question, answer, str(cand)
    return sid, question, answer, ""


def _covost_rows(direction: str, limit: int) -> List[Dict[str, Any]]:
    if direction == "en_zh":
        path = COVOST / "CoVoST_en_zh-CN_test_msswift_norm.jsonl"
        src_lang, tgt_lang = "en", "zh"
        direction_text = "Translate into Chinese"
    else:
        path = COVOST / "CoVoST_zh-CN_en_test_msswift_norm.jsonl"
        src_lang, tgt_lang = "zh", "en"
        direction_text = "Translate into English"
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(out) >= limit:
                break
            obj = json.loads(line)
            audios = obj.get("audios") or []
            if not audios:
                continue
            wav = COVOST / audios[0]
            if not wav.is_file():
                continue
            # assistant target text if present
            ref = ""
            for msg in obj.get("messages") or []:
                if msg.get("role") == "assistant":
                    content = msg.get("content")
                    if isinstance(content, str):
                        ref = content
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("text"):
                                ref = str(part["text"])
            out.append(
                {
                    "sample_id": str(obj.get("id") or wav.stem),
                    "capability": "speech_translation",
                    "rubric": "translation",
                    "WavPath": str(wav),
                    "question": direction_text,
                    "reference": ref,
                    "pred": "",
                    "lang": tgt_lang,
                    "direction": direction_text,
                    "source_lang": src_lang,
                    "oracle_transcript": "",
                    "source_dataset": str(path),
                }
            )
    return out


def _sample(rows: List[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    if len(rows) <= n:
        return list(rows)
    return rng.sample(rows, n)


def _seed_from_existing_preds(capability: str, rubric: str) -> List[Dict[str, Any]]:
    """Prefer smoke/compare-pack rows that already have preds."""
    compare = Path(
        "/mnt/afs/users/wangyl/VoxMatrix/scripts/data/semantic_judge_compare_pack.jsonl"
    )
    rows: List[Dict[str, Any]] = []
    if not compare.is_file():
        return rows
    with compare.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("capability") != capability or obj.get("rubric") != rubric:
                continue
            if not str(obj.get("pred") or "").strip():
                continue
            wav = obj.get("WavPath")
            if not wav or not Path(str(wav)).is_file():
                continue
            rows.append(
                {
                    "sample_id": obj["sample_id"],
                    "capability": capability,
                    "rubric": rubric,
                    "WavPath": str(wav),
                    "question": obj.get("question") or "",
                    "reference": obj.get("reference") or "",
                    "pred": obj.get("pred") or "",
                    "lang": obj.get("lang") or "en",
                    "direction": obj.get("direction") or "",
                    "oracle_transcript": obj.get("question") or "",
                    "source_dataset": "compare_pack_seed",
                    "pred_source": "seed_existing",
                }
            )
    return rows


def build_cell(
    capability: str,
    rubric: str,
    n: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = _seed_from_existing_preds(capability, rubric)
    seeded_ids = {r["sample_id"] for r in rows}

    if capability == "qa" and rubric == "open":
        for subset in ("WildchatEval", "AlpacaEval", "Wildchat-zh", "AlpacaEval-zh", "Claude-zh"):
            for obj, _path, sub in _uro_rows(subset):
                wav = _resolve_uro_wav(sub, str(obj.get("source_wav") or ""))
                if not wav:
                    continue
                rows.append(
                    {
                        "sample_id": f"uro-{sub}-{obj.get('id')}",
                        "capability": "qa",
                        "rubric": "open",
                        "WavPath": wav,
                        "question": str(obj.get("source_text") or ""),
                        "reference": "",
                        "pred": "",
                        "lang": "zh" if sub.endswith("-zh") or sub.endswith("zh") else "en",
                        "direction": "",
                        "oracle_transcript": str(obj.get("source_text") or ""),
                        "source_dataset": sub,
                    }
                )
        for item in _vb_json(VB, "single_round.json"):
            wav = _vb_wav(VB, str(item.get("Audio") or ""))
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": str(item.get("Qid") or item.get("Audio")),
                    "capability": "qa",
                    "rubric": "open",
                    "WavPath": wav,
                    "question": str(item.get("Question") or ""),
                    "reference": str(item.get("Answer") or ""),
                    "pred": "",
                    "lang": "en",
                    "direction": "",
                    "oracle_transcript": str(item.get("Question") or ""),
                    "source_dataset": "vocalbench_single_round",
                }
            )

    elif capability == "qa" and rubric == "semi-open":
        for subset in ("Summary", "LCSTS-zh", "StoralEval", "TruthfulEval"):
            for obj, _path, sub in _uro_rows(subset):
                wav = _resolve_uro_wav(sub, str(obj.get("source_wav") or ""))
                if not wav:
                    continue
                rows.append(
                    {
                        "sample_id": f"uro-{sub}-{obj.get('id')}",
                        "capability": "qa",
                        "rubric": "semi-open",
                        "WavPath": wav,
                        "question": str(obj.get("source_text") or ""),
                        "reference": str(obj.get("target_text") or ""),
                        "pred": "",
                        "lang": "zh" if "zh" in sub.lower() else "en",
                        "direction": "",
                        "oracle_transcript": str(obj.get("source_text") or ""),
                        "source_dataset": sub,
                    }
                )

    elif capability == "qa" and rubric == "binary":
        for subset in ("Gsm8kEval", "GaokaoEval", "OpenbookQA-zh", "SQuAD-zh", "HSK5-zh"):
            for obj, _path, sub in _uro_rows(subset):
                wav = _resolve_uro_wav(sub, str(obj.get("source_wav") or ""))
                if not wav:
                    continue
                rows.append(
                    {
                        "sample_id": f"uro-{sub}-{obj.get('id')}",
                        "capability": "qa",
                        "rubric": "binary",
                        "WavPath": wav,
                        "question": str(obj.get("source_text") or ""),
                        "reference": str(obj.get("target_text") or ""),
                        "pred": "",
                        "lang": "zh" if "zh" in sub.lower() else "en",
                        "direction": "",
                        "oracle_transcript": str(obj.get("source_text") or ""),
                        "source_dataset": sub,
                    }
                )
        for item in _vb_json(VB, "knowledge.json"):
            wav = _vb_wav(VB, str(item.get("Audio") or ""))
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": str(item.get("Qid") or item.get("Audio")),
                    "capability": "qa",
                    "rubric": "binary",
                    "WavPath": wav,
                    "question": str(item.get("Question") or ""),
                    "reference": str(item.get("Answer") or ""),
                    "pred": "",
                    "lang": "en",
                    "direction": "",
                    "oracle_transcript": str(item.get("Question") or ""),
                    "source_dataset": "vocalbench_knowledge",
                }
            )

    elif capability == "audio_reasoning" and rubric == "semi-open":
        for root, lang, ds in (
            (VB, "en", "vocalbench_reasoning"),
            (VBZH, "zh", "vocalbench_zh_reasoning"),
        ):
            for item in _vb_json(root, "reasoning.json"):
                wav = _vb_wav(root, str(item.get("Audio") or ""))
                if not wav:
                    continue
                rows.append(
                    {
                        "sample_id": str(item.get("Qid") or item.get("Audio")),
                        "capability": "audio_reasoning",
                        "rubric": "semi-open",
                        "WavPath": wav,
                        "question": str(item.get("Question") or ""),
                        "reference": str(item.get("Answer") or ""),
                        "pred": "",
                        "lang": lang,
                        "direction": "",
                        "oracle_transcript": str(item.get("Question") or ""),
                        "source_dataset": ds,
                    }
                )
        for obj in _mmsu_rows("MMSU_reasoning") or []:
            sid, q, a, wav = _mmsu_fields(obj)
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "capability": "audio_reasoning",
                    "rubric": "semi-open",
                    "WavPath": wav,
                    "question": q,
                    "reference": a,
                    "pred": "",
                    "lang": "en",
                    "direction": "",
                    "oracle_transcript": q,
                    "source_dataset": "mmsu_reasoning",
                }
            )

    elif capability == "audio_reasoning" and rubric == "open":
        # Protocol: same reasoning pool but strip hard reference for open rubric.
        for item in _vb_json(VB, "reasoning.json"):
            wav = _vb_wav(VB, str(item.get("Audio") or ""))
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": f"ar-open-{item.get('Qid') or item.get('Audio')}",
                    "capability": "audio_reasoning",
                    "rubric": "open",
                    "WavPath": wav,
                    "question": str(item.get("Question") or ""),
                    "reference": "",
                    "pred": "",
                    "lang": "en",
                    "direction": "",
                    "oracle_transcript": str(item.get("Question") or ""),
                    "source_dataset": "vocalbench_reasoning_open_protocol",
                    "note": "reference stripped for open rubric",
                }
            )

    elif capability == "speech_translation" and rubric == "translation":
        rows.extend(_covost_rows("en_zh", n * 2))
        rows.extend(_covost_rows("zh_en", n * 2))
        for obj in _mmsu_rows("MMSU_speech_translation") or []:
            sid, q, a, wav = _mmsu_fields(obj)
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "capability": "speech_translation",
                    "rubric": "translation",
                    "WavPath": wav,
                    "question": q or "Translate into English",
                    "reference": a,
                    "pred": "",
                    "lang": "en",
                    "direction": "Translate into English",
                    "oracle_transcript": "",
                    "source_dataset": "mmsu_speech_translation",
                }
            )

    elif capability == "code_switch" and rubric == "open":
        for item in _vb_json(VBZH, "cs_open_ended.json"):
            wav = _vb_wav(VBZH, str(item.get("Audio") or ""))
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": str(item.get("Qid") or item.get("Audio")),
                    "capability": "code_switch",
                    "rubric": "open",
                    "WavPath": wav,
                    "question": str(item.get("Question") or ""),
                    "reference": str(item.get("Answer") or ""),
                    "pred": "",
                    "lang": "mix",
                    "direction": "",
                    "oracle_transcript": str(item.get("Question") or ""),
                    "source_dataset": "vocalbench_zh_cs_open_ended",
                }
            )

    elif capability == "code_switch" and rubric == "binary":
        for item in _vb_json(VBZH, "cs_knowledge.json"):
            wav = _vb_wav(VBZH, str(item.get("Audio") or ""))
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": str(item.get("Qid") or item.get("Audio")),
                    "capability": "code_switch",
                    "rubric": "binary",
                    "WavPath": wav,
                    "question": str(item.get("Question") or ""),
                    "reference": str(item.get("Answer") or ""),
                    "pred": "",
                    "lang": "mix",
                    "direction": "",
                    "oracle_transcript": str(item.get("Question") or ""),
                    "source_dataset": "vocalbench_zh_cs_knowledge",
                }
            )
        for obj in _mmsu_rows("MMSU_code_switch") or []:
            sid, q, a, wav = _mmsu_fields(obj)
            if not wav:
                continue
            rows.append(
                {
                    "sample_id": sid,
                    "capability": "code_switch",
                    "rubric": "binary",
                    "WavPath": wav,
                    "question": q,
                    "reference": a,
                    "pred": "",
                    "lang": "mix",
                    "direction": "",
                    "oracle_transcript": q,
                    "source_dataset": "mmsu_code_switch",
                }
            )

    elif capability == "code_switch" and rubric == "semi-open":
        # Soft-ref CS: MMSU CS answers as soft references under semi-open rubric.
        for obj in _mmsu_rows("MMSU_code_switch") or []:
            sid, q, a, wav = _mmsu_fields(obj)
            if not wav or not a:
                continue
            rows.append(
                {
                    "sample_id": f"cs-semi-{sid}",
                    "capability": "code_switch",
                    "rubric": "semi-open",
                    "WavPath": wav,
                    "question": q,
                    "reference": a,
                    "pred": "",
                    "lang": "mix",
                    "direction": "",
                    "oracle_transcript": q,
                    "source_dataset": "mmsu_code_switch_semi_open_protocol",
                }
            )
        for item in _vb_json(VBZH, "cs_open_ended.json"):
            wav = _vb_wav(VBZH, str(item.get("Audio") or ""))
            ans = str(item.get("Answer") or "").strip()
            if not wav or not ans:
                continue
            rows.append(
                {
                    "sample_id": f"cs-semi-{item.get('Qid') or item.get('Audio')}",
                    "capability": "code_switch",
                    "rubric": "semi-open",
                    "WavPath": wav,
                    "question": str(item.get("Question") or ""),
                    "reference": ans,
                    "pred": "",
                    "lang": "mix",
                    "direction": "",
                    "oracle_transcript": str(item.get("Question") or ""),
                    "source_dataset": "vocalbench_zh_cs_open_ended_semi_open",
                }
            )

    # filter missing wav
    rows = [r for r in rows if r.get("WavPath") and Path(r["WavPath"]).is_file()]
    # dedupe by sample_id; keep seeded preds first
    seen = set()
    uniq = []
    for r in rows:
        sid = r["sample_id"]
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(r)
    # Prefer keeping all seeded rows, then sample the rest to fill n
    seeded = [r for r in uniq if r.get("pred_source") == "seed_existing"]
    others = [r for r in uniq if r.get("pred_source") != "seed_existing"]
    need = max(0, n - len(seeded))
    picked = list(seeded) + _sample(others, need, rng)
    return picked[:n]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "/mnt/afs/users/wangyl/VoxMatrix/scripts/data/semantic_judge_stratified_pack.jsonl"
        ),
    )
    args = parser.parse_args()
    rng = random.Random(args.seed)
    pack: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for cap, rub in CELLS:
        rows = build_cell(cap, rub, args.per_cell, rng)
        counts[f"{cap}|{rub}"] = len(rows)
        pack.extend(rows)
        print(f"{cap}|{rub}: {len(rows)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in pack:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "n": len(pack),
        "per_cell_target": args.per_cell,
        "counts": counts,
        "seed": args.seed,
        "out": str(args.out),
    }
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    missing = [k for k, v in counts.items() if v < args.per_cell]
    if missing:
        print("WARNING underfilled:", missing)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
