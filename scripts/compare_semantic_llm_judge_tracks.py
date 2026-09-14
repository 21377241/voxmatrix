#!/usr/bin/env python3
"""Three-track semantic LLM-judge comparison: A0 (no ASR) / A1 (ASR) / B (raw audio).

Builds a small real-sample pack from existing smoke preds+manifests, optionally
runs local ASR for A1, judges each track, and writes score + usage summary.

Requires OPENAI_API_KEY (+ optional OPENAI_BASE_URL). Does not print secrets.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = Path("/tmp/semantic_judge_track_compare.json")
DEFAULT_PACK = ROOT / "scripts" / "data" / "semantic_judge_compare_pack.jsonl"
DEFAULT_ASR_CACHE = Path("/tmp/semantic_judge_asr_cache")
DELIVER_MD = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_三路Judge对比报告.md")


def _load_eval_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("type") == "eval":
                rows.append(obj["data"])
            elif "pred" in obj:
                rows.append(obj)
    return rows


def _load_manifest_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid = str(row.get("sample_id") or "")
            if sid:
                out[sid] = row
    return out


def _ref_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(x) for x in value if str(x).strip())
    return str(value).strip()


def build_pack(limit_per_source: int = 4) -> List[Dict[str, Any]]:
    """Assemble real samples with WavPath + pred from known smoke runs."""
    sources = [
        {
            "capability": "qa",
            "lang": "en",
            "results": Path(
                "/mnt/afs/users/wangyl/8p31/runs/qa_uro_bench_smoke/"
                "cluster_run_20260911_062245/results.jsonl"
            ),
            "manifest": Path(
                "/mnt/afs/users/wangyl/8p31/runs/qa_uro_bench_smoke/qa_uro_bench_10.jsonl"
            ),
            "allow_modes": {"open", "semi-open", "qa"},
        },
        {
            "capability": "audio_reasoning",
            "lang": "en",
            "results": Path(
                "/mnt/afs/users/wangyl/8p31/runs/audio_reasoning_vocalbench_smoke/"
                "cluster_run_20260831_104516/results.jsonl"
            ),
            "manifest": Path(
                "/mnt/afs/users/wangyl/8p31/runs/audio_reasoning_vocalbench_smoke/"
                "audio_reasoning_vocalbench_10.jsonl"
            ),
            "allow_modes": None,
        },
        {
            "capability": "speech_translation",
            "lang": "en",
            "results": Path(
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_st_all/results_mmsu_open/"
                "mmsu_open.jsonl"
            ),
            "manifest": Path(
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_st_all/mmsu_open/manifest.jsonl"
            ),
            "allow_modes": None,
            "rubric": "translation",
            "direction": "Translate into English",
        },
        {
            "capability": "code_switch",
            "lang": "mix",
            "results": Path(
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_cs_all/results_mmsu_open/"
                "mmsu_open.jsonl"
            ),
            "manifest": Path(
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_cs_all/mmsu_open/manifest.jsonl"
            ),
            "allow_modes": None,
            "rubric": "binary",
        },
        {
            "capability": "code_switch",
            "lang": "zh",
            "results": Path(
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_cs_all/results/"
                "vocalbench_zh_open_ended.jsonl"
            ),
            "manifest": Path(
                "/mnt/afs/users/wangyl/VoxMatrix/smoke_cs_all/"
                "vocalbench_zh_open_ended/manifest.jsonl"
            ),
            "allow_modes": None,
            "rubric": "open",
        },
    ]

    pack: List[Dict[str, Any]] = []
    for src in sources:
        man = _load_manifest_map(src["manifest"])
        evals = _load_eval_rows(src["results"])
        taken = 0
        for ev in evals:
            if taken >= limit_per_source:
                break
            mode = str(ev.get("eval_mode") or "")
            if src.get("allow_modes") and mode and mode not in src["allow_modes"]:
                continue
            sid = str(ev.get("sample_id") or "")
            meta = man.get(sid, {})
            wav = meta.get("WavPath") or meta.get("audio_locator") or ev.get("WavPath")
            if not wav or not Path(str(wav)).is_file():
                continue
            question = (
                meta.get("question")
                or meta.get("prompt")
                or ev.get("question")
                or ""
            )
            ref = _ref_text(ev.get("ref") if ev.get("ref") is not None else meta.get("text") or meta.get("answer"))
            rubric = src.get("rubric")
            if not rubric:
                if mode == "qa":
                    rubric = "binary"
                elif mode == "semi-open":
                    rubric = "semi-open"
                elif mode == "open":
                    rubric = "open"
                elif ref:
                    rubric = "semi-open"
                else:
                    rubric = "open"
            pack.append(
                {
                    "sample_id": sid,
                    "capability": src["capability"],
                    "rubric": rubric,
                    "WavPath": str(wav),
                    "question": str(question),
                    "pred": str(ev.get("pred") or ""),
                    "reference": ref,
                    "lang": src["lang"],
                    "direction": src.get("direction") or "",
                    "source_results": str(src["results"]),
                }
            )
            taken += 1
    return pack


def audio_seconds(wav_path: str) -> Optional[float]:
    try:
        with wave.open(wav_path, "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:  # noqa: BLE001
        pass
    try:
        import soundfile as sf

        info = sf.info(wav_path)
        return float(info.duration)
    except Exception:  # noqa: BLE001
        return None


def probe_mm_model(candidates: List[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """Return first working AdvancedGPT registry name for a tiny wav+text call."""
    from audio_evals.registry import registry

    notes: List[Dict[str, Any]] = []
    # Prefer a very short existing wav from the pack later; here use /dev/null skip
    return None, notes  # filled in run() after pack exists


def _probe_with_wav(wav_path: str, candidates: List[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    from audio_evals.registry import registry

    notes: List[Dict[str, Any]] = []
    for name in candidates:
        try:
            model = registry.get_model(name)
            if model is None:
                notes.append({"model": name, "ok": False, "error": "not_registered"})
                continue
            out = model._inference(
                [
                    {
                        "role": "user",
                        "contents": [
                            {"type": "audio", "value": wav_path},
                            {"type": "text", "value": "Reply with only the digit 1."},
                        ],
                    }
                ],
                max_tokens=8,
                temperature=0.0,
            )
            notes.append({"model": name, "ok": True, "raw": str(out)[:80]})
            return name, notes
        except Exception as exc:  # noqa: BLE001
            notes.append({"model": name, "ok": False, "error": str(exc)[:300]})
    return None, notes


def judge_one(ev, sample: Dict[str, Any], track: str, transcript: str, mm_model: Optional[str]):
    kwargs = {
        "capability": sample["capability"],
        "rubric": sample.get("rubric"),
        "question": sample.get("question") or "",
        "direction": sample.get("direction") or "",
        "WavPath": sample["WavPath"],
    }
    if track == "A0":
        kwargs["judge_template"] = "text"
        kwargs["audio_transcript"] = ""
        kwargs["require_transcript"] = False
        kwargs["judge_model_name"] = "gpt4o-mini"
    elif track == "A1":
        kwargs["judge_template"] = "text"
        kwargs["audio_transcript"] = transcript
        kwargs["require_transcript"] = True
        kwargs["judge_model_name"] = "gpt4o-mini"
    else:
        kwargs["judge_template"] = "multimodal"
        kwargs["audio_transcript"] = ""
        if mm_model:
            kwargs["judge_model_name"] = mm_model
    return ev._eval(sample["pred"], sample.get("reference") or "", **kwargs)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = row["sample_id"]
        by_sample.setdefault(sid, {"sample_id": sid, "capability": row["capability"], "tracks": {}})
        by_sample[sid]["tracks"][row["track"]] = {
            "score_0_100": row.get("score_0_100"),
            "skipped": row.get("skipped"),
            "skip_reason": row.get("skip_reason"),
            "gpt_score": row.get("gpt_score"),
            "usage": row.get("judge_usage"),
            "judge_model": row.get("judge_model"),
        }

    pairs = []
    agree_a1_a0 = 0
    agree_b_a0 = 0
    n_cmp = 0
    diffs = []
    for sid, item in by_sample.items():
        t = item["tracks"]
        a0 = t.get("A0", {}).get("score_0_100")
        a1 = t.get("A1", {}).get("score_0_100")
        b = t.get("B", {}).get("score_0_100")
        entry = {
            "sample_id": sid,
            "capability": item["capability"],
            "A0": a0,
            "A1": a1,
            "B": b,
            "A1_minus_A0": None if a0 is None or a1 is None else a1 - a0,
            "B_minus_A0": None if a0 is None or b is None else b - a0,
            "B_minus_A1": None if a1 is None or b is None else b - a1,
        }
        pairs.append(entry)
        if a0 is not None and a1 is not None:
            n_cmp += 1
            if a0 == a1:
                agree_a1_a0 += 1
            diffs.append(("A1-A0", sid, abs(a1 - a0), entry))
        if a0 is not None and b is not None:
            if a0 == b:
                agree_b_a0 += 1
            diffs.append(("B-A0", sid, abs(b - a0), entry))

    diffs_sorted = sorted(diffs, key=lambda x: x[2], reverse=True)[:10]
    usage_totals: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row.get("skipped"):
            continue
        track = row["track"]
        usage = row.get("judge_usage") or {}
        bucket = usage_totals.setdefault(
            track,
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0, "n": 0},
        )
        bucket["prompt_tokens"] += float(usage.get("prompt_tokens") or 0)
        bucket["completion_tokens"] += float(usage.get("completion_tokens") or 0)
        bucket["total_tokens"] += float(usage.get("total_tokens") or 0)
        bucket["calls"] += float(usage.get("calls") or 1)
        bucket["n"] += 1

    return {
        "n_samples": len(by_sample),
        "n_rows": len(rows),
        "agree_A1_A0": agree_a1_a0,
        "agree_B_A0": agree_b_a0,
        "pairs": pairs,
        "top_diffs": [
            {"pair": a, "sample_id": b, "abs_diff": c, "scores": d}
            for a, b, c, d in diffs_sorted
        ],
        "usage_by_track": usage_totals,
    }


def write_markdown(summary: Dict[str, Any], probe: List[Dict[str, Any]], out_path: Path) -> None:
    lines = [
        "# 开放语义 · 三路 LLM Judge 对比报告",
        "",
        "条件：A0=无转写文本 Judge；A1=Whisper/paraformer 转写；B=原音频多模态 Judge。",
        "",
        f"- 样本数：{summary.get('n_samples')}",
        f"- A1 与 A0 同分条数：{summary.get('agree_A1_A0')}",
        f"- B 与 A0 同分条数：{summary.get('agree_B_A0')}",
        "",
        "## 多模态模型探测",
        "",
        "```json",
        json.dumps(probe, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分轨 usage（实测）",
        "",
        "```json",
        json.dumps(summary.get("usage_by_track"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 样本分数",
        "",
        "| sample_id | capability | A0 | A1 | B | A1-A0 | B-A0 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for p in summary.get("pairs") or []:
        lines.append(
            "| {sample_id} | {capability} | {A0} | {A1} | {B} | {A1_minus_A0} | {B_minus_A0} |".format(
                sample_id=p.get("sample_id"),
                capability=p.get("capability"),
                A0=p.get("A0"),
                A1=p.get("A1"),
                B=p.get("B"),
                A1_minus_A0=p.get("A1_minus_A0"),
                B_minus_A0=p.get("B_minus_A0"),
            )
        )
    lines.extend(["", "## 分歧 Top", "", "```json", json.dumps(summary.get("top_diffs"), ensure_ascii=False, indent=2), "```", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--rebuild-pack", action="store_true")
    parser.add_argument("--limit-per-source", type=int, default=3)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--asr-cache", type=Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-mm", action="store_true")
    parser.add_argument("--tracks", default="A0,A1,B")
    parser.add_argument("--report-md", type=Path, default=DELIVER_MD)
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY missing", file=sys.stderr)
        return 2

    if args.rebuild_pack or not args.pack.is_file():
        pack = build_pack(limit_per_source=args.limit_per_source)
        args.pack.parent.mkdir(parents=True, exist_ok=True)
        with args.pack.open("w", encoding="utf-8") as handle:
            for row in pack:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote pack n={len(pack)} -> {args.pack}")
    else:
        pack = [json.loads(l) for l in args.pack.read_text(encoding="utf-8").splitlines() if l.strip()]
        print(f"loaded pack n={len(pack)} from {args.pack}")

    if not pack:
        print("empty pack", file=sys.stderr)
        return 1

    from audio_evals.evaluator.asr_for_judge import transcribe_for_judge
    from audio_evals.evaluator.semantic_llm_judge import SemanticLLMJudgeEvaluator

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    mm_model = None
    probe_notes: List[Dict[str, Any]] = []
    if "B" in tracks and not args.skip_mm:
        candidates = [
            os.environ.get("MESH_MM_JUDGE_MODEL", "").strip(),
            "gpt4o-mini-audio",
            "gpt4o-audio",
        ]
        # unique preserve order
        seen = set()
        cand = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                cand.append(c)
        # Prefer a short wav for probe.
        probe_wav = min(
            (s for s in pack if Path(s["WavPath"]).is_file()),
            key=lambda s: audio_seconds(s["WavPath"]) or 1e9,
        )["WavPath"]
        mm_model, probe_notes = _probe_with_wav(probe_wav, cand)
        if not mm_model:
            print(
                "multimodal probe failed on ChatAnywhere/compatible endpoint; "
                "B track will skip (need OpenAI audio-capable model).",
                file=sys.stderr,
            )
            probe_notes.append(
                {
                    "fatal": "no_mm_model",
                    "note": (
                        "ChatAnywhere lists audio model ids but rejects chat "
                        "completions with input_audio (404/400). Keep A0/A1."
                    ),
                }
            )

    ev = SemanticLLMJudgeEvaluator()
    rows: List[Dict[str, Any]] = []
    for sample in pack:
        transcript = ""
        if "A1" in tracks and not args.skip_asr:
            try:
                transcript = transcribe_for_judge(
                    sample["WavPath"],
                    lang=sample.get("lang"),
                    cache_dir=str(args.asr_cache),
                )
            except Exception as exc:  # noqa: BLE001
                transcript = ""
                asr_err = str(exc)
            else:
                asr_err = None
        else:
            asr_err = "skipped" if args.skip_asr else None

        for track in tracks:
            if track == "B" and (args.skip_mm or not mm_model):
                result = {
                    "skipped": 1,
                    "skip_reason": "mm_unavailable",
                    "score_0_100": None,
                    "judge_usage": {},
                }
            else:
                result = judge_one(ev, sample, track, transcript, mm_model)
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "capability": sample["capability"],
                    "rubric": sample.get("rubric"),
                    "track": track,
                    "lang": sample.get("lang"),
                    "audio_seconds": audio_seconds(sample["WavPath"]),
                    "audio_transcript": transcript if track == "A1" else "",
                    "asr_error": asr_err if track == "A1" else None,
                    "score_0_100": result.get("score_0_100"),
                    "gpt_score": result.get("gpt_score"),
                    "match": result.get("match"),
                    "skipped": result.get("skipped"),
                    "skip_reason": result.get("skip_reason"),
                    "judge_usage": result.get("judge_usage"),
                    "judge_model": result.get("judge_model"),
                    "prompt_id": result.get("prompt_id"),
                    "raw_judge_output": result.get("raw_judge_output"),
                }
            )
            print(
                f"{sample['sample_id']} {track} score={result.get('score_0_100')} "
                f"skip={result.get('skip_reason')}",
                flush=True,
            )

    summary = summarize(rows)
    payload = {
        "pack": str(args.pack),
        "n": len(pack),
        "mm_model": mm_model,
        "probe": probe_notes,
        "summary": summary,
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, probe_notes, args.report_md)
    print(f"wrote {args.out}")
    print(f"wrote {args.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
