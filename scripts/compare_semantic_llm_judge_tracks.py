#!/usr/bin/env python3
"""Paired A1 (transcript) vs B (transcript+audio) semantic LLM-judge comparison.

Same judge model for both arms (default qwen3-omni-thinking). Builds a best-effort
stratified pack from existing smoke preds+manifests, runs ASR once per sample,
judges A1/B, and writes paired Δ summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = Path("/tmp/semantic_judge_track_compare.json")
DEFAULT_PACK = ROOT / "scripts" / "data" / "semantic_judge_compare_pack.jsonl"
DEFAULT_ASR_CACHE = Path("/tmp/semantic_judge_asr_cache")
DELIVER_MD = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_三路Judge对比报告.md")
DEFAULT_JUDGE = "qwen3-omni-thinking"


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
            elif isinstance(obj.get("data"), dict) and "pred" in obj["data"]:
                rows.append(obj["data"])
    return rows


def _load_manifest_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            sid = str(row.get("sample_id") or row.get("id") or "")
            if sid:
                out[sid] = row
    return out


def _ref_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " | ".join(str(x) for x in value if str(x).strip())
    return str(value).strip()


def _pack_sources() -> List[Dict[str, Any]]:
    """All known smoke sources with WavPath+pred (best-effort 9-cell fill)."""
    r8 = Path("/mnt/afs/users/wangyl/8p31/runs")
    st = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_st_all")
    cs = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_cs_all")
    return [
        {
            "capability": "qa",
            "lang": "en",
            "results": r8 / "qa_uro_bench_smoke/cluster_run_20260911_062245/results.jsonl",
            "manifest": r8 / "qa_uro_bench_smoke/qa_uro_bench_10.jsonl",
            "allow_modes": {"open", "semi-open", "qa"},
        },
        {
            "capability": "qa",
            "lang": "en",
            "results": r8 / "qa_vocalbench_smoke/cluster_run_20260831_114828/results.jsonl",
            "manifest": r8 / "qa_vocalbench_smoke/qa_vocalbench_10.jsonl",
            "allow_modes": None,
            "rubric_from_track": True,
        },
        {
            "capability": "qa",
            "lang": "zh",
            "results": r8 / "qa_vocalbench_zh_smoke/cluster_run_20260911_062247/results.jsonl",
            "manifest": r8 / "qa_vocalbench_zh_smoke/qa_vocalbench_zh_10.jsonl",
            "allow_modes": None,
            "rubric": "binary",
        },
        {
            "capability": "qa",
            "lang": "en",
            "results": r8 / "qa_spoken_squad_smoke/cluster_run_20260911_062226/results.jsonl",
            "manifest": r8 / "qa_spoken_squad_smoke/qa_spoken_squad_10.jsonl",
            "allow_modes": None,
            "rubric": "binary",
        },
        {
            "capability": "qa",
            "lang": "en",
            "results": r8 / "qa_ear_wdyl_smoke/cluster_run_20260831_055552/results.jsonl",
            "manifest": r8 / "qa_ear_wdyl_smoke/qa_ear_wdyl_10.jsonl",
            "allow_modes": None,
            "rubric": "binary",
        },
        {
            "capability": "audio_reasoning",
            "lang": "en",
            "results": r8 / "audio_reasoning_vocalbench_smoke/cluster_run_20260831_104516/results.jsonl",
            "manifest": r8 / "audio_reasoning_vocalbench_smoke/audio_reasoning_vocalbench_10.jsonl",
            "allow_modes": None,
        },
        {
            "capability": "audio_reasoning",
            "lang": "zh",
            "results": r8
            / "audio_reasoning_vocalbench_zh_smoke/cluster_run_20260831_104455/results.jsonl",
            "manifest": r8
            / "audio_reasoning_vocalbench_zh_smoke/audio_reasoning_vocalbench_zh_10.jsonl",
            "allow_modes": None,
        },
        {
            "capability": "audio_reasoning",
            "lang": "en",
            "results": r8 / "audio_reasoning_mmsu_smoke/cluster_run_20260911_061531/results.jsonl",
            "manifest": r8 / "audio_reasoning_mmsu_smoke/audio_reasoning_mmsu_open_10.jsonl",
            "allow_modes": None,
        },
        {
            "capability": "speech_translation",
            "lang": "zh",
            "results": st / "results/covost2_en_zh.jsonl",
            "manifest": st / "covost2_en_zh/manifest.jsonl",
            "rubric": "translation",
            "direction": "Translate into Chinese",
        },
        {
            "capability": "speech_translation",
            "lang": "en",
            "results": st / "results/covost2_zh_en.jsonl",
            "manifest": st / "covost2_zh_en/manifest.jsonl",
            "rubric": "translation",
            "direction": "Translate into English",
        },
        {
            "capability": "speech_translation",
            "lang": "en",
            "results": st / "results_mmsu_open/mmsu_open.jsonl",
            "manifest": st / "mmsu_open/manifest.jsonl",
            "rubric": "translation",
            "direction": "Translate into English",
        },
        {
            "capability": "code_switch",
            "lang": "mix",
            "results": cs / "results_mmsu_open/mmsu_open.jsonl",
            "manifest": cs / "mmsu_open/manifest.jsonl",
            "rubric": "binary",
        },
        {
            "capability": "code_switch",
            "lang": "zh",
            "results": cs / "results/vocalbench_zh_knowledge.jsonl",
            "manifest": cs / "vocalbench_zh_knowledge/manifest.jsonl",
            "rubric": "binary",
        },
        {
            "capability": "code_switch",
            "lang": "zh",
            "results": cs / "results/vocalbench_zh_open_ended.jsonl",
            "manifest": cs / "vocalbench_zh_open_ended/manifest.jsonl",
            "rubric": "open",
        },
    ]


def _infer_rubric(src: Dict[str, Any], ev: Dict[str, Any], meta: Dict[str, Any], ref: str) -> str:
    if src.get("rubric"):
        return str(src["rubric"])
    mode = str(ev.get("eval_mode") or meta.get("eval_mode") or "").lower()
    track = str(meta.get("qa_track") or meta.get("subset") or meta.get("task") or "").lower()
    if mode == "qa":
        return "binary"
    if mode == "semi-open":
        return "semi-open"
    if mode == "open":
        return "open"
    if src.get("rubric_from_track"):
        if "single_round" in track or "open" in track:
            return "open"
        if "knowledge" in track:
            return "binary"
    if ref:
        return "semi-open"
    return "open"


def build_pack(limit_per_source: int = 50) -> List[Dict[str, Any]]:
    """Assemble real samples with WavPath + pred; dedupe by sample_id."""
    pack: List[Dict[str, Any]] = []
    seen: set = set()
    for src in _pack_sources():
        man = _load_manifest_map(Path(src["manifest"]))
        evals = _load_eval_rows(Path(src["results"]))
        taken = 0
        for ev in evals:
            if taken >= limit_per_source:
                break
            mode = str(ev.get("eval_mode") or "")
            if src.get("allow_modes") and mode and mode not in src["allow_modes"]:
                continue
            sid = str(ev.get("sample_id") or "")
            if not sid or sid in seen:
                continue
            meta = man.get(sid, {})
            wav = meta.get("WavPath") or meta.get("audio_locator") or ev.get("WavPath")
            if not wav or not Path(str(wav)).is_file():
                continue
            pred = str(ev.get("pred") or "").strip()
            if not pred:
                continue
            question = (
                meta.get("question")
                or meta.get("prompt")
                or ev.get("question")
                or ""
            )
            ref = _ref_text(
                ev.get("ref")
                if ev.get("ref") is not None
                else meta.get("text") or meta.get("answer") or meta.get("reference")
            )
            rubric = _infer_rubric(src, ev, meta, ref)
            # Strip ref for intentional AR open cells when source marks open_no_ref.
            if src.get("force_open_no_ref"):
                rubric = "open"
                ref = ""
            seen.add(sid)
            pack.append(
                {
                    "sample_id": sid,
                    "capability": src["capability"],
                    "rubric": rubric,
                    "WavPath": str(wav),
                    "question": str(question),
                    "pred": pred,
                    "reference": ref,
                    "lang": src.get("lang") or "en",
                    "direction": src.get("direction") or "",
                    "source_results": str(src["results"]),
                }
            )
            taken += 1
    return pack


def cell_counts(pack: List[Dict[str, Any]]) -> Dict[str, int]:
    c: Counter = Counter()
    for row in pack:
        c[f"{row['capability']}|{row['rubric']}"] += 1
    return dict(sorted(c.items()))


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


def _local_weight_ok(model_name: str) -> Optional[str]:
    from audio_evals.registry import registry

    try:
        spec = registry._model.get(model_name)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(spec, dict):
        return None
    path = (spec.get("args") or {}).get("path")
    if isinstance(path, str) and Path(path).is_dir():
        return path
    return None


def _probe_with_wav(wav_path: str, candidates: List[str]) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    from audio_evals.registry import registry

    notes: List[Dict[str, Any]] = []
    for name in candidates:
        if name in ("qwen3-omni-thinking", "qwen3-omni-audio", "qwen3-omni-speech"):
            weight = _local_weight_ok(name)
            if weight:
                notes.append({"model": name, "ok": True, "probe": "local_weights", "path": weight})
                return name, notes
            notes.append({"model": name, "ok": False, "error": "local_weights_missing"})
            continue
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


def resolve_paired_judge(mm_model: Optional[str]) -> str:
    return (
        os.environ.get("MESH_JUDGE_MODEL", "").strip()
        or os.environ.get("MESH_MM_JUDGE_MODEL", "").strip()
        or mm_model
        or DEFAULT_JUDGE
    )


def judge_one(ev, sample: Dict[str, Any], track: str, transcript: str, judge_model: str):
    kwargs = {
        "capability": sample["capability"],
        "rubric": sample.get("rubric"),
        "question": sample.get("question") or "",
        "direction": sample.get("direction") or "",
        "WavPath": sample["WavPath"],
        "judge_model_name": judge_model,
        "audio_transcript": transcript,
    }
    if track == "A0":
        kwargs["judge_template"] = "text"
        kwargs["audio_transcript"] = ""
        kwargs["require_transcript"] = False
    elif track == "A1":
        kwargs["judge_template"] = "text"
        kwargs["require_transcript"] = True
    else:
        kwargs["judge_template"] = "multimodal"
        kwargs["require_transcript"] = True
    return ev._eval(sample["pred"], sample.get("reference") or "", **kwargs)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_sample: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        sid = row["sample_id"]
        by_sample.setdefault(
            sid,
            {
                "sample_id": sid,
                "capability": row["capability"],
                "rubric": row.get("rubric"),
                "tracks": {},
            },
        )
        by_sample[sid]["tracks"][row["track"]] = {
            "score_0_100": row.get("score_0_100"),
            "skipped": row.get("skipped"),
            "skip_reason": row.get("skip_reason"),
            "gpt_score": row.get("gpt_score"),
            "usage": row.get("judge_usage"),
            "judge_model": row.get("judge_model"),
            "audio_transcript_used": row.get("audio_transcript_used"),
        }

    pairs = []
    agree_exact = 0
    agree_close = 0  # |Δ|<=20
    n_pair = 0
    deltas: List[float] = []
    cell_stats: Dict[str, Dict[str, Any]] = {}
    for sid, item in by_sample.items():
        t = item["tracks"]
        a1 = t.get("A1", {}).get("score_0_100")
        b = t.get("B", {}).get("score_0_100")
        a0 = t.get("A0", {}).get("score_0_100")
        delta = None if a1 is None or b is None else b - a1
        entry = {
            "sample_id": sid,
            "capability": item["capability"],
            "rubric": item.get("rubric"),
            "A0": a0,
            "A1": a1,
            "B": b,
            "delta_B_minus_A1": delta,
            "agree_exact": a1 is not None and b is not None and a1 == b,
            "agree_close": a1 is not None and b is not None and abs(b - a1) <= 20,
        }
        pairs.append(entry)
        key = f"{item['capability']}|{item.get('rubric')}"
        cell = cell_stats.setdefault(
            key,
            {"n": 0, "n_paired": 0, "agree_exact": 0, "agree_close": 0, "delta_sum": 0.0},
        )
        cell["n"] += 1
        if delta is not None:
            n_pair += 1
            cell["n_paired"] += 1
            deltas.append(delta)
            cell["delta_sum"] += delta
            if entry["agree_exact"]:
                agree_exact += 1
                cell["agree_exact"] += 1
            if entry["agree_close"]:
                agree_close += 1
                cell["agree_close"] += 1

    for cell in cell_stats.values():
        np_ = cell["n_paired"] or 1
        cell["delta_mean"] = cell["delta_sum"] / np_ if cell["n_paired"] else None
        cell["agree_exact_rate"] = cell["agree_exact"] / np_ if cell["n_paired"] else None
        cell["agree_close_rate"] = cell["agree_close"] / np_ if cell["n_paired"] else None

    diffs_sorted = sorted(
        [p for p in pairs if p.get("delta_B_minus_A1") is not None],
        key=lambda x: abs(x["delta_B_minus_A1"]),
        reverse=True,
    )[:15]

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
        "n_paired": n_pair,
        "agree_exact": agree_exact,
        "agree_close": agree_close,
        "agree_exact_rate": (agree_exact / n_pair) if n_pair else None,
        "agree_close_rate": (agree_close / n_pair) if n_pair else None,
        "delta_mean": (sum(deltas) / len(deltas)) if deltas else None,
        "cell_stats": cell_stats,
        "pairs": pairs,
        "top_abs_deltas": diffs_sorted,
        "usage_by_track": usage_totals,
    }


def write_markdown(
    summary: Dict[str, Any],
    probe: List[Dict[str, Any]],
    out_path: Path,
    *,
    judge_model: str,
    cell_pack: Dict[str, int],
) -> None:
    """Write machine JSON-style A1/B summary markdown.

    Refuses to overwrite the locked formal case-study report
    (``开放语义_A1B音频增益实验报告.md`` with ``A1B_REPORT_LOCK``).
    """
    out_path = Path(out_path)
    lock_token = "A1B_REPORT_LOCK"
    formal = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B音频增益实验报告.md")
    if out_path.resolve() == formal.resolve() or (
        out_path.is_file() and lock_token in out_path.read_text(encoding="utf-8", errors="ignore")
    ):
        alt = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B成对Judge自动稿.md")
        print(
            f"[a1b] refuse overwrite locked report {out_path}; writing auto draft -> {alt}",
            file=sys.stderr,
        )
        out_path = alt

    lines = [
        "# 开放语义 · A1 vs B 成对 LLM Judge 对比报告",
        "",
        "> 本文为 JSON 自动稿；正式样例分析见 `开放语义_A1B音频增益实验报告.md`（已锁定，勿覆盖）。",
        "",
        "## 结论摘要",
        "",
        "- **设计**：A1=同转写文本臂；B=**同一转写 + 原音频**；**同一** `judge_model`。",
        f"- **Judge**：`{judge_model}`",
        f"- 样本数：{summary.get('n_samples')}；成对有效：{summary.get('n_paired')}",
        f"- 精确一致率：{summary.get('agree_exact_rate')}",
        f"- 接近一致率（|Δ|≤20）：{summary.get('agree_close_rate')}",
        f"- Δ=B−A1 均值：{summary.get('delta_mean')}",
        "",
        "## Pack 格子计数（capability|rubric）",
        "",
        "```json",
        json.dumps(cell_pack, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 按格成对统计",
        "",
        "```json",
        json.dumps(summary.get("cell_stats"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 模型探测",
        "",
        "```json",
        json.dumps(probe, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 分轨 usage",
        "",
        "```json",
        json.dumps(summary.get("usage_by_track"), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 样本分数",
        "",
        "| sample_id | capability | rubric | A1 | B | Δ(B−A1) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for p in summary.get("pairs") or []:
        lines.append(
            "| {sample_id} | {capability} | {rubric} | {A1} | {B} | {delta} |".format(
                sample_id=p.get("sample_id"),
                capability=p.get("capability"),
                rubric=p.get("rubric"),
                A1=p.get("A1"),
                B=p.get("B"),
                delta=p.get("delta_B_minus_A1"),
            )
        )
    lines.extend(
        [
            "",
            "## |Δ| Top",
            "",
            "```json",
            json.dumps(summary.get("top_abs_deltas"), ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--rebuild-pack", action="store_true")
    parser.add_argument("--limit-per-source", type=int, default=50)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--asr-cache", type=Path, default=DEFAULT_ASR_CACHE)
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-mm", action="store_true")
    parser.add_argument("--tracks", default="A1,B")
    parser.add_argument("--report-md", type=Path, default=DELIVER_MD)
    parser.add_argument("--judge-model", default="")
    parser.add_argument("--max-samples", type=int, default=0, help="0=all")
    parser.add_argument(
        "--stub-judge",
        action="store_true",
        help="Use deterministic stub judge (no GPU/API); for pathway validation only",
    )
    parser.add_argument(
        "--transcript-mode",
        choices=("asr", "oracle", "auto"),
        default="auto",
        help="asr=Whisper/paraformer; oracle=pack oracle_transcript/question; auto=asr else oracle",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --out / .partial.json; skip already scored (sample_id, track)",
    )
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard id")
    parser.add_argument("--num-shards", type=int, default=1, help="split pack across workers")
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="Skip markdown report (for sharded workers; merge writes the report)",
    )
    args = parser.parse_args()

    if args.rebuild_pack or not args.pack.is_file():
        pack = build_pack(limit_per_source=args.limit_per_source)
        args.pack.parent.mkdir(parents=True, exist_ok=True)
        with args.pack.open("w", encoding="utf-8") as handle:
            for row in pack:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote pack n={len(pack)} cells={cell_counts(pack)} -> {args.pack}")
    else:
        pack = [
            json.loads(l)
            for l in args.pack.read_text(encoding="utf-8").splitlines()
            if l.strip()
        ]
        print(f"loaded pack n={len(pack)} cells={cell_counts(pack)} from {args.pack}")

    if args.max_samples and args.max_samples > 0:
        pack = pack[: args.max_samples]
        print(f"truncated pack to n={len(pack)}")

    if args.num_shards < 1:
        print("--num-shards must be >= 1", file=sys.stderr)
        return 2
    if not (0 <= args.shard_index < args.num_shards):
        print("--shard-index out of range", file=sys.stderr)
        return 2
    if args.num_shards > 1:
        full_n = len(pack)
        pack = [s for i, s in enumerate(pack) if i % args.num_shards == args.shard_index]
        print(
            f"shard {args.shard_index}/{args.num_shards}: n={len(pack)} of {full_n}",
            flush=True,
        )

    if not pack:
        print("empty pack", file=sys.stderr)
        return 1

    from audio_evals.evaluator.asr_for_judge import transcribe_for_judge
    from audio_evals.evaluator.semantic_llm_judge import SemanticLLMJudgeEvaluator

    # Drop rows without pred (cannot judge).
    before = len(pack)
    pack = [s for s in pack if str(s.get("pred") or "").strip()]
    if len(pack) < before:
        print(f"dropped {before - len(pack)} rows without pred; remain n={len(pack)}")
    if not pack:
        print("no samples with pred", file=sys.stderr)
        return 1

    tracks = [t.strip() for t in args.tracks.split(",") if t.strip()]
    mm_model = None
    probe_notes: List[Dict[str, Any]] = []
    stub_mode = bool(args.stub_judge)

    if stub_mode:
        judge_model = "stub-judge"
        mm_model = "stub-judge"
        probe_notes.append(
            {
                "model": "stub-judge",
                "ok": True,
                "note": "deterministic pathway stub; not for formal scores",
            }
        )
    else:
        need_local = any(t in ("A1", "B") for t in tracks)
        if need_local and not args.skip_mm:
            candidates = [
                args.judge_model.strip(),
                os.environ.get("MESH_JUDGE_MODEL", "").strip(),
                os.environ.get("MESH_MM_JUDGE_MODEL", "").strip(),
                DEFAULT_JUDGE,
                "gpt4o-mini-audio",
                "gpt4o-audio",
            ]
            seen = set()
            cand = []
            for c in candidates:
                if c and c not in seen:
                    seen.add(c)
                    cand.append(c)
            probe_wav = min(
                (s for s in pack if Path(s["WavPath"]).is_file()),
                key=lambda s: audio_seconds(s["WavPath"]) or 1e9,
            )["WavPath"]
            mm_model, probe_notes = _probe_with_wav(probe_wav, cand)

        judge_model = resolve_paired_judge(mm_model if not args.skip_mm else None)
        if args.judge_model.strip():
            judge_model = args.judge_model.strip()

        if "B" in tracks and (args.skip_mm or not mm_model) and not _local_weight_ok(judge_model):
            print(
                "multimodal/local judge unavailable; B will skip. "
                "Set MESH_JUDGE_MODEL or ensure Thinking weights.",
                file=sys.stderr,
            )
            probe_notes.append({"fatal": "no_mm_model", "judge_model": judge_model})

    class _StubJudge:
        """Deterministic judge for pathway tests (same A1/B score from text)."""

        model_name = "stub-judge"

        def inference(self, prompt, **kwargs):
            blob = json.dumps(prompt, ensure_ascii=False)
            # Binary prompts ask Yes/No.
            if "Yes" in blob and "No" in blob and "only output a single" in blob.lower():
                return "Yes" if ("Reference" in blob and "Response" in blob) else "No"
            # Score 1-5 from crude length bucket for stability.
            return "3"

    ev = SemanticLLMJudgeEvaluator(
        judge_model_name=judge_model,
        mm_judge_model_name=judge_model,
    )
    if stub_mode:
        # Monkeypatch model resolution for both arms.
        import audio_evals.evaluator.voice_bench as vb

        vb.get_judge_model = lambda name: _StubJudge()  # type: ignore
        vb.resolve_judge_model_name = lambda name: "stub-judge"  # type: ignore

    rows: List[Dict[str, Any]] = []
    done_keys: set = set()
    pack_ids = {str(s["sample_id"]) for s in pack}
    if args.resume:
        resume_paths = [args.out, args.out.with_suffix(".partial.json")]
        # Shared seeds / prior full run (non-sharded).
        resume_paths.extend(
            [
                args.out.parent / "semantic_judge_a1b_full.json",
                args.out.parent / "semantic_judge_a1b_full.partial.json",
            ]
        )
        for path in resume_paths:
            if not path.is_file():
                continue
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                print(f"resume skip unreadable {path}: {exc}", flush=True)
                continue
            added = 0
            for row in prev.get("rows") or []:
                key = (str(row.get("sample_id")), str(row.get("track")))
                if key in done_keys:
                    continue
                if key[0] not in pack_ids:
                    continue
                # Only resume successful scores; re-judge skips (e.g. parse_failed).
                if row.get("score_0_100") is not None and not row.get("skipped"):
                    rows.append(row)
                    done_keys.add(key)
                    added += 1
            if added:
                print(f"resume from {path} +{added}", flush=True)
        print(f"resume loaded n_rows={len(rows)} keys={len(done_keys)}", flush=True)

    for sample in pack:
        transcript = ""
        asr_err = None
        transcript_source = None
        want_tr = any(t in ("A1", "B") for t in tracks) and not args.skip_asr
        if want_tr:
            mode = args.transcript_mode
            oracle = str(
                sample.get("oracle_transcript")
                or sample.get("question")
                or sample.get("direction")
                or ""
            ).strip()
            if mode in ("oracle", "auto") and (mode == "oracle" or stub_mode):
                transcript = oracle
                transcript_source = "oracle"
                if not transcript:
                    asr_err = "empty_oracle_transcript"
            else:
                try:
                    transcript = transcribe_for_judge(
                        sample["WavPath"],
                        lang=sample.get("lang"),
                        cache_dir=str(args.asr_cache),
                    )
                    transcript_source = "asr"
                except Exception as exc:  # noqa: BLE001
                    if mode == "auto" and oracle:
                        transcript = oracle
                        transcript_source = "oracle_fallback"
                        asr_err = f"asr_failed:{exc}"
                    else:
                        transcript = ""
                        asr_err = str(exc)
        elif args.skip_asr:
            asr_err = "skipped"

        for track in tracks:
            key = (str(sample["sample_id"]), track)
            if key in done_keys:
                print(f"{sample['sample_id']} {track} resume_skip", flush=True)
                continue
            if (
                not stub_mode
                and track == "B"
                and (args.skip_mm or (not mm_model and not _local_weight_ok(judge_model)))
            ):
                result = {
                    "skipped": 1,
                    "skip_reason": "mm_unavailable",
                    "score_0_100": None,
                    "judge_usage": {},
                    "judge_model": judge_model,
                }
            else:
                result = judge_one(ev, sample, track, transcript, judge_model)
            row = {
                "sample_id": sample["sample_id"],
                "capability": sample["capability"],
                "rubric": sample.get("rubric"),
                "track": track,
                "lang": sample.get("lang"),
                "audio_seconds": audio_seconds(sample["WavPath"]),
                "audio_transcript": transcript,
                "transcript_source": transcript_source,
                "audio_transcript_used": result.get("audio_transcript_used"),
                "asr_error": asr_err,
                "score_0_100": result.get("score_0_100"),
                "gpt_score": result.get("gpt_score"),
                "match": result.get("match"),
                "skipped": result.get("skipped"),
                "skip_reason": result.get("skip_reason"),
                "judge_usage": result.get("judge_usage"),
                "judge_model": result.get("judge_model") or judge_model,
                "prompt_id": result.get("prompt_id"),
                "raw_judge_output": result.get("raw_judge_output"),
            }
            rows.append(row)
            done_keys.add(key)
            print(
                f"{sample['sample_id']} {track} score={result.get('score_0_100')} "
                f"skip={result.get('skip_reason')} model={result.get('judge_model') or judge_model}",
                flush=True,
            )

            # Checkpoint every 20 scored arms so long GPU runs are recoverable.
            if len(rows) % 20 == 0:
                mid = {
                    "pack": str(args.pack),
                    "n": len(pack),
                    "cell_counts": cell_counts(pack),
                    "judge_model": judge_model,
                    "mm_model": mm_model,
                    "probe": probe_notes,
                    "summary": summarize(rows),
                    "rows": rows,
                    "partial": True,
                }
                args.out.parent.mkdir(parents=True, exist_ok=True)
                ckpt = args.out.with_suffix(".partial.json")
                ckpt.write_text(json.dumps(mid, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"checkpoint {ckpt} n_rows={len(rows)}", flush=True)

    summary = summarize(rows)
    payload = {
        "pack": str(args.pack),
        "n": len(pack),
        "cell_counts": cell_counts(pack),
        "judge_model": judge_model,
        "mm_model": mm_model,
        "probe": probe_notes,
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    if not args.skip_report:
        write_markdown(
            summary,
            probe_notes,
            args.report_md,
            judge_model=judge_model,
            cell_pack=cell_counts(pack),
        )
        print(f"wrote {args.report_md}")
    try:
        from audio_evals.evaluator.voice_bench import clear_judge_model_cache

        clear_judge_model_cache()
    except Exception:  # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
