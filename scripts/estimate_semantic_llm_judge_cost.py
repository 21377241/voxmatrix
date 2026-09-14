#!/usr/bin/env python3
"""Estimate full open-semantic LLM-judge API cost from measured per-track usage.

Reads compare output (/tmp/semantic_judge_track_compare.json) when present,
falls back to heuristic tokens/call. Writes markdown under 8p31/交付/.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARE = Path("/tmp/semantic_judge_track_compare.json")
DEFAULT_PRICE = ROOT / "scripts" / "data" / "semantic_llm_judge_pricing.yaml"
DEFAULT_OUT = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_LLM-Judge_API开销估算.md")

# Heuristic fallbacks if compare run has no usage (tokens / call)
FALLBACK_TOKENS = {
    "A0": {"prompt_tokens": 450, "completion_tokens": 4, "model": "gpt-4o-mini"},
    "A1": {"prompt_tokens": 650, "completion_tokens": 4, "model": "gpt-4o-mini"},
    "B": {
        "prompt_tokens": 2500,
        "completion_tokens": 4,
        "model": "gpt-4o-mini-audio-preview",
    },
}


def load_price(path: Path) -> Dict[str, Any]:
    raw = os.environ.get("SEMANTIC_JUDGE_PRICE_JSON", "").strip()
    if raw:
        return json.loads(raw)
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(text)
    # minimal fallback parser not needed if pyyaml present in env
    raise RuntimeError("PyYAML required to load pricing.yaml")


def per_call_from_compare(compare: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    out = {k: dict(v) for k, v in FALLBACK_TOKENS.items()}
    if not compare:
        return out
    usage = (compare.get("summary") or {}).get("usage_by_track") or {}
    mm_model = compare.get("mm_model")
    for track, bucket in usage.items():
        n = float(bucket.get("n") or 0)
        if n <= 0:
            continue
        out[track] = {
            "prompt_tokens": float(bucket.get("prompt_tokens") or 0) / n,
            "completion_tokens": float(bucket.get("completion_tokens") or 0) / n,
            "model": (
                (compare.get("rows") or [{}])[0].get("judge_model")
                if track != "B"
                else (mm_model or FALLBACK_TOKENS["B"]["model"])
            )
            or FALLBACK_TOKENS.get(track, {}).get("model"),
        }
    # Prefer model name from a successful row per track
    for row in compare.get("rows") or []:
        if row.get("skipped"):
            continue
        track = row.get("track")
        if track in out and row.get("judge_model"):
            out[track]["model"] = row["judge_model"]
    return out


def cost_for_tokens(model: str, prompt_t: float, completion_t: float, price: Dict[str, Any]) -> float:
    models = price.get("models") or {}
    # strip registry aliases
    key = model
    for alias, real in {
        "gpt4o-mini": "gpt-4o-mini",
        "gpt4o-mini-audio": "gpt-4o-mini-audio-preview",
        "gpt4o-audio": "gpt-4o-audio-preview",
    }.items():
        if model == alias:
            key = real
    cfg = models.get(key) or models.get("gpt-4o-mini") or {}
    inp = float(cfg.get("input_per_mtok") or 0.15)
    outp = float(cfg.get("output_per_mtok") or 0.60)
    return (prompt_t / 1e6) * inp + (completion_t / 1e6) * outp


def full_n(price: Dict[str, Any]) -> int:
    counts = price.get("full_eval_counts") or {}
    return int(sum(int(v) for v in counts.values()))


def render_md(
    price: Dict[str, Any],
    per_call: Dict[str, Dict[str, float]],
    compare_path: Optional[Path],
) -> str:
    n_full = full_n(price)
    lines = [
        "# 开放语义 · LLM Judge API 开销估算",
        "",
        "日期：2026-09-14",
        "",
        "## 口径",
        "",
        "- 仅计 **Judge LLM API**（ChatAnywhere / OpenAI 兼容）；本地 Whisper / paraformer **不计** API 费。",
        "- 全量样本数按交付须走 LLM Judge 的开放/语义子集固定表求和（见下）。",
        f"- 实测单价来源：`{compare_path}`" if compare_path else "- 实测缺失，使用启发式 tokens/call。",
        "- 美元价默认取公开 list price（可被 `scripts/data/semantic_llm_judge_pricing.yaml` 覆盖）。",
        "",
        "## 全量样本计数",
        "",
        "| 子集 | n |",
        "|---|---:|",
    ]
    for k, v in (price.get("full_eval_counts") or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.extend([f"| **合计** | **{n_full}** |", ""])

    lines.extend(
        [
            "## 实测 / 假设 · 单次调用",
            "",
            "| track | model | prompt_tokens | completion_tokens | USD/call |",
            "|---|---|---:|---:|---:|",
        ]
    )
    usd_per = {}
    for track in ("A0", "A1", "B"):
        cfg = per_call[track]
        usd = cost_for_tokens(
            str(cfg.get("model")),
            float(cfg["prompt_tokens"]),
            float(cfg["completion_tokens"]),
            price,
        )
        usd_per[track] = usd
        lines.append(
            f"| {track} | {cfg.get('model')} | {cfg['prompt_tokens']:.1f} | "
            f"{cfg['completion_tokens']:.1f} | ${usd:.6f} |"
        )

    lines.extend(["", "## 场景外推（全量）", "", "| 场景 | 说明 | 公式概要 | 估计 USD |", "|---|---|---|---:|"])
    for name, sc in (price.get("scenarios") or {}).items():
        total = (
            n_full * float(sc.get("A0") or 0) * usd_per["A0"]
            + n_full * float(sc.get("A1") or 0) * usd_per["A1"]
            + n_full * float(sc.get("B") or 0) * usd_per["B"]
        )
        formula = (
            f"N×({sc.get('A0')}·A0 + {sc.get('A1')}·A1 + {sc.get('B')}·B), N={n_full}"
        )
        lines.append(
            f"| `{name}` | {sc.get('description','')} | {formula} | **${total:.2f}** |"
        )

    lines.extend(
        [
            "",
            "## 建议",
            "",
            "- 正式榜默认 **`only_A0`**（与 URO/VoiceBench 文本 judge 对齐，最便宜）。",
            "- 翻译 / CS / AR 等依赖源音频内容的子集，加 **`A0_plus_A1`** 或仅对这些子集跑 A1。",
            "- 模板 B 作对照：用 **`A0_plus_B_sample10`**，避免全量三路上界费用。",
            "",
            "## 备注",
            "",
            "- **B 路实测**：当前 ChatAnywhere 不支持 `input_audio`；上表 B 的 tokens 为启发式占位，换官方音频模型后请重跑 `compare_semantic_llm_judge_tracks.py` 再估。",
            "- **ASR 费用未计入** Judge API；本地 Whisper/paraformer 为 GPU/CPU 算力；本次冒烟可用 `JUDGE_ASR_BACKEND=api`（`whisper-1`）加速，仍属独立 ASR。",
            "- **美元价**按公开 list price；ChatAnywhere 多为积分/套餐，实际 RMB 以中转账单为准，可把单价写入 `semantic_llm_judge_pricing.yaml` 后重算。",
            "",
            "## 复现",
            "",
            "```bash",
            "export OPENAI_API_KEY=...",
            "export OPENAI_BASE_URL=https://api.chatanywhere.tech/v1",
            "PYTHONPATH=/mnt/afs/users/wangyl/VoxMatrix \\",
            "  python scripts/compare_semantic_llm_judge_tracks.py --rebuild-pack",
            "PYTHONPATH=/mnt/afs/users/wangyl/VoxMatrix \\",
            "  python scripts/estimate_semantic_llm_judge_cost.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", type=Path, default=DEFAULT_COMPARE)
    parser.add_argument("--price", type=Path, default=DEFAULT_PRICE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    price = load_price(args.price)
    compare = None
    if args.compare.is_file():
        compare = json.loads(args.compare.read_text(encoding="utf-8"))
    per_call = per_call_from_compare(compare)
    md = render_md(price, per_call, args.compare if compare else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
