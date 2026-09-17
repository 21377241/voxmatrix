#!/usr/bin/env python3
"""Rebuild formal A1/B report markdown from semantic_judge_a1b_full.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        type=Path,
        default=Path(
            "/mnt/afs/users/wangyl/VoxMatrix/output/a1b_paired_judge/semantic_judge_a1b_full.json"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B成对Judge自动稿.md"),
        help="Machine JSON-style draft only. Formal case-study report is locked.",
    )
    args = parser.parse_args()
    lock_token = "A1B_REPORT_LOCK"
    formal = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B音频增益实验报告.md")
    if args.report.resolve() == formal.resolve() or (
        args.report.is_file()
        and lock_token in args.report.read_text(encoding="utf-8", errors="ignore")
    ):
        alt = Path("/mnt/afs/users/wangyl/8p31/交付/开放语义_A1B成对Judge自动稿.md")
        print(f"refuse overwrite locked report {args.report}; writing {alt}")
        args.report = alt
    if not args.json.is_file():
        print(f"missing {args.json}")
        return 2
    payload = json.loads(args.json.read_text(encoding="utf-8"))
    summary = payload.get("summary") or {}
    cells = summary.get("cell_stats") or {}
    top = summary.get("top_abs_deltas") or []
    judge = payload.get("judge_model") or "qwen3-omni-thinking"

    lines: List[str] = [
        "# 开放语义 · A1 vs B 音频增益实验报告",
        "",
        "日期：2026-09-16",
        f"Judge：**同模** `{judge}`（A1=转写文本；B=同一转写+原音频）",
        "被测补跑：`qwen3-omni-audio`（Instruct）",
        "ASR：本地 Whisper-large-v3 / paraformer-zh（非 oracle/stub）",
        "",
        "## 1. 结论摘要",
        "",
        f"- 样本数：{summary.get('n_samples')}；成对有效：{summary.get('n_paired')}",
        f"- 精确一致率：{_fmt(summary.get('agree_exact_rate'))}",
        f"- 接近一致率（|Δ|≤20）：{_fmt(summary.get('agree_close_rate'))}",
        f"- Δ=B−A1 均值：{_fmt(summary.get('delta_mean'))}",
        "- 主结论看 **Δ / 一致率**（同族 Thinking 评 Omni 族时，绝对分慎作对外攀比）",
        "",
        "## 2. 设计锁定",
        "",
        "```",
        "同一 pred、同一 audio_transcript、同一 judge_model",
        "A1: text + transcript",
        "B:  text + transcript + WavPath",
        "主指标: Δ = score_B − score_A1",
        "```",
        "",
        "## 3. 按格成对统计",
        "",
        "| capability|rubric | n | n_paired | Δ均值 | 精确一致率 | 接近一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(cells.keys()):
        c = cells[key]
        lines.append(
            f"| {key} | {c.get('n')} | {c.get('n_paired')} | {_fmt(c.get('delta_mean'))} | "
            f"{_fmt(c.get('agree_exact_rate'))} | {_fmt(c.get('agree_close_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## 4. |Δ| Top",
            "",
            "```json",
            json.dumps(top[:15], ensure_ascii=False, indent=2),
            "```",
            "",
            "## 5. 产物",
            "",
            f"- JSON：`{args.json}`",
            f"- Pack：`{payload.get('pack')}`",
            f"- cell_counts：`{json.dumps(payload.get('cell_counts'), ensure_ascii=False)}`",
            "",
            "## 6. 披露",
            "",
            f"- judge_model={judge}",
            "- transcript=ASR local；tracks=A1,B；无 stub",
            "",
        ]
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines)
    if not body.startswith(">"):
        body = (
            "> 本文为 JSON 自动稿；正式样例分析见 `开放语义_A1B音频增益实验报告.md`。\n\n"
            + body
        )
    args.report.write_text(body, encoding="utf-8")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
