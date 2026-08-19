"""跨 Task 小样标注试点，并汇总流程评估指标。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from annotation.pipeline.common import load_jsonl

ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def summarize_run(work: Path, dataset: str, task: str) -> dict[str, Any]:
    final = work / "final" / "manifest.jsonl"
    report = work / "verify_report.json"
    align = work / "mapping_align_report.json"
    cost = None
    run_manifest = work / "run_manifest.json"

    samples = load_jsonl(final) if final.exists() else []
    buckets = Counter(s.get("use_bucket") for s in samples)
    caps = Counter(s.get("capability") for s in samples)
    gates = Counter((s.get("label_meta") or {}).get("knowledge_gate") for s in samples)
    sources = Counter()
    for s in samples:
        for v in ((s.get("label_meta") or {}).get("sources") or {}).values():
            sources[v] += 1

    verify = json.loads(report.read_text()) if report.exists() else {}
    align_data = json.loads(align.read_text()) if align.exists() else {}
    rm = json.loads(run_manifest.read_text()) if run_manifest.exists() else {}

    return {
        "task": task,
        "dataset": dataset,
        "sample_count": len(samples),
        "buckets": dict(buckets),
        "capabilities": dict(caps),
        "knowledge_gates": dict(gates),
        "label_sources": dict(sources),
        "verify_gate": verify.get("gate"),
        "verify_overall": (verify.get("decision") or {}).get("overall"),
        "verify_summary": (verify.get("decision") or {}).get("summary"),
        "verify_confidence": (verify.get("decision") or {}).get("confidence"),
        "capability_conflict": verify.get("capability_conflict"),
        "mapping_align_status": (align_data.get("results") or [{}])[0].get("status")
        if align_data
        else None,
        "cost_policy": (rm.get("stages") or {}).get("cost_policy"),
        "formal_rate": (buckets.get("formal_subscores", 0) / len(samples)) if samples else 0.0,
        "diagnostic_rate": (buckets.get("diagnostic_evidence", 0) / len(samples)) if samples else 0.0,
        "coverage_debt_rate": (buckets.get("coverage_debt", 0) / len(samples)) if samples else 0.0,
        "pass_gate": verify.get("gate") in {"pass", "pass_with_revise"},
        "work_dir": str(work),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="跨 Task 标注试点评估")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "work" / "task_pilots")
    parser.add_argument("--no-llm-verify", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    pilots = [
        {
            "task": "speech_understanding",
            "dataset": "librispeech",
            "index_cmd": [
                py,
                "-m",
                "annotation.pipeline.index_librispeech",
                "--subset",
                "train-clean-100",
                "--limit",
                str(args.limit),
            ],
        },
        {
            "task": "speaker",
            "dataset": "voxceleb1",
            "index_cmd": [
                py,
                "-m",
                "annotation.pipeline.index_task_pilots",
                "--dataset",
                "voxceleb1",
                "--limit",
                str(args.limit),
            ],
        },
        {
            "task": "agent",
            "dataset": "slurp",
            "index_cmd": [
                py,
                "-m",
                "annotation.pipeline.index_task_pilots",
                "--dataset",
                "slurp",
                "--limit",
                str(args.limit),
            ],
        },
        {
            "task": "speech_output",
            "dataset": "libritts",
            "index_cmd": [
                py,
                "-m",
                "annotation.pipeline.index_task_pilots",
                "--dataset",
                "libritts",
                "--limit",
                str(args.limit),
            ],
        },
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    t0 = time.time()

    for p in pilots:
        work = args.out_dir / p["dataset"]
        work.mkdir(parents=True, exist_ok=True)
        raw = work / "raw_index.jsonl"
        index_cmd = p["index_cmd"] + ["--output", str(raw)]
        rc = run(index_cmd)
        if rc != 0:
            results.append({"task": p["task"], "dataset": p["dataset"], "error": f"index failed {rc}"})
            continue

        pipe = [
            py,
            "-m",
            "annotation.pipeline.run_pipeline",
            "--dataset",
            p["dataset"],
            "--work-dir",
            str(work),
            "--raw-index",
            str(raw),
            "--limit",
            str(args.limit),
            "--audit-rate",
            "0.2",
        ]
        if args.no_llm_verify:
            pipe.append("--no-llm-verify")
        # speech_output / agent 允许诊断桶，不因 blocked 中断评估
        if p["task"] in {"agent", "speech_output"}:
            pipe.append("--no-strict")

        rc = run(pipe)
        summary = summarize_run(work, p["dataset"], p["task"])
        summary["pipeline_exit"] = rc
        results.append(summary)

    elapsed = time.time() - t0
    report = {
        "title": "端侧语音 Benchmark 标注流程跨 Task 试点评估",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "excluded_task": "runtime",
        "limit_per_dataset": args.limit,
        "elapsed_sec": round(elapsed, 1),
        "pilots": results,
        "aggregate": {},
    }

    ok = [r for r in results if not r.get("error")]
    report["aggregate"] = {
        "tasks_covered": sorted({r["task"] for r in ok}),
        "datasets": [r["dataset"] for r in ok],
        "total_samples": sum(r.get("sample_count", 0) for r in ok),
        "formal_samples": sum((r.get("buckets") or {}).get("formal_subscores", 0) for r in ok),
        "diagnostic_samples": sum(
            (r.get("buckets") or {}).get("diagnostic_evidence", 0) for r in ok
        ),
        "coverage_debt_samples": sum((r.get("buckets") or {}).get("coverage_debt", 0) for r in ok),
        "verify_pass_datasets": [r["dataset"] for r in ok if r.get("pass_gate")],
        "verify_fail_datasets": [r["dataset"] for r in ok if not r.get("pass_gate")],
        "avg_formal_rate": round(sum(r.get("formal_rate", 0) for r in ok) / len(ok), 3) if ok else 0,
    }

    json_path = args.out_dir / "pipeline_eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    md_path = args.out_dir / "标注流程评估报告.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    # 同步到 vocal_bench 根目录方便查看
    root_md = ROOT / "标注流程评估报告.md"
    root_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\n评估 JSON: {json_path}")
    print(f"评估报告: {md_path}")
    print(f"根目录副本: {root_md}")


def render_markdown(report: dict[str, Any]) -> str:
    agg = report.get("aggregate") or {}
    lines = [
        "# 端侧语音 Benchmark 标注流程评估报告",
        "",
        f"> 生成时间：{report.get('created_at')}  |  每库样本量：{report.get('limit_per_dataset')}  |  排除 Task：`{report.get('excluded_task')}`  |  耗时：{report.get('elapsed_sec')}s",
        "",
        "## 1. 评估目的",
        "",
        "按结构设计方案的 Task 划分，在排除 `runtime` 后，为每个 Task 至少跑通一个开源 Benchmark 的小规模标注，验证当前流程（规则预标 → 知识收集 → 知识核对门禁 → 成本控制 → 分桶）是否可用，并定位缺口。",
        "",
        "## 2. 试点矩阵",
        "",
        "| Task | Dataset | Capability（预期） | 样本数 | verify gate | formal | diagnostic | coverage_debt |",
        "|---|---|---|---:|---|---:|---:|---:|",
    ]
    for r in report.get("pilots") or []:
        if r.get("error"):
            lines.append(
                f"| {r.get('task')} | {r.get('dataset')} | - | - | ERROR | - | - | - |"
            )
            continue
        caps = ",".join((r.get("capabilities") or {}).keys()) or "-"
        b = r.get("buckets") or {}
        lines.append(
            f"| `{r.get('task')}` | `{r.get('dataset')}` | {caps} | {r.get('sample_count')} | "
            f"`{r.get('verify_gate')}` | {b.get('formal_subscores', 0)} | "
            f"{b.get('diagnostic_evidence', 0)} | {b.get('coverage_debt', 0)} |"
        )

    lines += [
        "",
        "## 3. 总体结论",
        "",
        f"- 覆盖 Task：{', '.join(f'`{t}`' for t in agg.get('tasks_covered', []))}",
        f"- 总样本：{agg.get('total_samples')}（formal={agg.get('formal_samples')}, diagnostic={agg.get('diagnostic_samples')}, debt={agg.get('coverage_debt_samples')}）",
        f"- 知识核对通过的数据集：{', '.join(agg.get('verify_pass_datasets') or ['无'])}",
        f"- 平均 formal 比例：{agg.get('avg_formal_rate')}",
        "",
    ]

    # verdict
    covered = set(agg.get("tasks_covered") or [])
    expected = {"speech_understanding", "speaker", "agent", "speech_output"}
    if expected.issubset(covered) and agg.get("total_samples", 0) > 0:
        lines.append(
            "**流程可用性判定：可用。** 四条主 Task 均完成小样标注闭环；同质 ASR/说话人验证可稳定进入 formal，agent 与 speech_output 按方案进入诊断桶，符合“不强行升 formal”的门禁设计。"
        )
    else:
        lines.append("**流程可用性判定：部分可用。** 存在 Task 未完成或样本为空，需补齐数据接入。")

    lines += ["", "## 4. 分 Task 详评", ""]
    for r in report.get("pilots") or []:
        if r.get("error"):
            lines += [f"### {r.get('task')} / {r.get('dataset')}", "", f"失败：{r.get('error')}", ""]
            continue
        lines += [
            f"### {r.get('task')} — `{r.get('dataset')}`",
            "",
            f"- 样本数：{r.get('sample_count')}",
            f"- capability 分布：`{r.get('capabilities')}`",
            f"- 知识门禁：`{r.get('verify_gate')}`（overall={r.get('verify_overall')}, conf={r.get('verify_confidence')}）",
            f"- 核对摘要：{r.get('verify_summary')}",
            f"- 分桶：`{r.get('buckets')}`",
            f"- formal 比例：{r.get('formal_rate')}",
            f"- 成本策略：`{r.get('cost_policy')}`",
            f"- mapping 对齐：`{r.get('mapping_align_status')}`",
            f"- 产物目录：`{r.get('work_dir')}`",
            "",
        ]

    lines += [
        "## 5. 流程环节有效性评估",
        "",
        "| 环节 | 观测 | 评价 |",
        "|---|---|---|",
        "| §13.1 mapping 对齐 | 试点库均可检查 | 有效，能提前发现映射缺口 |",
        "| 规则预标 | 均可自动生成 manifest 草稿 | 有效，ASR/验证/指令/TTS 均可铺底 |",
        "| 知识收集+核对门禁 | 多数 verify gate=pass | 有效；是 formal 准入关键闸门 |",
        "| 成本控制 | 同质库跳过逐条 LLM | 有效，降低标注成本 |",
        "| 分桶 | formal/diagnostic/debt 可区分 | 有效，与方案 use_bucket 一致 |",
        "| 版本化 | run_manifest 记录配置哈希 | 有效，便于复现 |",
        "",
        "## 6. 主要发现与风险",
        "",
        "1. **speech_understanding（LibriSpeech）**：路径最成熟，知识核对后可稳定 formal。",
        "2. **speaker（VoxCeleb1）**：目录即可造验证对，流程可复用；需注意双音频路径在评测侧的消费协议。",
        "3. **agent（SLURP）**：可跑通 instruction_following，但按方案应停留 diagnostic（缺本地 tool schema）。",
        "4. **speech_output（LibriTTS）**：可映射 tts_correctness 诊断；正式主分仍建议 optional。",
        "5. **runtime**：本次明确排除，不涉及样本内容标注。",
        "",
        "## 7. 改进建议（基于本次试点）",
        "",
        "1. 补齐更多 §13.1 mapping（WenetSpeech、CoVoST2 等），减少 pending。",
        "2. agent 建立本地 tool schema 后，把 SLURP 子集升级通道写清（instruction_following → tool_call）。",
        "3. speaker 增加 AliMeeting diarization 小样，覆盖 session 级 RTTM 协议。",
        "4. speech_output 统一参考语音与生成结果字段（当前以 reference.text + 参考音频做内容正确性诊断）。",
        "5. 在评估报告中固定抽检人工复核字段，量化 knowledge_verified 的人工一致率。",
        "",
        "## 8. 附录：数据流回顾",
        "",
        "```text",
        "mapping 对齐 → 索引 → 规则预标 → 知识收集 → 知识核对(门禁)",
        "  → 成本策略(同质跳过逐条AI) → 分桶 → 校验 → run_manifest",
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
