"""固定顺序标注流水线（P0+P1）。

P0: prelabel → collect_knowledge → verify_prelabel → assign_bucket
P1: mapping 对齐检查 + 子集策略 + 成本控制 + 版本化
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="P0+P1 标注流水线")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--raw-index", type=Path, default=None)
    parser.add_argument("--libri-root", type=Path, default=None)
    parser.add_argument("--libri-subset", default="train-clean-100")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--refresh-knowledge", action="store_true")
    parser.add_argument("--no-llm-verify", action="store_true")
    parser.add_argument(
        "--no-llm-knowledge",
        action="store_true",
        help="仅用本地种子事实生成知识包，不调用 LLM",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="完全离线：知识收集、知识核验和逐条标注都不调用 API",
    )
    parser.add_argument(
        "--with-llm-label",
        action="store_true",
        help="强制逐条 AI（覆盖同质跳过策略）",
    )
    parser.add_argument(
        "--audit-rate",
        type=float,
        default=None,
        help="抽检比例，默认读 mapping.cost_policy",
    )
    parser.add_argument("--no-strict", action="store_true")
    parser.add_argument("--skip-align-check", action="store_true")
    parser.add_argument(
        "--schema-version",
        choices=["1.0", "2.0"],
        default="2.0",
        help="New pipeline outputs default to canonical V2; V1 remains a compatibility option.",
    )
    args = parser.parse_args()
    if args.offline and args.with_llm_label:
        raise SystemExit("--offline 与 --with-llm-label 不能同时使用")

    py = sys.executable
    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)

    raw_index = args.raw_index or (work / "raw_index.jsonl")
    prelabeled = work / "prelabeled.jsonl"
    verified = work / "verified.jsonl"
    final_dir = work / "final"
    stages: dict = {}

    # P1-0: §13.1 对齐检查
    if not args.skip_align_check:
        align_report = work / "mapping_align_report.json"
        run(
            [
                py,
                "-m",
                "annotation.pipeline.check_mapping_align",
                "--dataset",
                args.dataset,
                "--report",
                str(align_report),
                "--strict",
            ]
        )
        stages["mapping_align"] = str(align_report)

    # 0) 索引
    if not raw_index.exists():
        if args.dataset == "librispeech":
            cmd = [
                py,
                "-m",
                "annotation.pipeline.index_librispeech",
                "--subset",
                args.libri_subset,
                "--output",
                str(raw_index),
            ]
            if args.libri_root:
                cmd.extend(["--root", str(args.libri_root)])
            if args.limit:
                cmd.extend(["--limit", str(args.limit)])
            run(cmd)
            stages["index"] = str(raw_index)
        else:
            raise SystemExit(f"缺少 raw_index: {raw_index}")

    # 1) 规则预标注（含 subset_policy）
    prelabel_cmd = [
        py,
        "-m",
        "annotation.pipeline.prelabel",
        "--dataset",
        args.dataset,
        "--input",
        str(raw_index),
        "--output",
        str(prelabeled),
    ]
    if args.limit:
        prelabel_cmd.extend(["--limit", str(args.limit)])
    run(prelabel_cmd)
    stages["prelabel"] = str(prelabeled)

    # 2) 知识收集
    collect_cmd = [py, "-m", "annotation.ai.collect_knowledge", "--dataset", args.dataset]
    if args.refresh_knowledge:
        collect_cmd.append("--force")
    if args.offline or args.no_llm_knowledge:
        collect_cmd.append("--no-llm")
    if args.offline or args.no_llm_knowledge:
        collect_cmd.append("--no-fetch")
    run(collect_cmd)
    stages["collect_knowledge"] = f"annotation/knowledge/cache/{args.dataset}.json"

    # 3) 知识核对
    verify_cmd = [
        py,
        "-m",
        "annotation.ai.verify_prelabel",
        "--dataset",
        args.dataset,
        "--input",
        str(prelabeled),
        "--output",
        str(verified),
        "--report",
        str(work / "verify_report.json"),
    ]
    if args.offline or args.no_llm_verify:
        verify_cmd.append("--no-llm")
    if args.no_strict:
        verify_cmd.append("--no-strict")
    run(verify_cmd)
    stages["verify"] = str(work / "verify_report.json")

    # P1: 成本策略
    cost_cmd = [
        py,
        "-m",
        "annotation.pipeline.cost_policy",
        "--dataset",
        args.dataset,
        "--verify-report",
        str(work / "verify_report.json"),
        "--input",
        str(verified),
        "--audit-output",
        str(work / "audit_sample.jsonl"),
    ]
    if args.audit_rate is not None:
        cost_cmd.extend(["--audit-rate", str(args.audit_rate)])
    if args.with_llm_label:
        cost_cmd.append("--force-llm-label")
    run(cost_cmd)

    # 解析是否跑 llm_label
    from annotation.pipeline.common import load_mapping
    from annotation.pipeline.cost_policy import should_run_llm_label

    mapping = load_mapping(args.dataset)
    verify_report = json.loads((work / "verify_report.json").read_text(encoding="utf-8"))
    run_llm, reason = should_run_llm_label(
        mapping, verify_report, force=args.with_llm_label
    )
    if args.offline:
        run_llm, reason = False, "offline_mode"
    stages["cost_policy"] = {"run_llm_label": run_llm, "reason": reason}

    labeled = verified
    if run_llm:
        labeled = work / "ai_labeled.jsonl"
        label_cmd = [
            py,
            "-m",
            "annotation.ai.llm_label",
            "--dataset",
            args.dataset,
            "--input",
            str(verified),
            "--output",
            str(labeled),
            "--fields",
            "capability",
        ]
        if args.limit:
            label_cmd.extend(["--limit", str(args.limit)])
        run(label_cmd)
        stages["llm_label"] = str(labeled)
    else:
        print(f"\n成本策略：跳过逐条 llm_label（{reason}）")
        print(f"抽检样本: {work / 'audit_sample.jsonl'}")

    # 4) 分桶
    final_dir.mkdir(parents=True, exist_ok=True)
    legacy_manifest = (
        final_dir / "manifest.v1.jsonl"
        if args.schema_version == "2.0"
        else final_dir / "manifest.jsonl"
    )
    run(
        [
            py,
            "-m",
            "annotation.pipeline.assign_bucket",
            "--input",
            str(labeled),
            "--output",
            str(legacy_manifest),
        ]
    )
    stages["assign_bucket"] = str(legacy_manifest)

    final_manifest = final_dir / "manifest.jsonl"
    if args.schema_version == "2.0":
        migration_report = final_dir / "migration_report.json"
        run(
            [
                py,
                "-m",
                "annotation.pipeline.migrate_v2",
                "--input",
                str(legacy_manifest),
                "--output",
                str(final_manifest),
                "--report",
                str(migration_report),
            ]
        )
        stages["migrate_v2"] = {
            "manifest": str(final_manifest),
            "report": str(migration_report),
        }
    else:
        final_manifest = legacy_manifest

    # 5) 校验
    run(
        [
            py,
            "-m",
            "annotation.pipeline.validate",
            "--manifest-file",
            str(final_manifest),
            "--schema",
            (
                "annotation/schema/evaluation_sample.v2.schema.json"
                if args.schema_version == "2.0"
                else "annotation/schema/manifest.schema.json"
            ),
        ]
    )
    stages["validate"] = "ok"

    # P1: 版本化
    from annotation.pipeline.versioning import write_run_manifest

    rm = write_run_manifest(
        work,
        args.dataset,
        stages=stages,
        extra={
            "limit": args.limit,
            "libri_subset": args.libri_subset,
            "offline": args.offline,
            "no_llm_knowledge": args.no_llm_knowledge,
            "no_llm_verify": args.no_llm_verify,
            "schema_version": args.schema_version,
        },
    )
    stages["run_manifest"] = str(rm)

    print("\n流水线完成（P0+P1）")
    print(f"  verified:      {verified}")
    print(f"  final:         {final_manifest}")
    print(f"  verify_report: {work / 'verify_report.json'}")
    print(f"  audit_sample:  {work / 'audit_sample.jsonl'}")
    print(f"  run_manifest:  {rm}")
    print(f"  buckets:       {final_dir / 'bucket_summary.json'}")


if __name__ == "__main__":
    main()
