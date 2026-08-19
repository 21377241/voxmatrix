"""AI 成本控制与抽检。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from annotation.pipeline.common import load_jsonl, load_mapping, write_jsonl


def is_homogeneous(mapping: dict[str, Any], verify_report: dict[str, Any] | None = None) -> bool:
    """同质数据集：verify 通过且 mapping.cost_policy.homogeneous=true。"""
    policy = mapping.get("cost_policy") or {}
    if policy.get("homogeneous") is False:
        return False
    if policy.get("homogeneous") is True:
        if verify_report:
            gate = (verify_report.get("gate") or verify_report.get("decision", {}).get("gate"))
            return gate in {"pass", "pass_with_revise"}
        return True
    # 默认：verify pass 且单一 capability 视作同质
    if not verify_report:
        return False
    gate = verify_report.get("gate") or (verify_report.get("decision") or {}).get("gate")
    caps = (verify_report.get("prelabel_stats") or {}).get("capability") or {}
    return gate in {"pass", "pass_with_revise"} and len(caps) <= 1


def should_run_llm_label(
    mapping: dict[str, Any],
    verify_report: dict[str, Any] | None,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    if force:
        return True, "force"
    policy = mapping.get("cost_policy") or {}
    if policy.get("skip_llm_label_if_verify_pass", True) and is_homogeneous(mapping, verify_report):
        return False, "homogeneous_verify_pass_skip_llm_label"
    gate = None
    if verify_report:
        gate = verify_report.get("gate") or (verify_report.get("decision") or {}).get("gate")
    if gate in {"needs_review", "blocked"}:
        return False, f"gate={gate}_skip_until_human"
    return True, "heterogeneous_or_policy_requires"


def audit_sample(
    samples: list[dict[str, Any]],
    rate: float,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if rate <= 0:
        return []
    if rate >= 1:
        return list(samples)
    rng = random.Random(seed)
    k = max(1, int(len(samples) * rate)) if samples else 0
    if k >= len(samples):
        return list(samples)
    return rng.sample(samples, k)


def main() -> None:
    parser = argparse.ArgumentParser(description="成本策略与抽检")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--verify-report", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="抽检用 jsonl")
    parser.add_argument("--audit-output", type=Path, default=None)
    parser.add_argument("--audit-rate", type=float, default=None, help="覆盖 mapping 中的抽检比例")
    parser.add_argument("--force-llm-label", action="store_true")
    args = parser.parse_args()

    mapping = load_mapping(args.dataset)
    report = json.loads(args.verify_report.read_text(encoding="utf-8"))
    run_llm, reason = should_run_llm_label(
        mapping, report, force=args.force_llm_label
    )
    policy = mapping.get("cost_policy") or {}
    rate = args.audit_rate
    if rate is None:
        rate = float(policy.get("audit_sample_rate", 0.01))

    decision = {
        "dataset": args.dataset,
        "run_llm_label": run_llm,
        "reason": reason,
        "homogeneous": is_homogeneous(mapping, report),
        "audit_sample_rate": rate,
        "verify_gate": report.get("gate") or (report.get("decision") or {}).get("gate"),
    }
    print(json.dumps(decision, ensure_ascii=False, indent=2))

    if args.input and args.audit_output:
        samples = load_jsonl(args.input)
        audited = audit_sample(samples, rate)
        for s in audited:
            s.setdefault("label_meta", {})["audit_sample"] = True
        write_jsonl(args.audit_output, audited)
        print(f"抽检 {len(audited)}/{len(samples)} → {args.audit_output}")


if __name__ == "__main__":
    main()
