"""基于知识包核对规则预标注（P0 门禁）。

固定位于：collect_knowledge 之后、逐条 llm_label / 分桶之前。

分流：
  confirm → gate=pass，可冲正式分
  revise  → 应用修正后 gate=pass_with_revise；capability 冲突须显式决议
  flag    → gate=needs_review，默认 diagnostic
  reject / capability 未决议冲突 → gate=blocked，禁止 formal
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from annotation.ai.client import get_client
from annotation.ai.collect_knowledge import CACHE_DIR, collect
from annotation.pipeline.common import load_jsonl, load_taxonomy, write_jsonl

VERIFY_PROMPT = """你是端侧语音 Benchmark 标注质检员。
请用「数据集知识包」核对「规则预标注汇总」，判断哪些标签可确认、哪些应修正。

重点优先顺序：
1. capability（最重要，决定评测协议；冲突必须明确 revise 或 flag，禁止含糊）
2. task
3. scenario / condition
4. metrics / use_bucket

不要改动 reference 文本内容（原生字幕）。

## 允许枚举
{taxonomy_json}

## 数据集知识包
{knowledge_json}

## 规则预标注汇总（来自 prelabeled.jsonl 统计，含 by_subset）
{prelabel_stats}

## 子集策略（mapping.subset_policy，若有）
{subset_policy_json}

## 预先检测到的 capability 冲突
{capability_conflict_json}

## 输出 JSON
{{
  "overall": "confirm|revise|reject",
  "confidence": 0.0,
  "summary": "2-3句中文结论",
  "capability_resolution": {{
    "status": "agree|revise_to_knowledge|flag|reject",
    "rule_value": "...",
    "knowledge_value": "...",
    "final_value": "...",
    "reason": "..."
  }},
  "field_decisions": [
    {{
      "field": "capability|task|scenario|condition.acoustic|...",
      "action": "keep|revise|flag",
      "rule_value": "...",
      "verified_value": "...",
      "reason": "..."
    }}
  ],
  "apply_patch": {{
    "task": null,
    "capability": null,
    "scenario": null,
    "condition": {{}},
    "metrics": null,
    "use_bucket": null
  }},
  "notes": ["..."]
}}

规则：
1. action=keep 时 verified_value=rule_value；apply_patch 对应项填 null
2. action=revise 时必须给出 verified_value，并写入 apply_patch
3. action=flag：证据不足，不强行改值
4. 若存在 capability 冲突：必须在 capability_resolution 给出决议，且 overall 不能是 confirm
5. 只输出 JSON
"""


def load_knowledge(
    dataset_id: str, *, refresh: bool = False, use_llm: bool = True
) -> dict[str, Any]:
    cache_path = CACHE_DIR / f"{dataset_id}.json"
    if refresh or not cache_path.exists():
        collect(
            dataset_id,
            use_llm=use_llm,
            force=refresh,
            fetch_sources=use_llm,
        )
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def knowledge_capability(knowledge: dict[str, Any]) -> Any:
    profile = knowledge.get("profile") or {}
    suggested = profile.get("suggested_mapping") or {}
    seed = knowledge.get("seed_facts") or {}
    typical = seed.get("typical_labels") or {}
    cap = (
        seed.get("primary_capability")
        or typical.get("capability")
        or suggested.get("capability")
    )
    if isinstance(cap, list):
        return cap[0] if cap else None
    return cap


def detect_capability_conflict(
    stats: dict[str, Any],
    knowledge: dict[str, Any],
) -> dict[str, Any] | None:
    """规则主导 capability 与知识包不一致时返回冲突描述。"""
    rule_caps = stats.get("capability") or {}
    if not rule_caps:
        return {
            "has_conflict": True,
            "rule_value": None,
            "knowledge_value": knowledge_capability(knowledge),
            "reason": "prelabeled 缺少 capability",
        }
    # 若多 capability 并存也视为需审
    if len(rule_caps) > 1:
        return {
            "has_conflict": True,
            "rule_value": rule_caps,
            "knowledge_value": knowledge_capability(knowledge),
            "reason": "规则预标注存在多种 capability，需知识裁定",
            "multi_capability": True,
        }
    rule_cap = next(iter(rule_caps))
    know_cap = knowledge_capability(knowledge)
    if know_cap and str(rule_cap) != str(know_cap):
        return {
            "has_conflict": True,
            "rule_value": rule_cap,
            "knowledge_value": know_cap,
            "reason": "规则 capability 与知识包 primary/suggested 不一致",
        }
    return {
        "has_conflict": False,
        "rule_value": rule_cap,
        "knowledge_value": know_cap or rule_cap,
        "reason": "一致",
    }


def summarize_prelabeled(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def count_field(getter) -> dict[str, int]:
        c: Counter = Counter()
        for s in samples:
            v = getter(s)
            if isinstance(v, dict):
                v = json.dumps(v, ensure_ascii=False, sort_keys=True)
            c[str(v)] += 1
        return dict(c.most_common(10))

    subsets = Counter(
        s.get("label_meta", {}).get("raw_subset") or s.get("split") or "unknown"
        for s in samples
    )

    by_subset: dict[str, Any] = {}
    for s in samples:
        sub = s.get("label_meta", {}).get("raw_subset") or "unknown"
        slot = by_subset.setdefault(
            sub,
            {"count": 0, "capability": Counter(), "acoustic": Counter()},
        )
        slot["count"] += 1
        slot["capability"][s.get("capability")] += 1
        slot["acoustic"][s.get("condition", {}).get("acoustic")] += 1
    by_subset_json = {
        k: {
            "count": v["count"],
            "capability": dict(v["capability"]),
            "acoustic": dict(v["acoustic"]),
        }
        for k, v in by_subset.items()
    }

    return {
        "sample_count": len(samples),
        "dataset": samples[0].get("dataset") if samples else None,
        "subset_distribution": dict(subsets),
        "by_subset": by_subset_json,
        "task": count_field(lambda s: s.get("task")),
        "capability": count_field(lambda s: s.get("capability")),
        "scenario": count_field(lambda s: s.get("scenario")),
        "condition.acoustic": count_field(lambda s: s.get("condition", {}).get("acoustic")),
        "condition.spatial": count_field(lambda s: s.get("condition", {}).get("spatial")),
        "condition.speaker": count_field(lambda s: s.get("condition", {}).get("speaker")),
        "condition.device": count_field(lambda s: s.get("condition", {}).get("device")),
        "condition.interaction": count_field(
            lambda s: s.get("condition", {}).get("interaction")
        ),
        "metrics": count_field(lambda s: ",".join(s.get("metrics") or [])),
        "use_bucket": count_field(lambda s: s.get("use_bucket")),
        "example_sample": {
            "sample_id": samples[0].get("sample_id"),
            "task": samples[0].get("task"),
            "capability": samples[0].get("capability"),
            "scenario": samples[0].get("scenario"),
            "condition": samples[0].get("condition"),
            "metrics": samples[0].get("metrics"),
            "use_bucket": samples[0].get("use_bucket"),
            "subset": samples[0].get("label_meta", {}).get("raw_subset"),
            "reference_preview": str(samples[0].get("reference", {}).get("text", ""))[:120],
        }
        if samples
        else None,
    }


def knowledge_compact(knowledge: dict[str, Any]) -> str:
    profile = knowledge.get("profile") or {}
    seed = knowledge.get("seed_facts") or {}
    return json.dumps(
        {
            "dataset_id": knowledge.get("dataset_id"),
            "seed_facts": seed,
            "profile": {
                "summary": profile.get("summary"),
                "suggested_mapping": profile.get("suggested_mapping"),
                "labeling_guidelines": profile.get("labeling_guidelines"),
                "missing_labels": profile.get("missing_labels"),
                "primary_capability": knowledge_capability(knowledge),
            },
        },
        ensure_ascii=False,
        indent=2,
    )


def run_verification(
    client: Any,
    knowledge: dict[str, Any],
    stats: dict[str, Any],
    taxonomy: dict[str, Any],
    capability_conflict: dict[str, Any],
    subset_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    taxonomy_json = json.dumps(
        {
            "tasks": taxonomy.get("tasks"),
            "capabilities": taxonomy.get("capabilities"),
            "scenarios": taxonomy.get("scenarios"),
            "conditions": taxonomy.get("conditions"),
            "use_buckets": taxonomy.get("use_buckets"),
        },
        ensure_ascii=False,
        indent=2,
    )
    prompt = VERIFY_PROMPT.format(
        taxonomy_json=taxonomy_json,
        knowledge_json=knowledge_compact(knowledge),
        prelabel_stats=json.dumps(stats, ensure_ascii=False, indent=2),
        subset_policy_json=json.dumps(subset_policy or {}, ensure_ascii=False, indent=2),
        capability_conflict_json=json.dumps(capability_conflict, ensure_ascii=False, indent=2),
    )
    return client.chat_json([{"role": "user", "content": prompt}])


def mechanical_fallback(
    knowledge: dict[str, Any],
    stats: dict[str, Any],
    capability_conflict: dict[str, Any],
) -> dict[str, Any]:
    profile = knowledge.get("profile") or {}
    suggested = profile.get("suggested_mapping") or {}
    seed = knowledge.get("seed_facts") or {}
    typical = seed.get("typical_labels") or suggested

    dominant_cap = next(iter(stats.get("capability", {})), None)
    knowledge_cap = knowledge_capability(knowledge)

    field_decisions = []
    apply_patch: dict[str, Any] = {
        "task": None,
        "capability": None,
        "scenario": None,
        "condition": {},
        "metrics": None,
        "use_bucket": None,
    }
    overall = "confirm"
    cap_resolution = {
        "status": "agree",
        "rule_value": dominant_cap,
        "knowledge_value": knowledge_cap,
        "final_value": dominant_cap,
        "reason": "一致",
    }

    def decide(field: str, rule_val: Any, know_val: Any) -> None:
        nonlocal overall
        if know_val is None or know_val == "":
            field_decisions.append(
                {
                    "field": field,
                    "action": "flag",
                    "rule_value": rule_val,
                    "verified_value": rule_val,
                    "reason": "知识包未给出该字段",
                }
            )
            return
        if str(rule_val) == str(know_val):
            field_decisions.append(
                {
                    "field": field,
                    "action": "keep",
                    "rule_value": rule_val,
                    "verified_value": know_val,
                    "reason": "与知识包一致",
                }
            )
        else:
            overall = "revise"
            field_decisions.append(
                {
                    "field": field,
                    "action": "revise",
                    "rule_value": rule_val,
                    "verified_value": know_val,
                    "reason": "与知识包不一致，采用知识包",
                }
            )
            if field.startswith("condition."):
                apply_patch["condition"][field.split(".", 1)[1]] = know_val
            else:
                apply_patch[field] = know_val

    # capability：冲突时默认采用知识包并标 revise；知识包缺失则 flag+reject
    if capability_conflict.get("has_conflict"):
        if knowledge_cap:
            overall = "revise"
            cap_resolution = {
                "status": "revise_to_knowledge",
                "rule_value": capability_conflict.get("rule_value"),
                "knowledge_value": knowledge_cap,
                "final_value": knowledge_cap,
                "reason": capability_conflict.get("reason"),
            }
            apply_patch["capability"] = knowledge_cap
            field_decisions.append(
                {
                    "field": "capability",
                    "action": "revise",
                    "rule_value": dominant_cap,
                    "verified_value": knowledge_cap,
                    "reason": capability_conflict.get("reason"),
                }
            )
        else:
            overall = "reject"
            cap_resolution = {
                "status": "reject",
                "rule_value": dominant_cap,
                "knowledge_value": None,
                "final_value": None,
                "reason": "存在冲突且知识包无 capability，阻断正式发布",
            }
            field_decisions.append(
                {
                    "field": "capability",
                    "action": "flag",
                    "rule_value": dominant_cap,
                    "verified_value": dominant_cap,
                    "reason": "capability 冲突且无法自动决议",
                }
            )
    else:
        decide("capability", dominant_cap, knowledge_cap or dominant_cap)
        cap_resolution["final_value"] = dominant_cap

    decide("task", next(iter(stats.get("task", {})), None), typical.get("task"))
    decide("scenario", next(iter(stats.get("scenario", {})), None), typical.get("scenario"))
    cond = typical.get("condition") or {}
    for ck in ("acoustic", "spatial", "speaker", "device", "interaction"):
        decide(
            f"condition.{ck}",
            next(iter(stats.get(f"condition.{ck}", {})), None),
            cond.get(ck),
        )

    # 有 flag 则至少不能 confirm
    if any(d.get("action") == "flag" for d in field_decisions) and overall == "confirm":
        overall = "revise"

    return {
        "overall": overall,
        "confidence": 0.75,
        "summary": "机械核对：对比规则主导标签与知识包",
        "capability_resolution": cap_resolution,
        "field_decisions": field_decisions,
        "apply_patch": apply_patch,
        "notes": ["no_llm fallback"],
    }


def normalize_decision(
    decision: dict[str, Any],
    capability_conflict: dict[str, Any],
) -> dict[str, Any]:
    """统一 gate 分流字段，补全 capability 冲突决议。"""
    out = deepcopy(decision)
    overall = (out.get("overall") or "reject").lower()
    if overall not in {"confirm", "revise", "reject"}:
        overall = "reject"
        out["overall"] = overall

    field_decisions = out.get("field_decisions") or []
    has_flag = any(d.get("action") == "flag" for d in field_decisions)
    cap_res = out.get("capability_resolution") or {}

    # 冲突未决议 → 阻断
    if capability_conflict.get("has_conflict"):
        status = (cap_res.get("status") or "").lower()
        final_cap = cap_res.get("final_value")
        patch_cap = (out.get("apply_patch") or {}).get("capability")
        if status in {"", "agree"} and not final_cap and not patch_cap:
            # LLM 漏写决议
            out["overall"] = "reject"
            overall = "reject"
            out["capability_resolution"] = {
                "status": "reject",
                "rule_value": capability_conflict.get("rule_value"),
                "knowledge_value": capability_conflict.get("knowledge_value"),
                "final_value": None,
                "reason": "capability 冲突未给出有效决议，阻断发布",
            }
            cap_res = out["capability_resolution"]
        elif status == "revise_to_knowledge" or (patch_cap or final_cap):
            # 确保 patch 带上 final capability
            resolved = patch_cap or final_cap or capability_conflict.get("knowledge_value")
            out.setdefault("apply_patch", {})["capability"] = resolved
            cap_res["final_value"] = resolved
            cap_res["status"] = cap_res.get("status") or "revise_to_knowledge"
            out["capability_resolution"] = cap_res
            if overall == "confirm":
                out["overall"] = "revise"
                overall = "revise"
        elif status in {"flag", "reject"}:
            out["overall"] = "reject"
            overall = "reject"

    # 计算 gate
    if overall == "reject" or (cap_res.get("status") or "").lower() in {"flag", "reject"}:
        gate = "blocked"
        route_bucket = "coverage_debt"
    elif has_flag or overall != "confirm" and any(
        d.get("action") == "flag" for d in field_decisions
    ):
        gate = "needs_review"
        route_bucket = "diagnostic_evidence"
    elif overall == "revise":
        gate = "pass_with_revise"
        route_bucket = "formal_subscores"  # 修正后仍可冲正式分，由分桶再判
    else:
        gate = "pass"
        route_bucket = "formal_subscores"

    # flag 但 overall 不是 reject → needs_review
    if has_flag and gate == "pass":
        gate = "needs_review"
        route_bucket = "diagnostic_evidence"
    if has_flag and gate == "pass_with_revise":
        gate = "needs_review"
        route_bucket = "diagnostic_evidence"

    out["gate"] = gate
    out["route_bucket_hint"] = route_bucket
    out["capability_conflict_input"] = capability_conflict
    return out


def apply_patch_to_sample(
    sample: dict[str, Any],
    decision: dict[str, Any],
    taxonomy: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    out = deepcopy(sample)
    meta = out.setdefault("label_meta", {})
    sources = meta.setdefault("sources", {})
    changes: list[str] = []

    patch = decision.get("apply_patch") or {}
    field_decisions = decision.get("field_decisions") or []
    decisions_by_field = {d.get("field"): d for d in field_decisions if d.get("field")}
    gate = decision.get("gate", "blocked")

    def set_source(field: str, action: str) -> None:
        if action == "keep":
            sources[field] = "knowledge_verified"
        elif action == "revise":
            sources[field] = "knowledge_revise"
        elif action == "flag":
            sources[field] = "knowledge_flagged"

    for key in ("task", "capability", "scenario", "metrics", "use_bucket"):
        new_val = patch.get(key)
        if new_val is None:
            continue
        if out.get(key) != new_val:
            out[key] = new_val
            changes.append(key)
        set_source(key, "revise")

    cond_patch = patch.get("condition") or {}
    if isinstance(cond_patch, dict):
        out.setdefault("condition", {})
        for ck, cv in cond_patch.items():
            if cv is None:
                continue
            if out["condition"].get(ck) != cv:
                out["condition"][ck] = cv
                changes.append(f"condition.{ck}")
            set_source(f"condition.{ck}", "revise")

    for field, d in decisions_by_field.items():
        action = d.get("action", "keep")
        if action == "keep":
            set_source(field, "keep")
        elif action == "flag":
            set_source(field, "flag")
            meta.setdefault("knowledge_flags", []).append(
                {"field": field, "reason": d.get("reason")}
            )

    cap = out.get("capability")
    default_metrics = taxonomy.get("default_metrics", {}).get(cap)
    if default_metrics and ("capability" in changes or not out.get("metrics")):
        out["metrics"] = list(default_metrics)

    meta["knowledge_verify_done"] = True
    meta["knowledge_verified"] = gate in {"pass", "pass_with_revise"}
    meta["knowledge_gate"] = gate
    meta["knowledge_verify_overall"] = decision.get("overall")
    meta["knowledge_verify_confidence"] = decision.get("confidence")
    meta["knowledge_verify_summary"] = decision.get("summary")
    meta["knowledge_route_bucket_hint"] = decision.get("route_bucket_hint")
    meta["capability_resolution"] = decision.get("capability_resolution")

    if gate in {"needs_review", "blocked"}:
        meta["needs_human_review"] = True
        meta["reviewed"] = False
    elif gate == "pass":
        # 知识确认通过：视为规则+知识双重确认，可免 llm 再审（仍可人工抽检）
        meta["needs_human_review"] = False
    elif gate == "pass_with_revise":
        meta["needs_human_review"] = False  # 已按知识修正；抽检即可

    # 分流预填 bucket hint（最终仍由 assign_bucket 裁决）
    if gate == "blocked":
        out["use_bucket"] = "coverage_debt"
    elif gate == "needs_review":
        out["use_bucket"] = "diagnostic_evidence"

    return out, changes


def write_branch_outputs(
    output: Path,
    samples: list[dict[str, Any]],
) -> dict[str, int]:
    """按 gate 拆分写入 pass / review / blocked 文件。"""
    buckets = {
        "pass": [],
        "pass_with_revise": [],
        "needs_review": [],
        "blocked": [],
    }
    for s in samples:
        gate = s.get("label_meta", {}).get("knowledge_gate", "blocked")
        buckets.setdefault(gate, []).append(s)

    out_dir = output.parent
    stem = output.stem
    counts = {}
    for gate, rows in buckets.items():
        counts[gate] = len(rows)
        if not rows:
            continue
        path = out_dir / f"{stem}.{gate}.jsonl"
        write_jsonl(path, rows)
    write_jsonl(output, samples)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="用知识包核对规则预标注（P0 门禁）")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True, type=Path, help="prelabeled.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="核对后 jsonl")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--refresh-knowledge", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="capability 冲突未决议或 gate=blocked 时以非零退出（默认开启）",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="关闭严格模式，blocked 仍写文件但不失败退出",
    )
    args = parser.parse_args()
    strict = args.strict and not args.no_strict

    knowledge = load_knowledge(
        args.dataset,
        refresh=args.refresh_knowledge,
        use_llm=not args.no_llm,
    )
    samples = load_jsonl(args.input)
    if not samples:
        raise SystemExit("prelabeled 为空")

    taxonomy = load_taxonomy()
    stats = summarize_prelabeled(samples)
    capability_conflict = detect_capability_conflict(stats, knowledge)

    from annotation.pipeline.common import load_mapping

    mapping = load_mapping(args.dataset)
    subset_policy = mapping.get("subset_policy") or {}

    # P1：子集策略机械校验（不等 LLM）
    subset_mismatches = []
    for sub, dist in (stats.get("by_subset") or {}).items():
        cfg = (subset_policy.get("subsets") or {}).get(sub)
        if not cfg:
            continue
        expect_ac = (cfg.get("condition") or {}).get("acoustic")
        actual_acs = dist.get("acoustic") or {}
        if expect_ac and expect_ac not in actual_acs:
            subset_mismatches.append(
                {"subset": sub, "field": "condition.acoustic", "expected": expect_ac, "actual": actual_acs}
            )
        expect_cap = cfg.get("capability")
        actual_caps = dist.get("capability") or {}
        if expect_cap and expect_cap not in actual_caps and len(actual_caps) == 1:
            # 可能全部被 long_form 升级
            if "long_form_asr" not in actual_caps:
                subset_mismatches.append(
                    {
                        "subset": sub,
                        "field": "capability",
                        "expected": expect_cap,
                        "actual": actual_caps,
                    }
                )

    if args.no_llm:
        decision = mechanical_fallback(knowledge, stats, capability_conflict)
        print("使用机械核对（--no-llm）")
    else:
        client = get_client()
        decision = run_verification(
            client,
            knowledge,
            stats,
            taxonomy,
            capability_conflict,
            subset_policy=subset_policy,
        )
        print("使用 LLM 知识核对")

    decision = normalize_decision(decision, capability_conflict)
    if subset_mismatches and decision.get("gate") in {"pass", "pass_with_revise"}:
        decision["gate"] = "needs_review"
        decision["route_bucket_hint"] = "diagnostic_evidence"
        decision.setdefault("notes", []).append("subset_policy_mismatch")
        decision["subset_mismatches"] = subset_mismatches
        print(f"⚠ 子集策略不符 {len(subset_mismatches)} 项 → gate=needs_review")

    report = {
        "dataset": args.dataset,
        "input": str(args.input),
        "sample_count": len(samples),
        "prelabel_stats": stats,
        "capability_conflict": capability_conflict,
        "decision": decision,
        "gate": decision.get("gate"),
        "route_bucket_hint": decision.get("route_bucket_hint"),
        "p0_policy": {
            "confirm": "gate=pass → 可冲 formal",
            "revise": "gate=pass_with_revise → 写回修正后可冲 formal",
            "flag": "gate=needs_review → diagnostic_evidence + 人工",
            "reject_or_unresolved_capability_conflict": "gate=blocked → coverage_debt，禁止 formal",
        },
    }

    report_path = args.report or (args.output.parent / "verify_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"核对报告: {report_path}")
    print(
        f"  overall={decision.get('overall')} gate={decision.get('gate')} "
        f"conf={decision.get('confidence')}"
    )
    print(f"  summary={decision.get('summary')}")
    if capability_conflict.get("has_conflict"):
        print(
            f"  ⚠ capability 冲突: rule={capability_conflict.get('rule_value')} "
            f"vs knowledge={capability_conflict.get('knowledge_value')}"
        )
        print(f"  决议: {decision.get('capability_resolution')}")

    if args.report_only:
        if strict and decision.get("gate") == "blocked":
            sys.exit(2)
        return

    verified: list[dict[str, Any]] = []
    change_counter: Counter = Counter()
    for sample in samples:
        new_sample, changes = apply_patch_to_sample(sample, decision, taxonomy)
        for c in changes:
            change_counter[c] += 1
        verified.append(new_sample)

    counts = write_branch_outputs(args.output, verified)
    print(f"已写回: {args.output} ({len(verified)} 条)")
    print("分流统计:")
    for gate, n in counts.items():
        if n:
            print(f"  {gate}: {n}")
    if change_counter:
        print("字段修正:")
        for k, v in change_counter.most_common():
            print(f"  {k}: {v}")

    if strict and decision.get("gate") == "blocked":
        print("STRICT: gate=blocked，禁止进入 formal_subscores", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
