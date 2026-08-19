"""AI 预标注：基于知识包对样本补全 scenario/condition/reference。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from annotation.ai.client import get_client
from annotation.ai.collect_knowledge import CACHE_DIR, collect
from annotation.pipeline.common import ANNOTATION_ROOT, load_jsonl, load_taxonomy, write_jsonl

PROMPTS_DIR = ANNOTATION_ROOT / "prompts"


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / name
    with open(path, encoding="utf-8") as f:
        lines = []
        for line in f:
            if line.startswith("#") and ("变量" in line or "Prompt" in line or "prompt" in line):
                continue
            lines.append(line)
        return "".join(lines).strip()


def fill_prompt(template: str, mapping: dict[str, str]) -> str:
    """只替换显式占位符，避免 JSON 花括号与 str.format 冲突。"""
    out = template
    for key, value in mapping.items():
        out = out.replace("{" + key + "}", value)
    return out


def load_knowledge(dataset_id: str, *, refresh: bool = False) -> dict[str, Any]:
    cache_path = CACHE_DIR / f"{dataset_id}.json"
    if refresh or not cache_path.exists():
        collect(dataset_id, use_llm=True, force=refresh)
    with open(cache_path, encoding="utf-8") as f:
        return json.load(f)


def knowledge_for_prompt(knowledge: dict[str, Any]) -> str:
    profile = knowledge.get("profile") or {}
    compact = {
        "dataset_id": knowledge.get("dataset_id"),
        "seed_facts": knowledge.get("seed_facts"),
        "profile": {
            "summary": profile.get("summary"),
            "suggested_mapping": profile.get("suggested_mapping"),
            "labeling_guidelines": profile.get("labeling_guidelines"),
            "missing_labels": profile.get("missing_labels"),
        },
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


def taxonomy_snippet(taxonomy: dict[str, Any]) -> str:
    return json.dumps(
        {
            "scenarios": taxonomy.get("scenarios"),
            "conditions": taxonomy.get("conditions"),
        },
        ensure_ascii=False,
        indent=2,
    )


def capabilities_snippet(taxonomy: dict[str, Any]) -> str:
    return json.dumps(taxonomy.get("capabilities", {}), ensure_ascii=False, indent=2)


def apply_ai_fields(sample: dict[str, Any], result: dict[str, Any], fields: str) -> dict[str, Any]:
    out = deepcopy(sample)
    out.setdefault("label_meta", {"sources": {}, "confidence": 1.0, "reviewed": False})
    sources = out["label_meta"].setdefault("sources", {})

    confidence = float(result.get("confidence", 0.5))
    old_conf = float(out["label_meta"].get("confidence", 1.0))
    out["label_meta"]["confidence"] = min(old_conf, confidence)
    out["label_meta"]["reviewed"] = False
    rationale_key = "ai_rationale" if fields != "capability" else "ai_capability_rationale"
    out["label_meta"][rationale_key] = result.get("rationale", "")
    if result.get("alternatives"):
        out["label_meta"]["ai_capability_alternatives"] = result["alternatives"]

    if fields in ("capability", "all") and result.get("capability"):
        if result.get("task"):
            out["task"] = result["task"]
            sources["task"] = "llm_draft"
        out["capability"] = result["capability"]
        sources["capability"] = "llm_draft"

    if fields in ("condition", "all") and "condition" in result:
        out.setdefault("condition", {}).update(result["condition"] or {})
        for k in (result.get("condition") or {}):
            sources[f"condition.{k}"] = "llm_draft"
        if "scenario" in result and result["scenario"]:
            out["scenario"] = result["scenario"]
            sources["scenario"] = "llm_draft"

    if fields in ("reference", "all") and "reference" in result:
        out.setdefault("reference", {}).update(result["reference"] or {})
        for k in (result.get("reference") or {}):
            sources[k] = "llm_draft"

    uncertain = result.get("uncertain_fields") or []
    if result.get("needs_human_review") or uncertain or confidence < 0.7:
        out["label_meta"]["needs_human_review"] = True
        if out.get("use_bucket") == "formal_subscores":
            out["use_bucket"] = "diagnostic_evidence"

    return out


def sync_metrics_for_capability(sample: dict[str, Any], taxonomy: dict[str, Any]) -> None:
    cap = sample.get("capability")
    metrics = taxonomy.get("default_metrics", {}).get(cap)
    if metrics:
        sample["metrics"] = list(metrics)


def _sample_brief(sample: dict[str, Any]) -> str:
    return json.dumps(
        {
            "sample_id": sample.get("sample_id"),
            "dataset": sample.get("dataset"),
            "task": sample.get("task"),
            "capability": sample.get("capability"),
            "scenario": sample.get("scenario"),
            "condition": sample.get("condition"),
            "reference": sample.get("reference"),
            "split": sample.get("split"),
            "input": {"audio_path": sample.get("input", {}).get("audio_path")},
            "raw_subset": sample.get("label_meta", {}).get("raw_subset"),
        },
        ensure_ascii=False,
        indent=2,
    )


def label_capability(
    client: Any,
    sample: dict[str, Any],
    knowledge: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    prompt = load_prompt("capability.txt")
    content = fill_prompt(
        prompt,
        {
            "capabilities_json": capabilities_snippet(taxonomy),
            "knowledge_json": knowledge_for_prompt(knowledge),
            "sample_json": _sample_brief(sample),
        },
    )
    return client.chat_json([{"role": "user", "content": content}])


def label_condition(
    client: Any,
    sample: dict[str, Any],
    knowledge: dict[str, Any],
    taxonomy: dict[str, Any],
) -> dict[str, Any]:
    prompt = load_prompt("scenario_condition.txt")
    content = fill_prompt(
        prompt,
        {
            "taxonomy_json": taxonomy_snippet(taxonomy),
            "knowledge_json": knowledge_for_prompt(knowledge),
            "sample_json": _sample_brief(sample),
        },
    )
    return client.chat_json([{"role": "user", "content": content}])


def label_reference(
    client: Any,
    sample: dict[str, Any],
    knowledge: dict[str, Any],
    taxonomy: dict[str, Any],
    tool_schema: str,
) -> dict[str, Any]:
    capability = sample.get("capability", "")
    required = taxonomy.get("required_reference", {}).get(capability, [])
    prompt_name = "tool_call.txt" if capability in {"tool_call", "navigation"} else "reference_infer.txt"
    prompt = load_prompt(prompt_name)

    if prompt_name == "tool_call.txt":
        transcript = (
            sample.get("reference", {}).get("transcript")
            or sample.get("reference", {}).get("text")
            or ""
        )
        content = fill_prompt(
            prompt,
            {
                "transcript": transcript,
                "available_tools": tool_schema or "[]",
                "tool_schema": tool_schema or "{}",
            },
        )
    else:
        content = fill_prompt(
            prompt,
            {
                "capability": capability,
                "required_fields": json.dumps(required, ensure_ascii=False),
                "knowledge_json": knowledge_for_prompt(knowledge),
                "sample_json": json.dumps(sample, ensure_ascii=False, indent=2)[:8000],
                "tool_schema": tool_schema or "{}",
            },
        )
    return client.chat_json([{"role": "user", "content": content}])


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible AI pre-labeling")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", required=True, type=Path, help="预标注 jsonl")
    parser.add_argument("--output", required=True, type=Path, help="AI 标注后 jsonl")
    parser.add_argument(
        "--fields",
        choices=["capability", "condition", "reference", "all"],
        default="capability",
        help="补标字段范围（默认 capability，为最关键标签）",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 条，0 表示全部")
    parser.add_argument("--refresh-knowledge", action="store_true")
    parser.add_argument("--tool-schema", type=Path, default=None, help="本地 tool schema JSON")
    parser.add_argument("--dry-run", action="store_true", help="只跑知识包，不标样本")
    args = parser.parse_args()

    knowledge = load_knowledge(args.dataset, refresh=args.refresh_knowledge)
    if args.dry_run:
        print(json.dumps(knowledge.get("profile"), ensure_ascii=False, indent=2))
        return

    taxonomy = load_taxonomy()
    client = get_client()
    samples = load_jsonl(args.input)
    if args.limit > 0:
        samples = samples[: args.limit]

    tool_schema = "{}"
    if args.tool_schema and args.tool_schema.exists():
        tool_schema = args.tool_schema.read_text(encoding="utf-8")

    labeled: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        merged = deepcopy(sample)
        try:
            # capability 优先：决定协议与指标
            if args.fields in ("capability", "all"):
                cap_result = label_capability(client, merged, knowledge, taxonomy)
                merged = apply_ai_fields(merged, cap_result, "capability")
                sync_metrics_for_capability(merged, taxonomy)
            if args.fields in ("condition", "all"):
                cond_result = label_condition(client, merged, knowledge, taxonomy)
                merged = apply_ai_fields(merged, cond_result, "condition")
            if args.fields in ("reference", "all"):
                ref_result = label_reference(
                    client, merged, knowledge, taxonomy, tool_schema
                )
                merged = apply_ai_fields(merged, ref_result, "reference")
            print(
                f"[{i + 1}/{len(samples)}] ✓ {merged.get('sample_id')} "
                f"capability={merged.get('capability')} "
                f"conf={merged.get('label_meta', {}).get('confidence')}"
            )
        except Exception as e:  # noqa: BLE001 - 单条失败不中断整批
            merged.setdefault("label_meta", {})["ai_error"] = str(e)
            merged["label_meta"]["needs_human_review"] = True
            print(f"[{i + 1}/{len(samples)}] ✗ {merged.get('sample_id')}: {e}")
        labeled.append(merged)

    write_jsonl(args.output, labeled)
    print(f"AI 标注完成: {len(labeled)} 条 → {args.output}")
    print("注意: llm_draft 默认需人工 reviewed=true 后才可进入 formal_subscores")


if __name__ == "__main__":
    main()
