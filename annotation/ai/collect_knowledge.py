"""从论文/官网收集数据集信息，生成 AI 标注上下文知识包。"""

from __future__ import annotations

import argparse
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from annotation.ai.client import get_client
from annotation.pipeline.common import ANNOTATION_ROOT, deep_merge, load_mapping

KNOWLEDGE_DIR = ANNOTATION_ROOT / "knowledge" / "datasets"
CACHE_DIR = ANNOTATION_ROOT / "knowledge" / "cache"
USER_AGENT = "VocalBench-Annotator/0.1 (+research; contact=local)"


def load_source_card(dataset_id: str) -> dict[str, Any]:
    path = KNOWLEDGE_DIR / f"{dataset_id}.yaml"
    mapping: dict[str, Any] = {}
    try:
        mapping = load_mapping(dataset_id)
    except FileNotFoundError:
        pass
    if not path.exists():
        knowledge_dataset = mapping.get("knowledge_dataset")
        if knowledge_dataset:
            path = KNOWLEDGE_DIR / f"{knowledge_dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到知识源配置: {path}\n"
            f"请先在 annotation/knowledge/datasets/ 下创建 {dataset_id}.yaml"
        )
    with open(path, encoding="utf-8") as f:
        card = yaml.safe_load(f) or {}
    if mapping.get("knowledge_overrides"):
        card = deep_merge(card, mapping["knowledge_overrides"])
    card["dataset_id"] = dataset_id
    return card


def html_to_text(content: str) -> str:
    content = re.sub(r"(?is)<script[^>]*>.*?</script>", " ", content)
    content = re.sub(r"(?is)<style[^>]*>.*?</style>", " ", content)
    content = re.sub(r"(?is)<[^>]+>", " ", content)
    content = html.unescape(content)
    content = re.sub(r"\s+", " ", content).strip()
    return content


def fetch_url(url: str, timeout: int = 30) -> str:
    """抓取网页/摘要文本。arxiv abs 页可用；PDF 仅尝试文本抽取失败则跳过。"""
    if url.endswith(".pdf"):
        # 优先用 html abs 页，避免 PDF 二进制
        abs_url = url.replace("/pdf/", "/abs/").replace(".pdf", "")
        if abs_url != url:
            url = abs_url

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    try:
        text = raw.decode(charset, errors="replace")
    except LookupError:
        text = raw.decode("utf-8", errors="replace")

    if "<html" in text.lower() or "<!doctype" in text.lower():
        return html_to_text(text)
    return text


def truncate(text: str, max_chars: int = 12000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


PROFILE_PROMPT = """你是语音数据集调研助手。根据提供的论文/官网摘录与种子事实，整理一份
端侧语音 Benchmark 标注所需的「数据集知识包」。

必须输出 JSON，字段：
{{
  "dataset_id": "...",
  "summary": "2-4 句中文概括",
  "language": "...",
  "domain": "...",
  "recording_setup": "采集方式/麦克风/远近场等",
  "annotation_coverage": "原生标注有哪些",
  "suggested_mapping": {{
    "task": "...",
    "capability": "... 或列表",
    "scenario": "phone|home|vehicle|meeting|outdoor",
    "condition": {{
      "acoustic": "...",
      "spatial": "...",
      "speaker": "...",
      "device": "...",
      "interaction": "..."
    }},
    "metrics": ["..."],
    "use_bucket": "formal_subscores|diagnostic_evidence|stress_test|coverage_debt"
  }},
  "missing_labels": ["阻碍 formal_subscores 的缺口"],
  "labeling_guidelines": ["给标注员/AI 补标的规则，3-8 条"],
  "evidence": [{{"source": "url", "claim": "支撑结论"}}]
}}

约束：
1. scenario/condition 枚举必须符合端侧方案，不可编造电话/车载等无据标签
2. 证据不足时 use_bucket 宁可用 diagnostic_evidence 或 coverage_debt
3. 不要输出 JSON 以外内容

数据集 ID: {dataset_id}
种子事实:
{seed_facts}

来源摘录:
{excerpts}
"""


def collect(
    dataset_id: str,
    *,
    use_llm: bool = True,
    force: bool = False,
    fetch_sources: bool = True,
) -> Path:
    card = load_source_card(dataset_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CACHE_DIR / f"{dataset_id}.json"
    if out_path.exists() and not force:
        print(f"缓存已存在（使用 --force 刷新）: {out_path}")
        return out_path

    excerpts: list[dict[str, str]] = []
    for src in card.get("sources", []) if fetch_sources else []:
        url = src.get("url", "")
        title = src.get("title", url)
        if not url:
            continue
        try:
            text = truncate(fetch_url(url))
            excerpts.append({"title": title, "url": url, "text": text, "error": ""})
            print(f"✓ 抓取: {title}")
        except (urllib.error.URLError, TimeoutError, ValueError) as e:
            excerpts.append({"title": title, "url": url, "text": "", "error": str(e)})
            print(f"✗ 抓取失败: {title} ({e})")

    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "sources": card.get("sources", []),
        "seed_facts": card.get("seed_facts", {}),
        "excerpts": excerpts,
        "profile": None,
    }

    if use_llm:
        client = get_client()
        usable = [e for e in excerpts if e.get("text")]
        excerpt_blob = "\n\n".join(
            f"### {e['title']}\nURL: {e['url']}\n{e['text'][:4000]}" for e in usable
        ) or "（网页抓取失败，请仅依据种子事实推理，并明确标注证据不足）"
        messages = [
            {
                "role": "user",
                "content": PROFILE_PROMPT.format(
                    dataset_id=dataset_id,
                    seed_facts=json.dumps(card.get("seed_facts", {}), ensure_ascii=False, indent=2),
                    excerpts=excerpt_blob,
                ),
            }
        ]
        profile = client.chat_json(messages)
        payload["profile"] = profile
        print("✓ LLM 知识包生成完成")
    else:
        payload["profile"] = {
            "dataset_id": dataset_id,
            "summary": "仅本地种子事实，未调用 LLM",
            "suggested_mapping": card.get("seed_facts", {}).get("typical_labels", {}),
            "labeling_guidelines": [],
            "missing_labels": [],
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"知识包已写入: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="收集论文/官网信息并生成数据集知识包")
    parser.add_argument("--dataset", required=True, help="数据集 ID，如 aishell_1")
    parser.add_argument("--no-llm", action="store_true", help="仅抓取，不调用 LLM 归纳")
    parser.add_argument("--no-fetch", action="store_true", help="不访问网络，仅使用本地种子事实")
    parser.add_argument("--force", action="store_true", help="强制刷新缓存")
    args = parser.parse_args()
    collect(
        args.dataset,
        use_llm=not args.no_llm,
        force=args.force,
        fetch_sources=not args.no_fetch,
    )


if __name__ == "__main__":
    main()
