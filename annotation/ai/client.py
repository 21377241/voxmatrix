"""OpenAI-compatible client for optional annotation enrichment."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

ANNOTATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-4o-mini"


def load_dotenv(path: Path | None = None) -> None:
    """轻量加载 .env，不依赖 python-dotenv。已存在的环境变量不覆盖。"""
    candidates = []
    if path:
        candidates.append(path)
    candidates.extend(
        [
            ANNOTATION_ROOT / ".env",
            ANNOTATION_ROOT.parent / ".env",
            Path.cwd() / ".env",
        ]
    )
    for env_path in candidates:
        if not env_path.is_file():
            continue
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = value
        break


def extract_json(text: str) -> Any:
    """从模型输出中提取 JSON 对象/数组。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"无法从模型输出解析 JSON: {text[:300]}")


class AnnotationClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        load_dotenv()
        self.api_key = (
            api_key
            or os.environ.get("VOXMATRIX_ANNOTATION_API_KEY")
            or os.environ.get("CHATANYWHERE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (
            base_url
            or os.environ.get("VOXMATRIX_ANNOTATION_BASE_URL")
            or os.environ.get("CHATANYWHERE_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or ""
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("VOXMATRIX_ANNOTATION_MODEL")
            or os.environ.get("CHATANYWHERE_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or DEFAULT_MODEL
        )
        if not self.api_key:
            raise RuntimeError(
                "未配置 API Key。请设置 VOXMATRIX_ANNOTATION_API_KEY，"
                "或在 annotation/.env 中写入（参考 .env.example）"
            )

        from openai import OpenAI

        client_options = {"api_key": self.api_key}
        if self.base_url:
            client_options["base_url"] = self.base_url
        self._client = OpenAI(**client_options)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> str:
        completion = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content
        return content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        model: str | None = None,
    ) -> Any:
        content = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            model=model,
        )
        return extract_json(content)


ChatAnywhereClient = AnnotationClient


def get_client(**kwargs: Any) -> AnnotationClient:
    return AnnotationClient(**kwargs)
