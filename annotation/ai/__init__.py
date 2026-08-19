"""AI 辅助标注：知识收集 + LLM 预标。"""

from annotation.ai.client import AnnotationClient, ChatAnywhereClient, get_client

__all__ = ["AnnotationClient", "ChatAnywhereClient", "get_client"]
