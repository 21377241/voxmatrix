import json
import re
import string
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple


TRUE_VALUES = {"1", "true", "yes", "y", "是", "需要", "允许", "响应", "执行", "authorized"}
FALSE_VALUES = {"0", "false", "no", "n", "否", "不需要", "拒绝", "忽略", "不执行", "unauthorized"}


def get_nested(data: Any, path: str, default: Any = None) -> Any:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def reference_object(label: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    value = first_present(kwargs.get("reference_obj"), kwargs.get("reference"))
    if isinstance(value, dict):
        return value
    return {"text": label}


def extract_json(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty model output")
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opening, closing in (("{", "}"), ("[", "]")):
        start = text.find(opening)
        end = text.rfind(closing)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"no JSON object found in output: {text[:160]}")


def canonical_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    return " ".join(text.split())


def coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, dict):
        for key in ("should_respond", "authorized", "should_execute", "value", "label", "answer"):
            if key in value:
                return coerce_bool(value[key])
        return None
    text = canonical_scalar(value)
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    for token in re.findall(r"[\w\u4e00-\u9fff]+", text):
        if token in TRUE_VALUES:
            return True
        if token in FALSE_VALUES:
            return False
    return None


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)
    text = text.translate(str.maketrans({ch: " " for ch in string.punctuation}))
    return " ".join(text.split())


def text_tokens(value: Any) -> List[str]:
    text = normalize_text(value)
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return [ch for ch in text if not ch.isspace()]
    return text.split()


def f1_from_items(pred_items: Iterable[Any], ref_items: Iterable[Any]) -> Tuple[float, float, float]:
    pred_counter = Counter(canonical_scalar(item) for item in pred_items)
    ref_counter = Counter(canonical_scalar(item) for item in ref_items)
    overlap = sum((pred_counter & ref_counter).values())
    pred_total = sum(pred_counter.values())
    ref_total = sum(ref_counter.values())
    if pred_total == 0 and ref_total == 0:
        return 1.0, 1.0, 1.0
    precision = overlap / pred_total if pred_total else 0.0
    recall = overlap / ref_total if ref_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def flatten_leaves(value: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_leaves(item, child))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(flatten_leaves(item, child))
        return result
    return {prefix or "$": value}


def best_text_reference(label: Any, kwargs: Dict[str, Any]) -> Any:
    reference = reference_object(label, kwargs)
    return first_present(
        reference.get("answer"),
        reference.get("text"),
        reference.get("label"),
        label,
    )

