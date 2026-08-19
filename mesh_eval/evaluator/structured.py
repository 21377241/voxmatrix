import re
from typing import Any, Dict, Iterable, List

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import (
    canonical_scalar,
    extract_json,
    f1_from_items,
    first_present,
    flatten_leaves,
    reference_object,
)


def _expected_structure(label: Any, kwargs: Dict[str, Any]) -> Any:
    reference = reference_object(label, kwargs)
    return first_present(
        reference.get("expected_output"),
        reference.get("json"),
        reference.get("slots"),
        kwargs.get("edge", {}).get("expected_action") if isinstance(kwargs.get("edge"), dict) else None,
        label,
    )


def _pair_items(value: Any) -> List[str]:
    return sorted(
        f"{key}={canonical_scalar(item)}"
        for key, item in flatten_leaves(value).items()
    )


class StructuredJsonEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        expected = _expected_structure(label, kwargs)
        try:
            parsed = extract_json(pred)
        except (TypeError, ValueError):
            return {
                "json_valid": 0,
                "structured_exact_match": 0,
                "parameter_precision": 0.0,
                "parameter_recall": 0.0,
                "parameter_f1": 0.0,
            }
        if isinstance(expected, str):
            try:
                expected = extract_json(expected)
            except (TypeError, ValueError):
                expected = {"value": expected}
        precision, recall, f1 = f1_from_items(_pair_items(parsed), _pair_items(expected))
        exact = int(_pair_items(parsed) == _pair_items(expected))
        return {
            "json_valid": 1,
            "structured_exact_match": exact,
            "parameter_precision": precision,
            "parameter_recall": recall,
            "parameter_f1": f1,
        }


def _tool_name(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    direct = first_present(
        value.get("tool"),
        value.get("name"),
        value.get("action"),
        value.get("type"),
        value.get("function"),
        value.get("intent"),
    )
    if isinstance(direct, dict):
        direct = first_present(direct.get("name"), direct.get("tool"))
    return canonical_scalar(direct)


def _tool_args(value: Any) -> Any:
    if not isinstance(value, dict):
        return {}
    function = value.get("function") if isinstance(value.get("function"), dict) else {}
    explicit = first_present(
        value.get("arguments"),
        value.get("parameters"),
        value.get("slots"),
        function.get("arguments"),
    )
    if explicit is not None:
        return explicit
    return {
        key: item
        for key, item in value.items()
        if key not in {"tool", "name", "action", "type", "function", "intent"}
    }


def _tool_calls(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def _parse_tool_prediction(value: Any) -> Any:
    if isinstance(value, str):
        tagged = re.findall(
            r"<tool_call>\s*([\s\S]*?)\s*</tool_call>",
            value,
            flags=re.IGNORECASE,
        )
        if tagged:
            calls = [extract_json(item) for item in tagged]
            return calls[0] if len(calls) == 1 else calls
    return extract_json(value)


class ToolCallEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        expected = _expected_structure(label, kwargs)
        try:
            parsed = _parse_tool_prediction(pred)
            if isinstance(expected, str):
                expected = extract_json(expected)
        except (TypeError, ValueError):
            return {
                "json_valid": 0,
                "tool_acc": 0,
                "parameter_acc": 0.0,
                "parameter_f1": 0.0,
                "task_success": 0,
            }
        parsed_calls = _tool_calls(parsed)
        expected_calls = _tool_calls(expected)
        pred_names = [_tool_name(item) for item in parsed_calls]
        ref_names = [_tool_name(item) for item in expected_calls]
        tool_acc = int(bool(ref_names) and pred_names == ref_names)
        pred_args = _pair_items(
            [_tool_args(item) for item in parsed_calls]
        )
        ref_args = _pair_items(
            [_tool_args(item) for item in expected_calls]
        )
        _, _, parameter_f1 = f1_from_items(pred_args, ref_args)
        parameter_acc = int(pred_args == ref_args)
        return {
            "json_valid": 1,
            "tool_acc": tool_acc,
            "parameter_acc": parameter_acc,
            "parameter_f1": parameter_f1,
            "task_success": int(tool_acc and parameter_acc),
        }


class SlotF1Evaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        expected = _expected_structure(label, kwargs)
        try:
            parsed = extract_json(pred)
        except (TypeError, ValueError):
            parsed = {}
        if isinstance(parsed, dict):
            parsed = first_present(
                parsed.get("slots"),
                parsed.get("arguments"),
                parsed.get("parameters"),
                parsed,
            )
        if isinstance(expected, str):
            try:
                expected = extract_json(expected)
            except (TypeError, ValueError):
                expected = {"value": expected}
        precision, recall, f1 = f1_from_items(_pair_items(parsed), _pair_items(expected))
        return {"slot_precision": precision, "slot_recall": recall, "slot_f1": f1}


class IntentEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = first_present(reference.get("intent"), reference.get("label"), label)
        parsed = pred
        try:
            parsed_json = extract_json(pred)
            if isinstance(parsed_json, dict):
                parsed = first_present(parsed_json.get("intent"), parsed_json.get("label"), pred)
        except (TypeError, ValueError):
            pass
        pred_value = canonical_scalar(parsed)
        ref_value = canonical_scalar(expected)
        return {
            "intent_acc": int(pred_value == ref_value),
            "classification_pred": pred_value,
            "classification_ref": ref_value,
        }


class ActionMatchEvaluator(ToolCallEvaluator):
    """Rule-based action matching for edge.expected_action."""

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        result = super()._eval(pred, label, **kwargs)
        result["action_match"] = result["task_success"]
        return result
