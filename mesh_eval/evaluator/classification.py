import re
from typing import Any, Dict, List

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import (
    canonical_scalar,
    extract_json,
    f1_from_items,
    first_present,
    reference_object,
)


def _prediction_value(pred: Any) -> Any:
    try:
        parsed = extract_json(pred)
    except (TypeError, ValueError):
        return pred
    if isinstance(parsed, dict):
        return first_present(
            parsed.get("label"),
            parsed.get("answer"),
            parsed.get("class"),
            parsed.get("scene_label"),
            parsed.get("value"),
            pred,
        )
    return parsed


class ClassificationEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = first_present(
            reference.get("label"),
            reference.get("answer"),
            reference.get("scene_label"),
            reference.get("count"),
            label,
        )
        pred_value = canonical_scalar(_prediction_value(pred))
        ref_value = canonical_scalar(expected)
        if ref_value.upper() in {"A", "B", "C", "D"}:
            matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", str(pred).upper())
            if matches:
                pred_value = matches[0].lower()
        return {
            "accuracy": int(bool(ref_value) and pred_value == ref_value),
            "classification_pred": pred_value,
            "classification_ref": ref_value,
        }


def _as_labels(value: Any) -> List[Any]:
    if isinstance(value, dict):
        return [key for key, enabled in value.items() if bool(enabled)]
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


class MultiLabelF1Evaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = first_present(
            reference.get("events"),
            reference.get("attributes"),
            reference.get("labels"),
            label,
        )
        parsed = pred
        try:
            parsed_json = extract_json(pred)
            if isinstance(parsed_json, dict):
                parsed = first_present(
                    parsed_json.get("events"),
                    parsed_json.get("attributes"),
                    parsed_json.get("labels"),
                    parsed_json,
                )
            else:
                parsed = parsed_json
        except (TypeError, ValueError):
            pass
        precision, recall, f1 = f1_from_items(_as_labels(parsed), _as_labels(expected))
        return {"label_precision": precision, "label_recall": recall, "f1": f1}


class EntityF1Evaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = first_present(reference.get("entities"), reference.get("slots"), label)
        parsed = pred
        try:
            parsed_json = extract_json(pred)
            if isinstance(parsed_json, dict):
                parsed = first_present(parsed_json.get("entities"), parsed_json.get("slots"), parsed_json)
            else:
                parsed = parsed_json
        except (TypeError, ValueError):
            pass
        precision, recall, f1 = f1_from_items(_as_labels(parsed), _as_labels(expected))
        return {"entity_precision": precision, "entity_recall": recall, "entity_f1": f1}
