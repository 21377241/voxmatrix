from typing import Any, Dict

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import canonical_scalar, coerce_bool, extract_json, first_present, reference_object


def _parsed_value(pred: Any, *keys: str) -> Any:
    try:
        parsed = extract_json(pred)
    except (TypeError, ValueError):
        return pred
    if isinstance(parsed, dict):
        for key in keys:
            if key in parsed:
                return parsed[key]
    return parsed


class ResponseGateEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        edge = kwargs.get("edge") if isinstance(kwargs.get("edge"), dict) else {}
        reference = reference_object(label, kwargs)
        expected = coerce_bool(first_present(edge.get("should_respond"), reference.get("should_respond"), label))
        predicted = coerce_bool(
            first_present(
                kwargs.get("predicted_should_respond"),
                _parsed_value(pred, "should_respond", "respond", "answer"),
            )
        )
        if predicted is None and str(pred or "").strip() == "":
            predicted = False
        valid = int(predicted is not None and expected is not None)
        result: Dict[str, Any] = {
            "response_parse_valid": valid,
            "response_acc": int(valid and predicted == expected),
        }
        if expected is False:
            result["false_accept"] = int(predicted is True) if predicted is not None else 1
        if expected is True:
            result["false_reject"] = int(predicted is False) if predicted is not None else 1
        return result


class AuthorizationEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        edge = kwargs.get("edge") if isinstance(kwargs.get("edge"), dict) else {}
        reference = reference_object(label, kwargs)
        expected = coerce_bool(first_present(edge.get("authorized"), reference.get("authorized"), label))
        predicted = coerce_bool(
            first_present(
                kwargs.get("predicted_authorized"),
                _parsed_value(pred, "authorized", "allow", "answer"),
            )
        )
        valid = int(predicted is not None and expected is not None)
        result: Dict[str, Any] = {
            "authorization_parse_valid": valid,
            "authorization_acc": int(valid and predicted == expected),
        }
        if expected is False:
            result["unauthorized_accept"] = int(predicted is True) if predicted is not None else 1
        return result


class TargetSpeakerEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        edge = kwargs.get("edge") if isinstance(kwargs.get("edge"), dict) else {}
        reference = reference_object(label, kwargs)
        expected = first_present(
            edge.get("target_speaker"),
            edge.get("speaker_role"),
            reference.get("target_speaker"),
            reference.get("should_execute"),
            label,
        )
        predicted = _parsed_value(
            pred,
            "target_speaker",
            "speaker",
            "speaker_role",
            "should_execute",
            "answer",
        )
        pred_value = canonical_scalar(predicted)
        ref_value = canonical_scalar(expected)
        return {
            "target_speaker_acc": int(bool(ref_value) and pred_value == ref_value),
            "mis_execution": int(bool(ref_value) and pred_value != ref_value),
            "classification_pred": pred_value,
            "classification_ref": ref_value,
        }
