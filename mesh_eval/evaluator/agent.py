"""Evaluators for the spoken-agent (T4) capability family.

The evaluators in this module deliberately separate *decision* metrics from
lexical similarity.  A spoken agent may answer correctly with wording that is
different from a reference, and a tool call may be syntactically valid while
still targeting the wrong tool.  They also fail closed when a benchmark's
official rubric requires an external judge that is not configured.
"""

from __future__ import annotations

import json
import inspect
import math
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.structured import (
    _parse_tool_prediction,
    _tool_args,
    _tool_calls,
    _tool_name,
)
from mesh_eval.evaluator.utils import (
    canonical_scalar,
    extract_json,
    first_present,
    f1_from_items,
    reference_object,
)


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if value is None:
        return None
    text = canonical_scalar(value)
    if text in {"true", "1", "yes", "y", "ask", "clarify", "需要", "是"}:
        return True
    if text in {"false", "0", "no", "n", "answer", "不需要", "否"}:
        return False
    return None


def _parse_object(pred: Any) -> Optional[Dict[str, Any]]:
    try:
        value = extract_json(pred)
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _json_native(value: Any) -> Any:
    """Normalize Arrow/NumPy values used by benchmark rubric implementations."""
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
    return value


class ClarificationEvaluator(Evaluator):
    """Score an explicit ``should_clarify`` decision.

    Natural-language question-mark heuristics are intentionally not accepted.
    The model must return JSON such as ``{"should_clarify": true,
    "response": "Which booking?"}``; this prevents a fluent answer from
    being silently interpreted as a decision.
    """

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = label if isinstance(label, Mapping) else reference_object(label, kwargs)
        answerable = _as_bool(reference.get("answerable"))
        expected = first_present(
            reference.get("should_clarify"),
            None if answerable is None else not answerable,
            reference.get("clarification_required"),
        )
        expected_bool = _as_bool(expected)
        if expected_bool is None:
            return {
                "status": "not_evaluated",
                "not_evaluated": 1,
                "not_evaluated_reason": "clarification reference decision is missing",
                "clarification_parse_valid": 0,
                "clarification_accuracy": None,
                "clarification_tp": None,
                "clarification_fp": None,
                "clarification_tn": None,
                "clarification_fn": None,
                "clarification_precision": None,
                "clarification_recall": None,
                "clarification_f1": None,
            }
        parsed = _parse_object(pred)
        predicted = _as_bool(parsed.get("should_clarify")) if parsed else None
        valid = int(predicted is not None)
        correct = int(valid and predicted == expected_bool)
        tp = int(valid and expected_bool is True and predicted is True)
        fp = int(valid and expected_bool is False and predicted is True)
        tn = int(valid and expected_bool is False and predicted is False)
        # A malformed/missing explicit decision is an error, not an excuse to
        # disappear from the denominator.  It is a false negative when the
        # reference requires clarification and always receives accuracy=0.
        fn = int(expected_bool is True and predicted is not True)
        # Per-sample sufficient statistics are consumed by MeshAgg.  Do not
        # emit a fabricated zero-valued F1 when parsing failed.
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            "status": "evaluated",
            "not_evaluated": 0,
            "clarification_parse_valid": valid,
            "clarification_accuracy": correct,
            "should_clarify": int(expected_bool),
            "predicted_should_clarify": int(predicted) if predicted is not None else None,
            "clarification_tp": tp,
            "clarification_fp": fp,
            "clarification_tn": tn,
            "clarification_fn": fn,
            "clarification_precision": precision,
            "clarification_recall": recall,
            "clarification_f1": f1,
            "clarification_response": parsed.get("response") if parsed else None,
        }


def _expected_tool_call(reference: Mapping[str, Any]) -> Any:
    return first_present(
        reference.get("expected_output"),
        reference.get("gold_call"),
        reference.get("expected_call"),
    )


def _tool_family(name: Any) -> str:
    """Map StepEval release aliases to a stable tool family."""
    value = canonical_scalar(name)
    return {
        "search": "web_search",
        "web search": "web_search",
        "web_search": "web_search",
        "timbre": "timbre",
        "timbre_rag": "timbre",
    }.get(value, value)


class ToolTriggerEvaluator(Evaluator):
    """Evaluate StepEval-style positive/negative tool triggering.

    Positive examples require any valid call, then separately score the target
    tool type and parameters.  Negative examples only count a call to the
    *target* tool as a false positive; calling an unrelated tool is reported
    but is not incorrectly folded into target-trigger precision.
    """

    def __init__(self, parameter_judge: Optional[Callable[..., Any]] = None):
        self.parameter_judge = parameter_judge

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = label if isinstance(label, Mapping) else {}
        if not isinstance(expected, Mapping):
            expected = {}
        target = str(
            first_present(
                expected.get("target_tool"),
                reference.get("target_tool"),
                "",
            )
            or ""
        )
        polarity = str(
            first_present(expected.get("polarity"), reference.get("polarity"), "positive")
        ).lower()
        try:
            parsed = _parse_tool_prediction(pred)
        except (TypeError, ValueError):
            parsed = None
        calls = _tool_calls(parsed)
        valid_calls = [item for item in calls if _tool_name(item)]
        names = [_tool_name(item) for item in valid_calls]
        invoked = bool(valid_calls)
        target_family = _tool_family(target)
        target_hit = bool(
            target_family
            and any(_tool_family(name) == target_family for name in names)
        )
        positive = polarity not in {"negative", "neg", "no", "false"}
        trigger_correct = int(invoked if positive else not target_hit)
        # Sufficient statistics use the intended binary event: trigger any
        # valid call for positives, target-tool call for negative false alarm.
        tp = int(positive and invoked)
        fp = int((not positive) and target_hit)
        fn = int(positive and not invoked)
        tn = int((not positive) and not target_hit)
        gold = first_present(expected.get("gold_call"), _expected_tool_call(reference))
        gold_calls = _tool_calls(gold)
        gold_name = _tool_name(gold_calls[0]) if gold_calls else target
        type_correct = int(
            bool(gold_name)
            and any(_tool_family(name) == _tool_family(gold_name) for name in names)
        )
        parameter_f1 = None
        parameter_exact = None
        parameter_judge_accuracy = None
        parameter_status = (
            "not_evaluated_no_valid_call" if positive else "not_applicable_negative"
        )
        if gold_calls and valid_calls:
            predicted_call = next(
                (
                    call
                    for call in valid_calls
                    if _tool_family(_tool_name(call)) == _tool_family(gold_name)
                ),
                valid_calls[0],
            )
            pred_args = _tool_args(predicted_call)
            gold_args = _tool_args(gold_calls[0])
            _, _, parameter_f1 = f1_from_items(
                [f"{k}={v}" for k, v in sorted(pred_args.items())],
                [f"{k}={v}" for k, v in sorted(gold_args.items())],
            )
            parameter_exact = int(
                canonical_scalar(pred_args) == canonical_scalar(gold_args)
            )
            if positive and not gold_args:
                # StepEval reports the date/time parameter metric as N/A:
                # that tool's native schema has no parameters.
                parameter_status = "not_applicable_no_parameters"
            elif positive and self.parameter_judge is not None:
                judged = self.parameter_judge(
                    {
                        "prediction": pred_args,
                        "reference": gold_args,
                        "tool": gold_name,
                        "context": (kwargs.get("input") or {}).get("messages"),
                    }
                )
                if isinstance(judged, Mapping):
                    judged = first_present(
                        judged.get("correct"),
                        judged.get("pass"),
                        judged.get("score"),
                    )
                judged_bool = _as_bool(judged)
                if judged_bool is None:
                    raise ValueError("StepEval parameter judge returned no binary decision")
                parameter_judge_accuracy = int(judged_bool)
                parameter_status = "evaluated"
            elif positive:
                parameter_status = "not_evaluated_official_judge_missing"
        return {
            "trigger_parse_valid": int(parsed is not None),
            "trigger_invoked": int(invoked),
            "target_tool": target,
            "target_tool_family": target_family,
            "target_tool_hit": int(target_hit),
            "trigger_correct": trigger_correct,
            "trigger_tp": tp,
            "trigger_fp": fp,
            "trigger_tn": tn,
            "trigger_fn": fn,
            # Native StepEval computes type accuracy only over triggered
            # positive calls. Missed calls are already captured by recall.
            "tool_type_accuracy": type_correct if positive and invoked else None,
            "parameter_judge_accuracy": parameter_judge_accuracy,
            "parameter_exact_match": parameter_exact if positive else None,
            "parameter_f1": parameter_f1,
            "parameter_evaluation_status": parameter_status,
            "polarity": polarity,
        }


def _audio_value(value: Any) -> Optional[str]:
    if isinstance(value, (str, Path)):
        text = str(value)
        if isinstance(value, str) and text.lstrip().startswith("{"):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, Mapping):
                return _audio_value(parsed)
        return text
    if isinstance(value, Mapping):
        for key in ("audio", "audio_path", "uri", "path", "value"):
            if value.get(key):
                return str(value[key])
    return None


def _has_audio_prediction(pred: Any, kwargs: Mapping[str, Any]) -> bool:
    return bool(_prediction_audio_paths(pred, kwargs))


def _prediction_audio_paths(pred: Any, kwargs: Mapping[str, Any]) -> List[str]:
    paths: List[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                collect(json.loads(value))
                return
            except json.JSONDecodeError:
                pass
        if isinstance(value, Mapping):
            for key in (
                "audio",
                "audio_path",
                "generated_audio",
                "output_audio",
                "audio_responses",
            ):
                if key in value:
                    collect(value[key])
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)
            return
        path = _audio_value(value)
        if path:
            try:
                exists = Path(path).is_file()
            except (OSError, ValueError):
                # Ordinary generated text is also a string.  It must not be
                # passed through as an unbounded filesystem path.
                exists = False
            if exists and path not in paths:
                paths.append(path)

    collect(pred)
    collect(kwargs.get("generated_audio"))
    collect(kwargs.get("output_audio"))
    return paths


class AgentRubricEvaluator(Evaluator):
    """Bridge benchmark rubrics to an external judge without unsafe fallbacks.

    ``judge`` may be injected in tests or by a benchmark-specific runner.  In
    production, a model is resolved lazily only when credentials are present.
    Missing credentials raise a clear error for text rubrics; audio-required
    rubrics return ``not_evaluated`` when no generated audio is available.
    """

    def __init__(
        self,
        model_name: str = "gpt4o-mini",
        judge: Optional[Callable[..., Any]] = None,
        require_audio: bool = False,
        audio_judge: bool = False,
        rubric_kind: str = "agent",
    ):
        self.model_name = model_name
        self.judge = judge
        self.require_audio = bool(require_audio)
        self.audio_judge = bool(audio_judge)
        self.rubric_kind = rubric_kind

    def _resolve_judge(self) -> Callable[..., Any]:
        if self.judge is not None:
            return self.judge
        if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("LLMCenterUserToken")):
            raise RuntimeError(
                f"{self.rubric_kind} judge requires OPENAI_API_KEY or "
                "LLMCenterUserToken; refusing to emit a numeric score"
            )
        from audio_evals.registry import registry

        model_name = self.model_name
        if (
            model_name == "gpt4o-mini"
            and not os.environ.get("OPENAI_API_KEY")
            and os.environ.get("LLMCenterUserToken")
        ):
            model_name = "mb-gpt4o-mini"
        model = registry.get_model(model_name)
        if model is None:
            raise RuntimeError(f"judge model is not registered: {model_name}")

        def call(payload: Any, **kwargs: Any) -> Any:
            if isinstance(payload, Mapping):
                if self.rubric_kind == "IHBench":
                    output_contract = (
                        'Return JSON only as {"tf_win": 0|0.5|1, '
                        '"rq_pass": 0|1, "reason": string}. Compare the candidate '
                        "with the baseline for task fulfillment (0=baseline wins, "
                        "0.5=tie, 1=candidate wins), then apply all response-quality rubrics."
                    )
                else:
                    output_contract = (
                        'Return JSON only as {"score": number, "reason": string}, '
                        "where score is between 0 and 1 and measures satisfaction "
                        "of the supplied benchmark rubric."
                    )
                prompt = (
                    "You are a strict speech-agent benchmark judge. "
                    + output_contract
                    + "\nEvaluation payload:\n"
                    + json.dumps(payload, ensure_ascii=False, default=str)
                )
            else:
                prompt = payload
            return model.inference(prompt, temperature=0, max_tokens=2048, **kwargs)

        return call

    @staticmethod
    def _judge_payload(pred: Any, reference: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        rubric = first_present(
            reference.get("rubric"),
            reference.get("rubrics"),
            kwargs.get("rubric"),
        )
        if rubric is None and (
            reference.get("task_fulfillment_rubric")
            or reference.get("response_quality_rubrics")
        ):
            rubric = {
                "task_fulfillment": reference.get("task_fulfillment_rubric"),
                "response_quality": reference.get("response_quality_rubrics"),
            }
        return {
            "prediction": pred,
            "reference": dict(reference),
            "rubric": rubric,
            "prompt": first_present(kwargs.get("question"), kwargs.get("instruction")),
            "input": _json_native(kwargs.get("input") or {}),
        }

    @staticmethod
    def _parse_judge_result(value: Any) -> Dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        try:
            parsed = extract_json(value)
            if isinstance(parsed, Mapping):
                return dict(parsed)
        except (TypeError, ValueError):
            pass
        text = str(value or "")
        match = re.search(r"(?:score|rating)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, re.I)
        if match:
            return {"score": float(match.group(1))}
        raise ValueError("judge response has no structured score")

    @staticmethod
    def _call_judge(judge: Callable[..., Any], payload: Dict[str, Any]) -> Any:
        """Support the payload and common two/three-argument judge APIs."""
        try:
            signature = inspect.signature(judge)
        except (TypeError, ValueError):
            return judge(payload)
        parameters = list(signature.parameters.values())
        positional = [
            item
            for item in parameters
            if item.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(
            item.kind == inspect.Parameter.VAR_POSITIONAL for item in parameters
        )
        has_varkw = any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters
        )
        keyword_only = [
            item for item in parameters if item.kind == inspect.Parameter.KEYWORD_ONLY
        ]
        required_positional = [
            item for item in positional if item.default is inspect.Parameter.empty
        ]
        if has_varkw:
            if len(required_positional) == 1 and required_positional[0].name not in payload:
                return judge(payload)
            return judge(**payload)
        if keyword_only and not positional:
            accepted = {item.name for item in keyword_only}
            return judge(**{key: value for key, value in payload.items() if key in accepted})
        if has_varargs or len(positional) <= 1:
            return judge(payload)
        if len(positional) == 2:
            return judge(payload["prediction"], payload["reference"])
        return judge(
            payload["prediction"],
            payload["reference"],
            payload.get("rubric"),
        )

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = label if isinstance(label, Mapping) else reference_object(label, kwargs)
        requires_audio = self.require_audio or bool(
            reference.get("requires_audio_output")
            or (kwargs.get("metadata") or {}).get("requires_audio_output")
            or kwargs.get("requires_audio_output")
        )
        audio_paths = _prediction_audio_paths(pred, kwargs)
        required_audio_outputs = int(reference.get("required_audio_outputs") or 1)
        if requires_audio and len(audio_paths) < required_audio_outputs:
            return {
                "status": "not_evaluated",
                "not_evaluated": 1,
                "not_evaluated_reason": (
                    "generated audio is required by this rubric "
                    f"({len(audio_paths)}/{required_audio_outputs} outputs present)"
                ),
            }
        if requires_audio and not self.audio_judge:
            return {
                "status": "not_evaluated",
                "not_evaluated": 1,
                "not_evaluated_reason": "an audio-capable rubric judge is not configured",
            }
        judge = self._resolve_judge()
        result = self._parse_judge_result(
            self._call_judge(judge, self._judge_payload(pred, reference, kwargs))
        )
        output: Dict[str, Any] = {"status": "evaluated", "not_evaluated": 0}
        for key in ("score", "tf_win", "tf_score", "rq_pass", "rq_score", "pass"):
            if key in result:
                value = result[key]
                if isinstance(value, bool):
                    value = int(value)
                output[key] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value
        for key in ("score", "rq_score"):
            if key in output and (
                not isinstance(output[key], (int, float))
                or not 0.0 <= float(output[key]) <= 1.0
            ):
                raise ValueError(f"judge {key} must be numeric in [0, 1]")
        for key in ("rq_pass", "pass"):
            if key in output and output[key] not in (0, 0.0, 1, 1.0):
                raise ValueError(f"judge {key} must be binary")
        for key in ("tf_win", "tf_score"):
            if key in output and output[key] not in (0, 0.0, 0.5, 1, 1.0):
                raise ValueError(f"judge {key} must be one of 0, 0.5, 1")
        if result.get("reason") not in (None, ""):
            output["judge_reason"] = str(result["reason"])
        if "score" not in output and not any(k in output for k in ("tf_win", "rq_pass", "rq_score")):
            raise ValueError("judge response did not contain a supported score")
        if "score" in output:
            output["rubric_score"] = output["score"]
            requested_metrics = set(kwargs.get("metrics") or [])
            if "constraint_satisfaction" in requested_metrics:
                output["constraint_satisfaction"] = output["score"]
            if "llm_judge" in requested_metrics:
                output["llm_judge"] = output["score"]
        return output


class IFEvalAdapter(Evaluator):
    """Run VoiceBench IFEval against canonical ``reference_obj`` fields."""

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = label if isinstance(label, Mapping) else reference_object(label, kwargs)
        prompt = str(reference.get("prompt") or label or "")
        instruction_ids = reference.get("instruction_id_list") or reference.get("instruction_ids") or []
        instruction_kwargs = reference.get("kwargs") or []
        if hasattr(instruction_ids, "tolist"):
            instruction_ids = instruction_ids.tolist()
        if hasattr(instruction_kwargs, "tolist"):
            instruction_kwargs = instruction_kwargs.tolist()
        instruction_ids = _json_native(instruction_ids)
        instruction_kwargs = _json_native(instruction_kwargs)
        if not isinstance(instruction_ids, list) or not instruction_ids:
            return {
                "status": "not_evaluated",
                "not_evaluated": 1,
                "not_evaluated_reason": "IFEval instruction ids are missing",
            }
        if not isinstance(instruction_kwargs, list) or len(instruction_kwargs) != len(instruction_ids):
            raise ValueError("IFEval kwargs must align one-to-one with instruction ids")

        from audio_evals.evaluator.ifeval import IFEval

        strict = IFEval(strict=True)(
            pred,
            prompt,
            instruction_id_list=instruction_ids,
            kwargs=instruction_kwargs,
        )
        loose = IFEval(strict=False)(
            pred,
            prompt,
            instruction_id_list=instruction_ids,
            kwargs=instruction_kwargs,
        )
        return {
            "status": "evaluated",
            "not_evaluated": 0,
            "constraint_satisfaction": strict["strict_prompt"],
            "strict_prompt": strict["strict_prompt"],
            "strict_instruction": strict["strict_instruction"],
            "loose_prompt": loose["loose_prompt"],
            "loose_instruction": loose["loose_instruction"],
        }


class IHBenchEvaluator(AgentRubricEvaluator):
    """Official IHBench TF/RQ rubric bridge (not lexical F1)."""

    def __init__(self, model_name: str = "gpt4o-mini", judge: Optional[Callable[..., Any]] = None):
        super().__init__(model_name=model_name, judge=judge, rubric_kind="IHBench")

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        result = super()._eval(pred, label, **kwargs)
        if result.get("status") == "not_evaluated":
            return result
        tf = result.get("tf_win", result.get("tf_score"))
        rq = result.get("rq_pass", result.get("rq_score"))
        if tf is None or rq is None:
            raise ValueError("IHBench judge must return both TF and RQ decisions")
        result["tf_win_score"] = tf
        result["rq_pass"] = rq
        return result
