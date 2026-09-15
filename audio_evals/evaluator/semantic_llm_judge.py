"""Semantic LLM judge evaluator (A1 text+transcript; B transcript+audio).

Paired design: A1 and B share the same judge model and the same
``audio_transcript``; B only adds WavPath. Default multimodal model:
``qwen3-omni-thinking``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from audio_evals.evaluator.base import Evaluator
from audio_evals.evaluator.llm_judge_prompts import build_prompt, prompt_id, resolve_route

DEFAULT_MM_JUDGE_MODEL = "qwen3-omni-thinking"
DEFAULT_PAIRED_JUDGE_MODEL = "qwen3-omni-thinking"


def _as_ref_text(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, list):
        parts = [str(x).strip() for x in label if x is not None and str(x).strip()]
        return " | ".join(parts)
    return str(label).strip()


def parse_score_1_to_5(raw: str) -> Optional[float]:
    text = (raw or "").strip()
    if not text:
        return None
    # Thinking models may emit long chains; prefer a trailing standalone score.
    tail = text[-200:] if len(text) > 200 else text
    for chunk in (tail, text):
        try:
            value = float(chunk.strip())
            if 1.0 <= value <= 5.0:
                return value
        except ValueError:
            pass
        match = re.search(r"\[\[(\d+(?:\.\d+)?)\]\]", chunk)
        if match:
            value = float(match.group(1))
            if 1.0 <= value <= 5.0:
                return value
        lines = [ln.strip() for ln in chunk.splitlines() if ln.strip()]
        for ln in reversed(lines[-8:]):
            if re.fullmatch(r"[1-5](?:\.0+)?", ln):
                return float(ln)
        match = re.search(r"\b([1-5])(?:\.0+)?\b", chunk)
        if match:
            return float(match.group(1))
    return None


def parse_yes_no(raw: str) -> Optional[int]:
    text = (raw or "").strip().lower()
    if not text:
        return None
    matches = list(re.finditer(r"\b(yes|no)\b", text))
    if not matches:
        return None
    return 1 if matches[-1].group(1) == "yes" else 0


def _collect_usage(model: Any) -> Dict[str, Any]:
    usage = getattr(model, "last_usage", None)
    if isinstance(usage, dict):
        return dict(usage)
    return {}


def _is_openai_chat_judge(model: Any) -> bool:
    return model.__class__.__name__ in ("AdvancedGPT", "GPT")


class SemanticLLMJudgeEvaluator(Evaluator):
    """Judge open/semantic tasks with task-specific prompts.

    kwargs (common):
      capability, rubric, judge_template ('text'|'multimodal'),
      question, audio_transcript, direction, eval_mode,
      WavPath / audio, require_transcript,
      judge_model_name, n_samples
    """

    BINARY_RUBRICS = frozenset({"binary"})

    def __init__(
        self,
        judge_model_name: str = DEFAULT_PAIRED_JUDGE_MODEL,
        n_samples: int = 1,
        default_template: str = "text",
        mm_judge_model_name: str = DEFAULT_MM_JUDGE_MODEL,
    ):
        self.judge_model_name = judge_model_name
        self.mm_judge_model_name = mm_judge_model_name
        self.n_samples = max(1, int(n_samples))
        self.default_template = default_template

    def _eval(self, pred, label, **kwargs) -> Dict[str, Any]:
        from audio_evals.evaluator.voice_bench import (
            get_judge_model,
            resolve_judge_model_name,
        )

        pred_s = "" if pred is None else str(pred)
        ref_s = _as_ref_text(label)
        capability = kwargs.get("capability") or kwargs.get("task_capability")
        if not capability:
            return self._skip(pred_s, ref_s, "missing_capability")

        try:
            route = resolve_route(
                str(capability),
                kwargs.get("rubric"),
                eval_mode=kwargs.get("eval_mode"),
                has_reference=bool(ref_s),
            )
        except ValueError as exc:
            return self._skip(pred_s, ref_s, f"route_error:{exc}")

        template = str(
            kwargs.get("judge_template")
            or kwargs.get("template")
            or self.default_template
            or "text"
        ).strip().lower()
        if template in ("mm", "multi", "audio"):
            template = "multimodal"
        pid = prompt_id(route["capability"], route["rubric"], template)

        question = kwargs.get("question") or kwargs.get("prompt") or ""
        audio_transcript = (
            kwargs.get("audio_transcript") or kwargs.get("source_transcript") or ""
        )
        direction = kwargs.get("direction") or ""
        wav_path = (
            kwargs.get("WavPath")
            or kwargs.get("wav_path")
            or kwargs.get("audio")
            or ""
        )
        wav_path = str(wav_path).strip()
        require_transcript = bool(kwargs.get("require_transcript"))

        if require_transcript and not str(audio_transcript).strip():
            return self._skip(
                pred_s,
                ref_s,
                "bad_audio_transcript",
                route=route,
                pid=pid,
                template=template,
            )

        if template == "multimodal":
            if not wav_path:
                return self._skip(
                    pred_s,
                    ref_s,
                    "missing_wavpath",
                    route=route,
                    pid=pid,
                    template=template,
                )
            if not os.path.isfile(wav_path):
                return self._skip(
                    pred_s,
                    ref_s,
                    f"wav_missing:{wav_path}",
                    route=route,
                    pid=pid,
                    template=template,
                )

        configured = kwargs.get("judge_model_name")
        if not configured:
            configured = (
                self.mm_judge_model_name
                if template == "multimodal"
                else self.judge_model_name
            )
        env_mm = os.environ.get("MESH_MM_JUDGE_MODEL", "").strip()
        if template == "multimodal" and env_mm:
            configured = env_mm

        try:
            resolve_judge_model_name(configured)
            model = get_judge_model(configured)
        except Exception as exc:  # noqa: BLE001
            return {
                **self._base(pred_s, ref_s, route, pid, template),
                "match": 0,
                "score_0_100": None,
                "skipped": 1,
                "skip_reason": f"judge_unavailable:{exc}",
            }

        try:
            real_prompt = build_prompt(
                capability=route["capability"],
                rubric=route["rubric"],
                template=template,
                question=str(question),
                pred=pred_s,
                reference=ref_s,
                audio_transcript=str(audio_transcript),
                direction=str(direction),
            )
        except ValueError as exc:
            return self._skip(
                pred_s,
                ref_s,
                f"prompt_error:{exc}",
                route=route,
                pid=pid,
                template=template,
            )

        n = int(kwargs.get("n_samples") or self.n_samples)
        raw_outputs: List[str] = []
        usage_rows: List[Dict[str, Any]] = []
        try:
            for _ in range(max(1, n)):
                if template == "multimodal":
                    prompt_struct = [
                        {
                            "role": "system",
                            "contents": [
                                {
                                    "type": "text",
                                    "value": (
                                        "You are a helpful assistant who tries to help "
                                        "answer the user's question."
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "contents": [
                                {"type": "audio", "value": wav_path},
                                {"type": "text", "value": real_prompt},
                            ],
                        },
                    ]
                    infer_kwargs: Dict[str, Any] = {}
                    if _is_openai_chat_judge(model):
                        infer_kwargs = {
                            "max_tokens": 64,
                            "temperature": 0.0,
                            "top_p": 1.0,
                        }
                    out = model.inference(prompt_struct, **infer_kwargs)
                elif _is_openai_chat_judge(model):
                    from audio_evals.registry import registry

                    wrap = registry.get_prompt("yes_no_judge")
                    out = model.inference(
                        wrap.load(real_prompt=real_prompt),
                        max_tokens=64,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        temperature=0.0,
                        top_p=1.0,
                    )
                else:
                    prompt_struct = [
                        {
                            "role": "user",
                            "contents": [
                                {"type": "text", "value": real_prompt},
                            ],
                        },
                    ]
                    out = model.inference(prompt_struct)
                raw_outputs.append("" if out is None else str(out))
                usage_rows.append(_collect_usage(model))
        except Exception as exc:  # noqa: BLE001
            return {
                **self._base(pred_s, ref_s, route, pid, template),
                "match": 0,
                "score_0_100": None,
                "skipped": 1,
                "skip_reason": f"judge_call_failed:{exc}",
                "raw_judge_output": raw_outputs,
                "judge_usage": usage_rows,
                "judge_model": getattr(model, "model_name", configured),
            }

        usage_agg = _aggregate_usage(usage_rows)
        base = {
            **self._base(pred_s, ref_s, route, pid, template),
            "raw_judge_output": raw_outputs,
            "judge_usage": usage_agg,
            "judge_model": getattr(model, "model_name", configured),
            "WavPath": wav_path or None,
            "audio_transcript_used": bool(str(audio_transcript).strip()),
        }

        binary = route["rubric"] in self.BINARY_RUBRICS
        if binary:
            votes = [parse_yes_no(x) for x in raw_outputs]
            votes = [v for v in votes if v is not None]
            if not votes:
                return {
                    **base,
                    "match": 0,
                    "score_0_100": None,
                    "skipped": 1,
                    "skip_reason": "judge_parse_failed",
                }
            yes = sum(votes)
            no = len(votes) - yes
            match = 1 if yes >= no else 0
            return {
                **base,
                "match": match,
                "score_0_100": 100.0 if match else 0.0,
                "gpt_score": float(match),
                "skipped": 0,
            }

        scores = [parse_score_1_to_5(x) for x in raw_outputs]
        scores = [s for s in scores if s is not None]
        if not scores:
            return {
                **base,
                "match": 0,
                "score_0_100": None,
                "skipped": 1,
                "skip_reason": "judge_parse_failed",
            }
        gpt_score = sum(scores) / len(scores)
        score_0_100 = float(gpt_score) / 5.0 * 100.0
        return {
            **base,
            "match": 1 if gpt_score >= 3.0 else 0,
            "score_0_100": score_0_100,
            "gpt_score": gpt_score,
            "skipped": 0,
        }

    @staticmethod
    def _base(pred, ref, route, pid, template):
        return {
            "pred": pred,
            "ref": ref,
            "capability": route["capability"],
            "rubric": route["rubric"],
            "prompt_id": pid,
            "judge_template": template,
        }

    def _skip(self, pred, ref, reason, route=None, pid=None, template=None):
        route = route or {"capability": None, "rubric": None}
        return {
            "pred": pred,
            "ref": ref,
            "capability": route.get("capability"),
            "rubric": route.get("rubric"),
            "prompt_id": pid,
            "judge_template": template or self.default_template,
            "match": 0,
            "score_0_100": None,
            "skipped": 1,
            "skip_reason": reason,
        }


def _aggregate_usage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    prompt = 0
    completion = 0
    total = 0
    have = False
    model = None
    for row in rows:
        if not row:
            continue
        model = row.get("model") or model
        pt = row.get("prompt_tokens")
        ct = row.get("completion_tokens")
        tt = row.get("total_tokens")
        if pt is not None or ct is not None or tt is not None:
            have = True
        prompt += int(pt or 0)
        completion += int(ct or 0)
        total += int(tt or ((pt or 0) + (ct or 0)))
    if not have:
        return {"model": model, "calls": len(rows)}
    return {
        "model": model,
        "calls": len(rows),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
