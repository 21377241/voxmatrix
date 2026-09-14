"""URO-Bench basic-track content scoring (task accomplish, no UTMOS/latency)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.wer import compute_wer


# subset_id (basic/<name>) -> scoring mode
URO_BASIC_MODE = {
    "basic/Repeat": "wer",
    "basic/Repeat-zh": "cer",
    "basic/Summary": "semi-open",
    "basic/LCSTS-zh": "semi-open",
    "basic/StoralEval": "semi-open",
    "basic/TruthfulEval": "semi-open",
    "basic/GaokaoEval": "qa",
    "basic/Gsm8kEval": "qa",
    "basic/MLC": "qa",
    "basic/HSK5-zh": "qa",
    "basic/APE-zh": "qa",
    "basic/MLC-zh": "qa",
    "basic/OpenbookQA-zh": "qa",
    "basic/SQuAD-zh": "qa",
    "basic/AlpacaEval": "open",
    "basic/AlpacaEval-zh": "open",
    "basic/CommonEval": "open",
    "basic/WildchatEval": "open",
    "basic/Wildchat-zh": "open",
    "basic/Claude-zh": "open",
}


def resolve_uro_mode(subset_id: Optional[str], eval_mode: Optional[str] = None) -> str:
    if eval_mode:
        return str(eval_mode)
    if not subset_id:
        return "open"
    if subset_id in URO_BASIC_MODE:
        return URO_BASIC_MODE[subset_id]
    # tolerate bare names
    keyed = subset_id if subset_id.startswith("basic/") else f"basic/{subset_id}"
    return URO_BASIC_MODE.get(keyed, "open")


def repeat_score_from_error_rate(error_rate: float) -> float:
    """Official-style mapping: error<=0.5 -> 100*(1-error), else 0."""
    if error_rate <= 0.5:
        return 100.0 * (1.0 - error_rate)
    return 0.0


class UROBasicContentEvaluator(Evaluator):
    """Per-sample content score in [0, 100] plus match in {0,1} for ACC-compatible aggs.

    - wer/cer: local, no judge
    - open / semi-open / qa: LLM judge when credentials exist; otherwise mark skipped
    """

    def __init__(self, judge_model_name: str = "gpt4o-mini"):
        self.judge_model_name = judge_model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, Any]:
        subset_id = kwargs.get("subset_id")
        mode = resolve_uro_mode(subset_id, kwargs.get("eval_mode"))
        pred_s = "" if pred is None else str(pred)
        refs: List[str]
        if isinstance(label, list):
            refs = [str(x) for x in label if x is not None and str(x).strip() != ""]
        elif label is None or str(label).strip() == "":
            refs = []
        else:
            refs = [str(label)]

        base = {
            "pred": pred_s,
            "ref": refs if len(refs) != 1 else refs[0],
            "subset_id": subset_id,
            "eval_mode": mode,
        }

        if mode in ("wer", "cer"):
            if not refs:
                return {
                    **base,
                    "match": 0,
                    "score_0_100": 0.0,
                    "skipped": 1,
                    "skip_reason": "repeat_missing_reference",
                }
            lang = "zh" if mode == "cer" or str(subset_id).endswith("-zh") else "en"
            err = float(compute_wer(refs[:1], [pred_s], language=lang))
            score = repeat_score_from_error_rate(err)
            return {
                **base,
                "match": 1 if score > 0 else 0,
                "score_0_100": score,
                "error_rate": err,
                "skipped": 0,
            }

        # Judge tracks: URO Appendix E open / semi-open / qa via SemanticLLMJudge
        if mode in ("open", "semi-open", "qa"):
            from audio_evals.evaluator.semantic_llm_judge import SemanticLLMJudgeEvaluator

            rubric = "binary" if mode == "qa" else mode
            judge = SemanticLLMJudgeEvaluator(judge_model_name=self.judge_model_name)
            question = kwargs.get("question") or kwargs.get("prompt") or ""
            judge_kwargs = {
                "capability": "qa",
                "rubric": rubric,
                "eval_mode": mode,
                "question": question,
                "judge_template": kwargs.get("judge_template") or "text",
                "audio_transcript": kwargs.get("audio_transcript")
                or kwargs.get("source_transcript")
                or "",
                "WavPath": kwargs.get("WavPath") or kwargs.get("wav_path") or "",
                "judge_model_name": kwargs.get("judge_model_name") or self.judge_model_name,
            }
            judge_out = judge._eval(pred_s, refs[0] if refs else "", **judge_kwargs)
            out = {
                **base,
                "match": judge_out.get("match", 0),
                "score_0_100": judge_out.get("score_0_100"),
                "gpt_score": judge_out.get("gpt_score"),
                "skipped": judge_out.get("skipped", 0),
                "prompt_id": judge_out.get("prompt_id"),
                "judge_template": judge_out.get("judge_template"),
                "raw_judge_output": judge_out.get("raw_judge_output"),
                "judge_usage": judge_out.get("judge_usage"),
            }
            if judge_out.get("skip_reason"):
                out["skip_reason"] = judge_out["skip_reason"]
            return out

        return {
            **base,
            "match": 0,
            "score_0_100": None,
            "skipped": 1,
            "skip_reason": f"unknown_mode:{mode}",
        }
