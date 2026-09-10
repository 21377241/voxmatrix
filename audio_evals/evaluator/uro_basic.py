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

        # Judge tracks
        if mode in ("open", "semi-open", "qa"):
            try:
                from audio_evals.evaluator.voice_bench import (
                    VoiceBenchQaOpenEvaluator,
                    get_judge_model,
                    resolve_judge_model_name,
                )

                resolve_judge_model_name(self.judge_model_name)
                get_judge_model(self.judge_model_name)
            except Exception as exc:  # noqa: BLE001
                return {
                    **base,
                    "match": 0,
                    "score_0_100": None,
                    "skipped": 1,
                    "skip_reason": f"judge_unavailable:{exc}",
                }

            question = kwargs.get("question") or kwargs.get("prompt") or ""
            # For open mode with empty gold, still score response vs instruction.
            judge = VoiceBenchQaOpenEvaluator(self.judge_model_name)
            judge_out = judge._eval(
                pred_s,
                refs[0] if refs else "",
                question=question,
                prompt=question,
            )
            gpt_score = judge_out.get("gpt_score")
            # VoiceBench open rating is 1-5; map to 0-100.
            if gpt_score is None:
                score_0_100 = None
                match = 0
            else:
                score_0_100 = float(gpt_score) / 5.0 * 100.0
                match = 1 if float(gpt_score) >= 3 else 0
            return {
                **base,
                "match": match,
                "score_0_100": score_0_100,
                "gpt_score": gpt_score,
                "skipped": 0,
            }

        return {
            **base,
            "match": 0,
            "score_0_100": None,
            "skipped": 1,
            "skip_reason": f"unknown_mode:{mode}",
        }
