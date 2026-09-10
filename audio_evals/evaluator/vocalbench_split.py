"""VocalBench QA: route knowledge (short) vs single_round (open) separately."""

from __future__ import annotations

from typing import Any, Dict

from audio_evals.evaluator.base import Evaluator
from audio_evals.evaluator.qa_exact_match import QAExistMatchEvaluator


class VocalBenchQASplitEvaluator(Evaluator):
    """Do not merge short-answer accuracy with open-ended judge into one number.

    - knowledge*: qa-exist-match (with CJK containment fallback)
    - single_round / open*: LLM judge when available; otherwise skipped
    """

    def __init__(self, judge_model_name: str = "gpt4o-mini"):
        self.judge_model_name = judge_model_name
        self._short = QAExistMatchEvaluator()

    def _eval(self, pred, label, **kwargs) -> Dict[str, Any]:
        subset = str(kwargs.get("subset_id") or "")
        track = kwargs.get("qa_track")
        if not track:
            if "single_round" in subset or subset in {"open_ended", "open"}:
                track = "open"
            else:
                track = "short"

        if track == "short":
            out = self._short._eval(pred, label, **kwargs)
            out["qa_track"] = "short"
            out["subset_id"] = subset
            return out

        # open track
        try:
            from audio_evals.evaluator.voice_bench import (
                VoiceBenchQaOpenEvaluator,
                get_judge_model,
                resolve_judge_model_name,
            )

            resolve_judge_model_name(self.judge_model_name)
            get_judge_model(self.judge_model_name)
            judge = VoiceBenchQaOpenEvaluator(self.judge_model_name)
            question = kwargs.get("question") or kwargs.get("prompt") or ""
            judged = judge._eval(pred, label, question=question, prompt=question)
            gpt = judged.get("gpt_score")
            return {
                "pred": pred,
                "ref": label,
                "match": 1 if gpt is not None and float(gpt) >= 3 else 0,
                "gpt_score": gpt,
                "qa_track": "open",
                "subset_id": subset,
                "skipped": 0,
            }
        except Exception as exc:  # noqa: BLE001
            return {
                "pred": pred,
                "ref": label,
                "match": 0,
                "qa_track": "open",
                "subset_id": subset,
                "skipped": 1,
                "skip_reason": f"judge_unavailable:{exc}",
            }
