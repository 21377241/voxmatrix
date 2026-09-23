from __future__ import annotations

from audio_evals.agg.base import AggPolicy


class FenseMean(AggPolicy):
    def _agg(self, score_detail):
        scores = [float(item["fense"]) for item in score_detail if item.get("fense") is not None]
        return {"FENSE": sum(scores) / len(scores) if scores else 0.0}
