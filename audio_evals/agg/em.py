from __future__ import annotations

from audio_evals.agg.base import AggPolicy


class EMMean(AggPolicy):
    def _agg(self, score_detail):
        values = [float(item["match"]) for item in score_detail if item.get("match") is not None]
        return {"em": sum(values) / len(values) if values else 0.0}
