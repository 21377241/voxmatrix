from typing import Dict, List

from sklearn.metrics import accuracy_score

from audio_evals.agg.base import AggPolicy


class OriginalACC(AggPolicy):
    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predictions = [str(item["pred"]) for item in score_detail]
        references = [str(item["ref"]) for item in score_detail]
        return {"acc(%)": accuracy_score(references, predictions) * 100}
