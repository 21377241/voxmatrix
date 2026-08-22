from abc import ABC, abstractmethod
from typing import Dict, List

import sacrebleu
from jiwer import cer, wer

from audio_evals.lib.wer import compute_wer


class AggPolicy(ABC):
    def __init__(self, need_score_col: List[str] = None):
        self.need_score_col = need_score_col
        if need_score_col is None:
            self.need_score_col = []

    @abstractmethod
    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, any]:
        raise NotImplementedError()

    def __call__(self, score_detail: List[Dict[str, any]]) -> Dict[str, any]:
        if len(score_detail) > 0:
            for col in self.need_score_col:
                assert col in score_detail[0], ValueError(
                    f"not found {col} score, but {score_detail[0]}"
                )
        try:
            return self._agg(score_detail)
        except Exception as e:
            return {"error": str(e)}


class WER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, ignore_case: bool = False):
        super().__init__(need_score_col)
        self.ignore_case = ignore_case

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        if self.ignore_case:
            predl, refl = [item.lower() for item in predl], [
                item.lower() for item in refl
            ]
        return {"wer(%)": wer(refl, predl) * 100}


class PracticeWER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, lang: str = "13a"):
        super().__init__(need_score_col)
        self.lang = lang

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        predl, refl = [item.lower() for item in predl], [item.lower() for item in refl]
        return {"wer(%)": compute_wer(refl, predl, self.lang) * 100}


class ACC(AggPolicy):

    def __call__(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        # Accuracy is a core contract: malformed rows should fail loudly instead
        # of being converted into an {"error": ...} payload by AggPolicy.
        return self._agg(score_detail)

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score

        if not score_detail:
            accuracy = 0.0
        elif all("match" in item for item in score_detail):
            accuracy = sum(float(item["match"]) for item in score_detail) / len(
                score_detail
            )
        else:
            predl = [str(item["pred"]) for item in score_detail]
            refl = [str(item["ref"]) for item in score_detail]
            accuracy = float(accuracy_score(refl, predl))
        return {"acc": accuracy, "acc(%)": accuracy * 100}


class NaiveMean(AggPolicy):
    def __init__(self, need_score_col: List[str] = None):
        super().__init__(need_score_col)

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        res = {}
        if not self.need_score_col:
            for k, v in score_detail[0].items():
                if isinstance(v, (int, float)):
                    self.need_score_col.append(k)
                else:
                    print(f"ignore {k} as it is not a number, but {v}")
        for item in self.need_score_col:
            valid_l = [c[item] for c in score_detail if c.get(item) is not None]
            if "%" in item:
                res[item] = sum(valid_l) / len(valid_l)
            else:
                res[f"{item}(%)"] = sum(valid_l) / len(valid_l) * 100
        return res


class CER(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, ignore_case: bool = False):
        super().__init__(need_score_col)
        self.ignore_case = ignore_case

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]
        if self.ignore_case:
            predl, refl = [item.lower() for item in predl], [
                item.lower() for item in refl
            ]
        return {"cer": cer(predl, refl)}


class Dump(AggPolicy):

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        return {}


class BLEU(AggPolicy):
    def __init__(self, need_score_col: List[str] = None, lang: str = "13a"):
        super().__init__(need_score_col)
        self.lang = "13a"
        if lang == "zh":
            self.lang = "zh"
        elif lang == "ja":
            self.lang = "ja-mecab"
        else:
            self.lang = lang

    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        predl, refl = [str(item["pred"]) for item in score_detail], [
            str(item["ref"]) for item in score_detail
        ]

        pred, ref = [], []
        for p, r in zip(predl, refl):
            if r:
                pred.append(p)
                ref.append(r)
        res = sacrebleu.corpus_bleu(pred, [ref], tokenize=self.lang)
        return {"bleu": res.score}


class Coco(AggPolicy):
    def _agg(self, score_detail: List[Dict[str, any]]) -> Dict[str, float]:
        from audio_evals.lib.coco import compute_caption

        predl, refl = [str(item["pred"]) for item in score_detail], [
            item["ref"] for item in score_detail
        ]

        pred, ref = [], []
        for p, r in zip(predl, refl):
            if r:
                pred.append(p)
                if isinstance(r, str):
                    r = [r]
                ref.append(r)
        res = compute_caption(ref, pred)
        return res
