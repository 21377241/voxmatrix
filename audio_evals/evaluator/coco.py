import re
from typing import Any, List, Union

from audio_evals.evaluator.base import Evaluator


def normalize_caption(prediction: Any, max_words: int = 64) -> str:
    text = " ".join(str(prediction).strip().split())
    sentences = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    first_sentence = sentences[0] if sentences else text
    return " ".join(first_sentence.split()[:max_words])


def caption_references(label: Union[str, List[str]], kwargs) -> List[str]:
    reference = kwargs.get("reference_obj") or kwargs.get("reference") or {}
    candidates = None
    if isinstance(reference, dict):
        candidates = (
            reference.get("captions")
            or reference.get("acceptable_answers")
            or reference.get("caption")
            or reference.get("text")
        )
    if candidates in (None, "", []):
        candidates = label
    if not isinstance(candidates, list):
        candidates = [candidates]
    return [str(item) for item in candidates if item not in (None, "")]


class Coco(Evaluator):
    def __init__(self, max_words: int = 64):
        self.max_words = max_words

    def _eval(self, pred: str, label: Union[str, List[str]], **kwargs):
        return {
            "caption_prediction": normalize_caption(pred, self.max_words),
            "caption_references": caption_references(label, kwargs),
        }
