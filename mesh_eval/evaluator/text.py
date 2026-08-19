from collections import Counter
from typing import Any, Dict

import sacrebleu

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import best_text_reference, normalize_text, text_tokens


class TextF1Evaluator(Evaluator):
    """SQuAD-style exact match and token F1 without model dependencies."""

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = best_text_reference(label, kwargs)
        references = reference if isinstance(reference, list) else [reference]
        pred_tokens = text_tokens(pred)
        best_f1 = 0.0
        best_em = 0
        for item in references:
            ref_tokens = text_tokens(item)
            common = Counter(pred_tokens) & Counter(ref_tokens)
            overlap = sum(common.values())
            if not pred_tokens and not ref_tokens:
                f1 = 1.0
            elif not pred_tokens or not ref_tokens:
                f1 = 0.0
            else:
                precision = overlap / len(pred_tokens)
                recall = overlap / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            best_f1 = max(best_f1, f1)
            best_em = max(best_em, int(normalize_text(pred) == normalize_text(item)))
        return {"text_f1": best_f1, "exact_match": best_em}


def _lcs_length(left: list[str], right: list[str]) -> int:
    if len(right) > len(left):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for l_item in left:
        current = [0]
        for index, r_item in enumerate(right, start=1):
            if l_item == r_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


class RougeLEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = best_text_reference(label, kwargs)
        references = reference if isinstance(reference, list) else [reference]
        pred_tokens = text_tokens(pred)
        best = 0.0
        for item in references:
            ref_tokens = text_tokens(item)
            if not pred_tokens and not ref_tokens:
                score = 1.0
            elif not pred_tokens or not ref_tokens:
                score = 0.0
            else:
                lcs = _lcs_length(pred_tokens, ref_tokens)
                precision = lcs / len(pred_tokens)
                recall = lcs / len(ref_tokens)
                score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            best = max(best, score)
        return {"rouge_l": best}


class RepetitionEvaluator(Evaluator):
    """Diagnostic repeated n-gram rate for long-form transcription."""

    def __init__(self, ngram_size: int = 4):
        self.ngram_size = ngram_size

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        tokens = text_tokens(pred)
        n = self.ngram_size
        ngrams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
        repeat_count = len(ngrams) - len(set(ngrams))
        rate = repeat_count / len(ngrams) if ngrams else 0.0
        return {"repeat_rate": rate}


class ChrFEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = best_text_reference(label, kwargs)
        references = reference if isinstance(reference, list) else [reference]
        score = sacrebleu.corpus_chrf([str(pred)], [[str(item)] for item in references]).score
        return {"chrf": score}


def _edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


class MixedErrorRateEvaluator(Evaluator):
    """Mixed Chinese-character/Latin-word error rate for code-switch ASR."""

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = best_text_reference(label, kwargs)
        pred_tokens = text_tokens(pred)
        ref_tokens = text_tokens(reference)
        error_rate = _edit_distance(ref_tokens, pred_tokens) / len(ref_tokens) if ref_tokens else float(bool(pred_tokens))
        return {"mer%": error_rate * 100}
