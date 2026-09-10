import re
import unicodedata
from typing import Dict

from audio_evals.evaluator.base import Evaluator
import regex
import string

'''
from https://github.com/DevSinghSachan/emdr2/blob/edb8cf6701bfa4ad7c961a21cdb461f4f7a0c72f/tasks/openqa/dense_retriever/evaluation/qa_validation.py
'''

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


class QAExactMatchEvaluator(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:

        if isinstance(label, list):
            for item in label:
                ans = self._eval(pred, item, **kwargs)
                if ans["match"] == 1:
                    return ans
            return {"match": 0, "pred": pred, "ref": label}

        match = exact_match_score(str(pred), str(label))

        return {
            "match": 1 if match else 0,
            # Keep the model output in pred for auditability; never overwrite with ref.
            "pred": pred,
            "ref": label,
        }


class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens


s_tokenizer = SimpleTokenizer()


def _normalize(text):
    return unicodedata.normalize('NFD', text)


_PUNCT_TABLE = str.maketrans("", "", string.punctuation + "。，、；：？！「」『』（）【】《》—…·")


def _collapse_for_containment(text: str) -> str:
    """Lowercase + strip punctuation/whitespace for character-level containment."""
    text = _normalize(str(text)).lower()
    text = text.translate(_PUNCT_TABLE)
    return "".join(text.split())


def has_answer_char_containment(answers, text) -> bool:
    """True if any answer string is a substring of text after light normalization.

    Needed for CJK: SimpleTokenizer keeps a whole Chinese sentence as one token,
    so token-span matching cannot find a short answer inside a long sentence.
    """
    haystack = _collapse_for_containment(text)
    if not haystack:
        return False
    for answer in answers:
        needle = _collapse_for_containment(answer)
        if needle and needle in haystack:
            return True
    return False


def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text_norm = _normalize(text)
    text_tokens = tokenizer.tokenize(text_norm, uncased=True)

    for answer in answers:
        answer_tokens = tokenizer.tokenize(_normalize(answer), uncased=True)
        for i in range(0, len(text_tokens) - len(answer_tokens) + 1):
            if answer_tokens and answer_tokens == text_tokens[i : i + len(answer_tokens)]:
                return True
    # Fallback for CJK / unsegmented spans where token matching fails.
    return has_answer_char_containment(answers, text)


class QAExistMatchEvaluator(Evaluator):

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:

        if isinstance(label, str):
            label = [label]

        match = has_answer(label, str(pred), s_tokenizer)

        return {
            "match": 1 if match else 0,
            # Keep the model output in pred for auditability; never overwrite with ref.
            "pred": pred,
            "ref": label,
        }