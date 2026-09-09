import re
import string

import editdistance as ed
import zhconv
from zhon.hanzi import punctuation as zh_punctuation

from audio_evals.lib.evaluate_tokenizer import EvaluationTokenizer, TOKENIZERS
from audio_evals.lib.text_normalization.basic import BasicTextNormalizer
from audio_evals.lib.text_normalization.cn_tn import TextNorm
from audio_evals.lib.text_normalization.en import EnglishTextNormalizer

english_normalizer = EnglishTextNormalizer()
chinese_normalizer = TextNorm(
    to_banjiao=False,
    to_upper=False,
    to_lower=False,
    remove_fillers=False,
    remove_erhua=False,
    check_chars=False,
    remove_space=False,
    cc_mode="",
)
basic_normalizer = BasicTextNormalizer()

# Standalone backchannels / fillers — scoring them is not meaningful for ASR eval.
FILLER_ZH_CHARS = set("呃啊嗯唉诶哦噢哈呀哼额")
FILLER_EN_WORDS = frozenset(
    {"hmm", "mm", "mhm", "mmm", "uh", "um", "uhhuh", "huh", "aha"}
)
# Whole-utterance unlabelable refs (e.g. SBCSAE "xxx") — no scorable text.
PLACEHOLDER_REF_WORDS = frozenset(
    {"xxx", "x", "xx", "unk", "spn", "garbage", "<unk>"}
)


class FillerOnlyReferenceSkipped(Exception):
    """Reference should not be scored (filler-only or whole-utterance placeholder)."""

    def __init__(self, ref: str, language: str, reason: str = "filler_only"):
        self.ref = ref
        self.language = language
        self.reason = reason
        super().__init__(f"Reference skipped ({reason}): {ref!r}")


def _tokenizer_for_language(language: str) -> EvaluationTokenizer:
    if language == "yue":
        tokenizer_type = "zh"
    elif language == "jp":
        tokenizer_type = "ja-mecab"
    else:
        tokenizer_type = language if language in TOKENIZERS else "13a"
    return EvaluationTokenizer(
        tokenizer_type=tokenizer_type,
        lowercase=True,
        punctuation_removal=False,
        character_tokenization=False,
    )


def _normalize_for_wer(text: str, language: str) -> str:
    text = english_normalizer(text)
    if language in ["yue"]:
        text = zhconv.convert(text, "zh-cn")
    if language in ["zh", "yue"]:
        text = chinese_normalizer(text)
    return text


def normalized_ref_token_count(ref: str, language: str) -> int:
    norm_ref = _normalize_for_wer(ref, language)
    tokenizer = _tokenizer_for_language(language)
    return len(tokenizer.tokenize(norm_ref).split())


def _raw_ref_is_filler_only(ref: str, language: str) -> bool:
    if language in ("zh", "yue"):
        stripped = ref.strip()
        for ch in zh_punctuation + string.punctuation:
            stripped = stripped.replace(ch, "")
        stripped = stripped.replace(" ", "")
        return bool(stripped) and all(c in FILLER_ZH_CHARS for c in stripped)

    tokens = re.findall(r"[a-z]+", ref.lower())
    if not tokens:
        return False
    return all(token in FILLER_EN_WORDS for token in tokens)


def _raw_ref_is_whole_utterance_placeholder(ref: str, language: str) -> bool:
    stripped = ref.strip()
    for ch in zh_punctuation + string.punctuation:
        stripped = stripped.replace(ch, "")
    stripped = stripped.strip().lower()
    if not stripped:
        return False
    if language in ("zh", "yue"):
        compact = stripped.replace(" ", "")
        if compact in PLACEHOLDER_REF_WORDS or re.fullmatch(r"x+", compact):
            return True
        return False

    tokens = re.findall(r"[a-z0-9]+", stripped)
    if not tokens:
        return False
    return all(
        token in PLACEHOLDER_REF_WORDS or re.fullmatch(r"x+", token) for token in tokens
    )


def is_whole_utterance_placeholder_reference(ref: str, language: str) -> bool:
    return _raw_ref_is_whole_utterance_placeholder(ref, language)


def get_reference_skip_reason(ref: str, language: str) -> str | None:
    if is_whole_utterance_placeholder_reference(ref, language):
        return "placeholder_ref"
    if is_filler_only_reference(ref, language):
        return "filler_only"
    return None


def is_filler_only_reference(ref: str, language: str) -> bool:
    if normalized_ref_token_count(ref, language) == 0:
        return True
    if language in ("zh", "yue"):
        return _raw_ref_is_filler_only(ref, language)
    return _raw_ref_is_filler_only(ref, language)


def compute_wer(refs, hyps, language="13a"):
    distance = 0
    ref_length = 0
    tokenizer = _tokenizer_for_language(language)
    for i in range(len(refs)):
        ref = _normalize_for_wer(refs[i], language)
        pred = _normalize_for_wer(hyps[i], language)

        ref_items = tokenizer.tokenize(ref).split()
        pred_items = tokenizer.tokenize(pred).split()

        distance += ed.eval(ref_items, pred_items)
        ref_length += len(ref_items)
    if ref_length == 0:
        raise ValueError("WER reference corpus is empty after normalization")
    return distance / ref_length
