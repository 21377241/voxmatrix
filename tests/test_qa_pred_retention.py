from audio_evals.evaluator.base import ExistMatch
from audio_evals.evaluator.qa_exact_match import (
    QAExactMatchEvaluator,
    QAExistMatchEvaluator,
)
from audio_evals.evaluator.uro_basic import (
    UROBasicContentEvaluator,
    resolve_uro_mode,
    repeat_score_from_error_rate,
)
from audio_evals.evaluator.vocalbench_split import VocalBenchQASplitEvaluator


def test_qa_exist_match_keeps_model_pred_on_hit():
    pred = (
        "According to the audio, Luther argued that every good work "
        "designed to attract God's favor is a **sin**."
    )
    label = ["a sin", "sin", "sin", "sin"]
    out = QAExistMatchEvaluator()._eval(pred, label)
    assert out["match"] == 1
    assert out["pred"] == pred
    assert out["ref"] == label


def test_qa_exact_match_keeps_model_pred_on_hit():
    pred = "sin"
    label = "sin"
    out = QAExactMatchEvaluator()._eval(pred, label)
    assert out["match"] == 1
    assert out["pred"] == pred
    assert out["ref"] == label


def test_exist_match_keeps_model_pred_on_hit():
    pred = "the answer is paris"
    label = "paris"
    out = ExistMatch()._eval(pred, label)
    assert out["match"] == 1
    assert out["pred"] == pred
    assert out["ref"] == label


def test_qa_exist_match_chinese_sentence_contains_answer():
    pred = "世界第一条地铁于1863年建于英国伦敦。"
    out = QAExistMatchEvaluator()._eval(pred, "伦敦")
    assert out["match"] == 1
    assert out["pred"] == pred


def test_uro_mode_mapping_and_repeat_score():
    assert resolve_uro_mode("basic/Repeat") == "wer"
    assert resolve_uro_mode("basic/AlpacaEval") == "open"
    assert resolve_uro_mode("basic/Gsm8kEval") == "qa"
    assert repeat_score_from_error_rate(0.2) == 80.0
    assert repeat_score_from_error_rate(0.6) == 0.0


def test_uro_repeat_scores_without_judge():
    out = UROBasicContentEvaluator()._eval(
        "hello world",
        "hello world",
        subset_id="basic/Repeat",
        eval_mode="wer",
    )
    assert out["eval_mode"] == "wer"
    assert out["skipped"] == 0
    assert out["score_0_100"] == 100.0


def test_vocalbench_split_short_track():
    out = VocalBenchQASplitEvaluator()._eval(
        "Wales",
        "Wales",
        subset_id="knowledge",
        qa_track="short",
    )
    assert out["qa_track"] == "short"
    assert out["match"] == 1


def test_vocalbench_open_track_skips_without_judge():
    out = VocalBenchQASplitEvaluator()._eval(
        "long open answer",
        "reference recipe",
        subset_id="single_round",
        qa_track="open",
        question="How to cook?",
    )
    assert out["qa_track"] == "open"
    assert out.get("skipped") == 1
