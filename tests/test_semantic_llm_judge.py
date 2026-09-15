"""Unit tests for semantic LLM judge prompt routing (no network)."""

from audio_evals.evaluator.llm_judge_prompts import build_prompt, prompt_id, resolve_route
from audio_evals.evaluator.semantic_llm_judge import parse_score_1_to_5, parse_yes_no


def test_resolve_route_defaults():
    assert resolve_route("speech_translation")["rubric"] == "translation"
    assert resolve_route("qa", has_reference=False)["rubric"] == "open"
    assert resolve_route("qa", eval_mode="qa", has_reference=True)["rubric"] == "binary"
    assert resolve_route("audio_reasoning", has_reference=True)["rubric"] == "semi-open"
    assert resolve_route("code_switch", has_reference=True)["rubric"] == "binary"


def test_prompt_omits_empty_transcript():
    text = build_prompt(
        capability="qa",
        rubric="open",
        template="text",
        question="How are you?",
        pred="I am fine.",
        audio_transcript="",
    )
    assert "### [AudioTranscript]" not in text
    assert "How are you?" in text
    assert "I am fine." in text


def test_qa_uro_scale_with_text_eval_preamble():
    qa = build_prompt(capability="qa", rubric="open", question="Q", pred="A")
    # text-eval framing (not speech-out transcription claim)
    assert "text" in qa.lower() and "speech output" not in qa.lower()
    assert "spoken-input" in qa or "spoken-input understanding" in qa
    # URO open scale kept
    assert "1 point:" in qa and "5 points:" in qa
    assert "extraneous information" in qa
    assert "After evaluating, please output the score only" in qa

    semi = build_prompt(
        capability="qa",
        rubric="semi-open",
        question="Q",
        pred="A",
        reference="R",
    )
    assert "doesn’t necessarily have to be identical" in semi
    assert "suggested answer" in semi

    binary = build_prompt(
        capability="qa",
        rubric="binary",
        question="Q",
        pred="A",
        reference="R",
    )
    assert "Yes" in binary and "No" in binary


def test_non_qa_aligned_to_same_standard():
    ar = build_prompt(
        capability="audio_reasoning",
        rubric="semi-open",
        question="Q",
        pred="A",
        reference="R",
        audio_transcript="someone knocked three times",
    )
    st = build_prompt(
        capability="speech_translation",
        rubric="translation",
        question="",
        pred="Hello",
        reference="Hello",
        audio_transcript="你好",
        direction="Translate into English",
    )
    cs = build_prompt(
        capability="code_switch",
        rubric="binary",
        question="What is the price?",
        pred="twenty yuan",
        reference="20 yuan",
        audio_transcript="这个多少 money",
    )
    # shared text-eval scene
    for text in (ar, st, cs):
        assert "Score **content quality only**" in text or "content quality only" in text.lower()
        assert "speech output" not in text.lower()
    assert "answer accuracy" in ar
    assert "reasoning quality" in ar or "logically rigorous" in ar
    assert "1 point:" in ar and "Soft-align to Reference" in ar
    assert "adequacy" in st
    assert "1 point:" in st and "Soft-align to Reference" in st
    assert "CODE-SWITCHING" in cs or "code-switching" in cs.lower()
    assert "Yes" in cs and "No" in cs
    cs_open = build_prompt(
        capability="code_switch",
        rubric="open",
        question="Explain self-attention",
        pred="It weights tokens by similarity.",
    )
    assert "1 point:" in cs_open
    assert "no single gold phrasing" in cs_open.lower() or "Do NOT require a single gold" in cs_open
    assert prompt_id("qa", "open", "text") == "qa.open.text"


def test_multimodal_preface():
    text = build_prompt(
        capability="qa",
        rubric="open",
        template="multimodal",
        question="Q",
        pred="A",
        audio_transcript="hello from asr",
    )
    assert "ASR transcript" in text or "transcript" in text.lower()
    assert "### [AudioTranscript]" in text
    assert "hello from asr" in text


def test_parsers():
    assert parse_score_1_to_5("4") == 4.0
    assert parse_score_1_to_5("Score: [[5]]") == 5.0
    assert parse_yes_no("Yes") == 1
    assert parse_yes_no("no, incorrect") == 0
