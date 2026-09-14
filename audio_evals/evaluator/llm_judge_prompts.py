"""Task-specific LLM-judge prompt builders for open/semantic capabilities.

prompt_id form: ``{capability}.{rubric}.{template}`` with template in {text, multimodal}.

Design norm (URO-Bench aligned, not verbatim copy):
  - Rubrics: open / semi-open / binary (+ translation for ST)
  - Detailed 1–5 scales (URO-level elaboration), or Yes/No for binary
  - Semi-open: soft alignment to reference (paraphrase OK)
  - Output score / Yes|No only; no explanations
  - Default: text judge of **text** Response (content track). Scenario text must
    not claim speech-out transcription when the pipeline only scores text pred.
  - Optional [AudioTranscript] for template A1; multimodal preface for template B.

QA keeps URO Appendix E **rating scales & principles**; preamble is adjusted for
text-response eval. AR / ST / CS use the same structural standard with task Focus.

See ``8p31/交付/开放语义_双模板LLM-Judge评测流程.md`` §1, §6.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

MM_PREFACE = (
    "You are also given the same user audio the model heard. "
    "Use the audio to verify content; do not score TTS quality.\n\n"
)

OUTPUT_SCORE = (
    "After evaluating, please output the score only without anything else.\n"
    "You don’t need to provide any explanations."
)
OUTPUT_YESNO = 'Please only output a single "Yes" or "No". Do not output anything else.'

# Shared scene for text-content judging (spoken input → text response).
_TEXT_SCENE = (
    "You are evaluating model responses in a spoken-input understanding scenario. "
    "The user spoke; the model produced a **text** response (or a transcript of a "
    "spoken response). Score **content quality only**—do not score speech waveform, "
    "TTS timbre, or latency."
)


def _block(title: str, value: Optional[str]) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    return f"### [{title}]\n{text}\n"


def _join_parts(*parts: str) -> str:
    return "\n".join(p for p in parts if p and p.strip()).strip() + "\n"


def build_prompt(
    *,
    capability: str,
    rubric: str,
    template: str = "text",
    question: str = "",
    pred: str = "",
    reference: str = "",
    audio_transcript: str = "",
    direction: str = "",
) -> str:
    """Build a full judge prompt; empty optional slots are omitted entirely."""
    capability = (capability or "").strip().lower()
    rubric = (rubric or "").strip().lower()
    template = (template or "text").strip().lower()
    if template not in ("text", "multimodal"):
        raise ValueError(f"unsupported judge template: {template}")

    builders = {
        ("qa", "open"): _qa_open,
        ("qa", "semi-open"): _qa_semi_open,
        ("qa", "binary"): _qa_binary,
        ("audio_reasoning", "open"): _ar_open,
        ("audio_reasoning", "semi-open"): _ar_semi_open,
        ("speech_translation", "translation"): _st_translation,
        ("code_switch", "binary"): _cs_binary,
        ("code_switch", "semi-open"): _cs_semi_open,
        ("code_switch", "open"): _cs_open,
    }
    key = (capability, rubric)
    if key not in builders:
        raise ValueError(f"unsupported prompt_id base: {capability}.{rubric}")

    body = builders[key](
        question=question,
        pred=pred,
        reference=reference,
        audio_transcript=audio_transcript,
        direction=direction,
        template=template,
    )
    if template == "multimodal":
        return MM_PREFACE + body
    return body


def prompt_id(capability: str, rubric: str, template: str = "text") -> str:
    return f"{capability}.{rubric}.{template}"


def resolve_route(
    capability: str,
    rubric: Optional[str] = None,
    *,
    eval_mode: Optional[str] = None,
    has_reference: bool = False,
) -> Dict[str, str]:
    """Map capability (+ optional hints) to capability/rubric defaults."""
    cap = (capability or "").strip().lower()
    mode = (eval_mode or "").strip().lower()
    rub = (rubric or "").strip().lower()

    if not rub:
        if cap == "speech_translation":
            rub = "translation"
        elif cap == "code_switch":
            rub = "binary" if has_reference else "semi-open"
        elif cap == "audio_reasoning":
            rub = "semi-open" if has_reference else "open"
        elif cap == "qa":
            if mode in ("open", "semi-open", "qa"):
                rub = "binary" if mode == "qa" else mode
            else:
                rub = "semi-open" if has_reference else "open"
        else:
            raise ValueError(f"cannot resolve rubric for capability={capability!r}")

    return {"capability": cap, "rubric": rub}


# --- QA: URO scales + text-eval preamble ------------------------------------


def _qa_open(**kw: Any) -> str:
    return _join_parts(
        _TEXT_SCENE,
        "Your task is to rate the model’s text response based on the user instruction "
        "[Instruction] and the model’s answer [Response]. "
        "If [AudioTranscript] is present, treat it as an ASR of the user speech to aid "
        "understanding; if absent, rely on [Instruction] alone.",
        "[Focus]\n"
        "- Prioritize relevance, accuracy, completeness, then conciseness "
        "(URO open dimensions).\n"
        "- Do NOT require a single gold phrasing (there is no reference).",
        # URO Appendix E open scale (kept as the rating standard)
        "Please evaluate the response on a scale of 1 to 5:\n"
        "1 point: The response is largely irrelevant, incorrect, or fails to address "
        "the user’s query. It may be off-topic or provide incorrect information.\n"
        "2 points: The response is somewhat relevant but lacks accuracy or completeness. "
        "It may only partially answer the user’s question or include extraneous information.\n"
        "3 points: The response is relevant and mostly accurate, but it may lack "
        "conciseness or include unnecessary details that don’t contribute to the main point.\n"
        "4 points: The response is relevant, accurate, and concise, providing a clear "
        "answer to the user’s question without unnecessary elaboration.\n"
        "5 points: The response is exceptionally relevant, accurate, and to the point. "
        "It directly addresses the user’s query in a highly effective and efficient "
        "manner, providing exactly the information needed.",
        "Below are the instruction and the model’s response:",
        _block("Instruction", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        OUTPUT_SCORE,
    )


def _qa_semi_open(**kw: Any) -> str:
    return _join_parts(
        _TEXT_SCENE,
        "Your task is to rate the model’s text response based on [Instruction], "
        "[Response], and suggested answers [Reference]. "
        "The model’s response doesn’t necessarily have to be identical to the suggested "
        "answers, as long as it aligns with the question and is reasonable.",
        "[Focus]\n"
        "- Soft-align to Reference (URO semi-open): paraphrase OK if meaning matches.\n"
        "- Prioritize relevance, accuracy, completeness, conciseness vs the question.",
        # URO Appendix E semi-open scale
        "Please evaluate the response on a scale of 1 to 5:\n"
        "1 point: The response is largely irrelevant, incorrect, or fails to address "
        "the user’s query. It may be off-topic or provide incorrect information. "
        "The response does not align with the question in any meaningful way.\n"
        "2 points: The response is somewhat relevant but lacks accuracy, completeness, "
        "or coherence. It may partially address the query but introduces unnecessary "
        "information or deviates from the core issue. The response may not align well "
        "with the suggested answer but still provides some value.\n"
        "3 points: The response is relevant and mostly accurate, but may lack "
        "conciseness or clarity. It addresses the question reasonably, but there might "
        "be slight deviations in approach or content. While it may not strictly align "
        "with the suggested answer, it still effectively addresses the core of the query.\n"
        "4 points: The response is relevant, accurate, and concise. It provides a clear "
        "answer to the user’s question and avoids unnecessary details. While it may not "
        "exactly mirror the suggested answer, it effectively addresses the user’s query "
        "in a logical and well-reasoned manner.\n"
        "5 points: The response is exceptionally relevant, accurate, and concise. It "
        "directly addresses the user’s query in the most efficient manner, providing "
        "exactly the information needed. The response may differ from the suggested "
        "answer in phrasing or approach but still aligns perfectly with the intent of "
        "the query, demonstrating a high level of reasoning and clarity.",
        "Below are the instruction, the model’s response, and the reference:",
        _block("Instruction", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        _block("Reference", kw.get("reference")),
        OUTPUT_SCORE,
    )


def _qa_binary(**kw: Any) -> str:
    return _join_parts(
        _TEXT_SCENE,
        "Your task is to decide whether the model’s text response is correct given "
        "[Question] and the correct answer [Reference] (URO qa / Yes-No track).",
        "[Focus]\n"
        "- Judge semantic equivalence to the reference for this question.\n"
        "- Ignore extra polite fluff unless it changes the answer.",
        "Below are the question, the model’s response, and the reference answer:",
        _block("Question", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        _block("Reference", kw.get("reference")),
        "Is the model’s response correct based on the question and reference answer?\n"
        + OUTPUT_YESNO,
    )


# --- AR / ST / CS: same structural standard + task Focus --------------------


def _ar_body(*, with_ref_focus: bool, **kw: Any) -> str:
    focus = (
        "[Focus]\n"
        "- Highest priority: groundedness on the audio evidence (or its transcript).\n"
        "- Then: correctness of the conclusion; then clarity of reasoning if shown.\n"
        "- Penalize hard: answers that ignore the audio, invent unaudible facts, or "
        "only restate the question without using audible information."
    )
    if with_ref_focus:
        focus += (
            "\n- Soft-align to Reference (URO semi-open principle): treat it as a "
            "suggested correct conclusion; accept equivalent reasoning paths / paraphrases."
        )
    return _join_parts(
        _TEXT_SCENE,
        "This is an AUDIO REASONING task: the model must use information present in the "
        "spoken/audio input (events, counts, relations, speakers, etc.) to answer—"
        "not generic world knowledge alone.",
        focus,
        "Please evaluate the response on a scale of 1 to 5:\n"
        "1 point: The response is ungrounded or wrong. It shows little to no use of "
        "audio evidence, invents facts not supported by the input, or fails the question.\n"
        "2 points: The response is only partially grounded. A major audio-dependent "
        "detail is missing or wrong, though some content may still be related.\n"
        "3 points: The response is mostly grounded and plausible, but has minor gaps, "
        "unclear steps, or small mistakes relative to the audio evidence"
        + (
            " or the suggested reference intent.\n"
            if with_ref_focus
            else ".\n"
        )
        + "4 points: The response is well grounded and reaches a correct conclusion "
        "with clear enough reasoning. Wording may differ from a suggested reference.\n"
        "5 points: The response is fully grounded, precise, and faithfully reflects "
        "the audio evidence"
        + (
            ", aligning with the intent of the suggested reference even if phrasing differs.\n"
            if with_ref_focus
            else ".\n"
        ),
        "Below are the question and the model’s response:",
        _block("Question", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        _block("Reference", kw.get("reference") if with_ref_focus else ""),
        OUTPUT_SCORE,
    )


def _ar_open(**kw: Any) -> str:
    return _ar_body(with_ref_focus=False, **kw)


def _ar_semi_open(**kw: Any) -> str:
    return _ar_body(with_ref_focus=True, **kw)


def _st_translation(**kw: Any) -> str:
    direction = kw.get("direction") or "Translate the source speech into the target language."
    if kw.get("template") == "multimodal":
        asr_note = (
            "- Listen to the source audio; use Reference as a suggested target rendering "
            "(soft alignment: paraphrase OK if meaning matches)."
        )
    else:
        asr_note = (
            "- If SourceTranscript may contain ASR errors, prefer meaning consistent with "
            "both transcript and reference; when they conflict, trust Reference for target "
            "meaning and reflect uncertainty only via a lower score (still output one score).\n"
            "- Soft-align to Reference (URO semi-open principle): paraphrase OK if meaning matches."
        )
    return _join_parts(
        _TEXT_SCENE,
        "This is SPEECH TRANSLATION (speech → target-language **text**). "
        "Score translation content quality only.",
        "[Focus]\n"
        "- Primary: adequacy — preserve meaning, entities, numbers, negation, and intent.\n"
        "- Secondary: fluency in the target language.\n"
        "- Do NOT reward literal n-gram overlap with the reference if meaning drifted.\n"
        "- Do NOT penalize style differences if content is faithful.\n"
        + asr_note,
        "Please evaluate the response on a scale of 1 to 5:\n"
        "1 point: Severe omission or mistranslation; largely unrelated to the source meaning. "
        "Does not align with the reference intent in any meaningful way.\n"
        "2 points: Only partial meaning is preserved; key facts are missing or wrong. "
        "May weakly relate to the suggested reference but provides limited value.\n"
        "3 points: Main idea is present with noticeable errors, omissions, or ambiguity. "
        "Not a strict match to the reference, but still addresses the core meaning.\n"
        "4 points: Faithful and mostly fluent; only minor issues. May not mirror the "
        "reference wording, but conveys the source meaning clearly and reasonably.\n"
        "5 points: Faithful, complete, and fluent; terminology and tone are appropriate. "
        "May differ from the reference in phrasing while aligning with its meaning.",
        "Below are the direction, source side, model translation, and reference:",
        _block("Direction", direction),
        _block("SourceTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        _block("Reference", kw.get("reference")),
        OUTPUT_SCORE,
    )


def _cs_binary(**kw: Any) -> str:
    return _join_parts(
        _TEXT_SCENE,
        "This is CODE-SWITCHING spoken QA: the user speech may mix languages "
        "(e.g. Mandarin–English). Decide whether the text response is correct.",
        "[Focus]\n"
        "- Semantic match of Response to Question + Reference (URO qa / Yes-No track).\n"
        "- Code-switching in the RESPONSE is NOT required unless the question demands it.\n"
        "- Be strict on critical entities, numbers, and names that depend on hearing the mix.",
        "Below are the question, the model’s response, and the reference answer:",
        _block("Question", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        _block("Reference", kw.get("reference")),
        "Is the model’s response correct based on the question and reference answer?\n"
        + OUTPUT_YESNO,
    )


def _cs_semi_open(**kw: Any) -> str:
    return _join_parts(
        _TEXT_SCENE,
        "This is CODE-SWITCHING spoken QA with a suggested reference. "
        "The user speech may mix languages.",
        "[Focus]\n"
        "- Understand the mixed-language question and answer correctly.\n"
        "- Soft-align to Reference (URO semi-open): paraphrase OK if meaning matches.\n"
        "- Code-switching in the RESPONSE is NOT required unless the question demands it.\n"
        "- Be strict on critical entities/numbers/names from the audio.",
        "Please evaluate the response on a scale of 1 to 5:\n"
        "1 point: Wrong or off-topic; misses the mixed-speech intent. Does not align "
        "with the question or reference in any meaningful way.\n"
        "2 points: Partially correct; a key mixed-speech detail is wrong or missing. "
        "May not align well with the suggested answer but still has some value.\n"
        "3 points: Mostly correct with minor gaps vs reference intent. Not a strict "
        "match, but addresses the core of the query.\n"
        "4 points: Correct and clear; paraphrase of the reference is OK. Addresses the "
        "query in a logical manner without unnecessary noise.\n"
        "5 points: Fully correct, precise on critical entities, clear and efficient. "
        "May differ in phrasing while aligning with the reference intent.",
        "Below are the question, the model’s response, and the reference:",
        _block("Question", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        _block("Reference", kw.get("reference")),
        OUTPUT_SCORE,
    )


def _cs_open(**kw: Any) -> str:
    return _join_parts(
        _TEXT_SCENE,
        "This is CODE-SWITCHING spoken QA with no single gold phrasing. "
        "The user speech may mix languages.",
        "[Focus]\n"
        "- Understand the mixed-language question and answer usefully "
        "(URO open dimensions: relevance, accuracy, completeness, conciseness).\n"
        "- Code-switching in the RESPONSE is NOT required unless the question demands it.\n"
        "- Be strict on critical entities/numbers/names grounded in the audio/transcript.\n"
        "- Do NOT require a single gold phrasing.",
        "Please evaluate the response on a scale of 1 to 5:\n"
        "1 point: Largely irrelevant, incorrect, or fails to address the mixed-speech "
        "query; may be off-topic or invent unsupported details.\n"
        "2 points: Somewhat relevant but incomplete or inaccurate; key detail wrong/"
        "missing, or much extraneous information.\n"
        "3 points: Relevant and mostly accurate, but wordy, missing secondary details, "
        "or slightly unclear on mixed-speech content.\n"
        "4 points: Relevant, accurate, and concise; clearly answers the query without "
        "unnecessary elaboration.\n"
        "5 points: Exceptionally on-point, accurate, and efficient; precise on critical "
        "entities and directly addresses the query.",
        "Below are the question and the model’s response:",
        _block("Question", kw.get("question")),
        _block("AudioTranscript", kw.get("audio_transcript") if kw.get("template") == "text" else ""),
        _block("Response", kw.get("pred")),
        OUTPUT_SCORE,
    )


def describe_supported() -> Mapping[str, Any]:
    return {
        "capabilities": ["qa", "audio_reasoning", "speech_translation", "code_switch"],
        "rubrics": {
            "qa": ["open", "semi-open", "binary"],
            "audio_reasoning": ["open", "semi-open"],
            "speech_translation": ["translation"],
            "code_switch": ["binary", "semi-open", "open"],
        },
        "templates": ["text", "multimodal"],
        "norm": "URO-aligned scales/principles; text-eval preamble; task Focus for AR/ST/CS",
    }
