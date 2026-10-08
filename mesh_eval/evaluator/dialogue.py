"""Multi-turn dialogue judges aligned with each benchmark's official protocol.

Protocols (``protocol`` constructor argument):

``vocalbench_multi_round``
    VocalBench / VocalBench-zh ``utils/multi_round.py``: text history + final
    question + reference answer + response; the judge replies Correct/Wrong
    (正确/错误).  Metric ``multi_round_acc``.
``voicebench_mtbench``
    VoiceBench ships MT-Bench data without its own judge, so the FastChat
    ``llm_judge`` single-answer grading prompts are used verbatim
    (``single-v1``/``single-v1-multi-turn`` and the reference-guided ``-math``
    variants for math/reasoning/coding).  Metric ``mtbench_score`` (1-10).
``audioagentbench``
    audio-arena-bench ``judging/llm_judge.py`` in rehydrated (golden prior
    context) mode: official system prompt, per-turn prompt with prompt-visible
    and full KB, golden history and Score Dimensions.  Metrics are the official
    dimension pass flags plus ``turn_pass`` (all scored dimensions pass).
``mtalk_rubric``
    MTalk-Bench ``src/audio_rubric_based.py``: turn hint + user audio +
    response hint + model audio concatenated with 1 s silence, judged once with
    the general and once with the sample-specific rubric (0/1 per criterion).
    Metrics ``rubric_general``, ``rubric_specific``, ``rubric_score``.

``force_close_thinking_tokens`` > 0 asks the vLLM-Omni judge to close an
unfinished thinking block at the token cap and continue for that many tokens;
such judgments are flagged ``judge_forced_close_<rubric>=1``.

Parse failures after ``max_attempts`` omit the numeric metric and report
``judge_parse_failed=1`` so aggregation never averages a fabricated score.
"""

from __future__ import annotations

import ast
import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import reference_object

RESOURCE_DIR = Path(__file__).resolve().parent / "resources"
AAB_RESOURCE_DIR = RESOURCE_DIR / "audio_agent_bench"
PROTOCOLS = (
    "vocalbench_multi_round",
    "voicebench_mtbench",
    "audioagentbench",
    "mtalk_rubric",
)

_THINK_CLOSE = "</think>"
# Must match FORCE_CLOSE_TEXT in audio_evals/lib/qwen3-omni-vllm-omni/main.py.
_FORCE_CLOSE_MARK = "Thinking budget exhausted; giving the final answer now."


def visible_text(raw: Any) -> str:
    """Strip a thinking block; Thinking judges emit ``...</think>answer``."""
    if isinstance(raw, Mapping):
        raw = raw.get("text", raw.get("content", ""))
    text = "" if raw is None else str(raw)
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    return text.strip()


def _prediction_object(pred: Any) -> Any:
    if isinstance(pred, (Mapping, list)):
        return pred
    text = "" if pred is None else str(pred)
    if text.lstrip().startswith(("{", "[")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _responses(pred: Any) -> List[str]:
    value = _prediction_object(pred)
    if isinstance(value, Mapping) and isinstance(value.get("responses"), list):
        return ["" if item is None else str(item) for item in value["responses"]]
    if isinstance(value, Mapping):
        return [str(value.get("text") or value.get("response") or "")]
    return ["" if value is None else str(value)]


def _audio_responses(pred: Any) -> List[Optional[str]]:
    value = _prediction_object(pred)
    items = value.get("audio_responses") if isinstance(value, Mapping) else None
    if not isinstance(items, list):
        return []
    paths: List[Optional[str]] = []
    for item in items:
        if isinstance(item, Mapping):
            item = item.get("path") or item.get("audio") or item.get("uri")
        paths.append(str(item) if item else None)
    return paths


def _truncate(text: str, limit: int) -> str:
    if limit and len(text) > limit:
        return text[:limit] + "\n...[truncated for judge]..."
    return text


def _history_from_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """Text turns before the final user message (system prompt excluded)."""
    turns: List[Dict[str, str]] = []
    body = [m for m in messages if m.get("role") in {"user", "assistant"}]
    for message in body[:-1]:
        texts = [
            str(item.get("value") or "")
            for item in message.get("contents") or []
            if item.get("type") == "text"
        ]
        turns.append({"from": message["role"], "value": "\n".join(texts)})
    return turns


# --------------------------------------------------------------------------
# VocalBench multi-round (utils/multi_round.py, verbatim prompt bodies)
# --------------------------------------------------------------------------


def vocalbench_multi_round_prompt(
    *, question: str, pred: str, reference: str, history: Sequence[Mapping[str, Any]], language: str
) -> str:
    context = ""
    for turn in history:
        if turn.get("from") == "user":
            context = context + f"User: {turn.get('value')}\n"
        elif turn.get("from") == "assistant":
            context = context + f"Model: {turn.get('value')}\n"
    if str(language).lower().startswith("zh"):
        return f'''
你是一个客观、严谨的语音问答评估助手。接下来，我将提供以下四项内容：

历史：用户与模型对话的历史。
问题：一个中文问题，在用户与模型完成历史对话后由用户提出。
正确答案：该问题对应的正确回答。
模型回答：模型针对该问题实际生成的回答。
请你根据以下要求进行分析与评估：

判断“模型回答”所包含的结果与正确答案是否一致。如果模型回答没有给出明确的结果或与问题无关，记为错误。如果模型回答包含了正确答案，记为正确。
注意：请忽略同音错误，如题目中的“小亮”和“晓亮”。如果模型回答为正确答案的同音内容，同样视为正确。
输出格式：正确/错误 [简单解释]
示例：正确 模型给出了明确的答案，并且与正确答案一致。

历史：{context}
问题：{question}
正确答案：{reference}
模型回答：{pred}
  ***
  ''' + "\n（如需推理请放在 <think>...</think> 内；可见回复必须以“正确”或“错误”开头。）"
    return f'''
You are an objective question-answering evaluation assistant. Next, I will provide the following four components:
History: The history of the user's dialogue with the model.
Question: A question from the user after their historical dialogue.
Answer: The correct answer to the question.
Response: The response generated by the model for the question.

Please analyze and evaluate according to the following requirements:
Determine whether the result contained in the "Response" is consistent with the "Answer". 

Output Format: Correct/Wrong [Simple Explanation]
Example: Correct. The model provided a clear answer, and it is consistent with the correct answer.

History: {context}
Question: {question}
Answer: {reference}
Response: {pred} 
  ''' + "\n(Keep any reasoning inside <think>...</think>. Your visible reply must begin with Correct or Wrong.)"


def parse_correct_wrong(raw: Any) -> Optional[int]:
    text = visible_text(raw)
    lines = [text] + [line for line in text.splitlines() if line.strip()]
    for line in lines:
        head = re.sub(r"^[^A-Za-z\u4e00-\u9fff]+", "", line)
        if head.startswith(("Correct", "正确")):
            return 1
        if head.startswith(("Wrong", "错误")):
            return 0
    return None


# --------------------------------------------------------------------------
# MT-Bench (FastChat llm_judge/data/judge_prompts.jsonl, verbatim)
# --------------------------------------------------------------------------

MTBENCH_NEED_REF_CATS = {"math", "reasoning", "coding", "arena-hard-200"}
_MTBENCH_RATING_RULE = (
    'After providing your explanation, you must rate the response on a scale of 1 to 10 '
    'by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".'
)
MTBENCH_SINGLE = (
    "You are a helpful assistant.",
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your evaluation should "
    "consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level "
    "of detail of the response. Begin your evaluation by providing a short explanation. Be as "
    "objective as possible. " + _MTBENCH_RATING_RULE + "\n\n[Question]\n{question}\n\n"
    "[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]",
)
MTBENCH_SINGLE_MATH = (
    "You are a helpful assistant.",
    "[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response "
    "provided by an AI assistant to the user question displayed below. Your evaluation should "
    "consider correctness and helpfulness. You will be given a reference answer and the "
    "assistant's answer. Begin your evaluation by comparing the assistant's answer with the "
    "reference answer. Identify and correct any mistakes. Be as objective as possible. "
    + _MTBENCH_RATING_RULE + "\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n"
    "{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n"
    "{answer}\n[The End of Assistant's Answer]",
)
MTBENCH_MULTI = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an "
    "AI assistant to the user question displayed below. Your evaluation should consider factors "
    "such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the "
    "response. You evaluation should focus on the assistant's answer to the second user question. "
    "Begin your evaluation by providing a short explanation. Be as objective as possible. "
    + _MTBENCH_RATING_RULE + "\n\n",
    "<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n"
    "### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n"
    "<|The End of Assistant A's Conversation with User|>",
)
MTBENCH_MULTI_MATH = (
    "Please act as an impartial judge and evaluate the quality of the response provided by an "
    "AI assistant to the user question. Your evaluation should consider correctness and "
    "helpfulness. You will be given a reference answer and the assistant's answer. You evaluation "
    "should focus on the assistant's answer to the second question. Begin your evaluation by "
    "comparing the assistant's answer with the reference answer. Identify and correct any "
    "mistakes. Be as objective as possible. " + _MTBENCH_RATING_RULE + "\n\n",
    "<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n"
    "{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n"
    "<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>"
    "\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n"
    "### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>",
)
_MTBENCH_SUFFIX = (
    "\n\n(Keep any private reasoning inside <think>...</think> and brief. The visible reply must "
    "contain the rating in the exact format [[N]].)"
)


def mtbench_turn_prompt(
    *, turn: int, questions: Sequence[str], answers: Sequence[str], refs: Sequence[str]
) -> Tuple[str, str]:
    """Return (system, user) for 1-based ``turn`` (MT-Bench has two turns)."""
    use_ref = len(refs) >= turn and all(str(item).strip() for item in refs[:turn])
    if turn == 1:
        system, template = MTBENCH_SINGLE_MATH if use_ref else MTBENCH_SINGLE
        user = template.format(
            question=questions[0], answer=answers[0], ref_answer_1=refs[0] if use_ref else ""
        )
    else:
        system, template = MTBENCH_MULTI_MATH if use_ref else MTBENCH_MULTI
        user = template.format(
            question_1=questions[0],
            question_2=questions[1],
            answer_1=answers[0],
            answer_2=answers[1],
            ref_answer_1=refs[0] if use_ref else "",
            ref_answer_2=refs[1] if use_ref else "",
        )
    return system, user + _MTBENCH_SUFFIX


def parse_mtbench_rating(raw: Any) -> Optional[float]:
    text = visible_text(raw)
    match = re.search(r"\[\[(\d+\.?\d*)\]\]", text) or re.search(r"\[(\d+\.?\d*)\]", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if 1 <= value <= 10 else None


# --------------------------------------------------------------------------
# AudioAgentBench (audio-arena-bench llm_judge.py, rehydrated mode)
# --------------------------------------------------------------------------

AAB_CORE_DIMENSIONS = ["kb_grounding", "instruction_following", "tool_use_correct"]
AAB_BOOL_DIMENSIONS = (
    "tool_use_correct",
    "instruction_following",
    "kb_grounding",
    "ambiguity_handling",
    "state_tracking",
)


@lru_cache(maxsize=None)
def _resource_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def aab_judge_system_prompt() -> str:
    override = _resource_text(str(AAB_RESOURCE_DIR / "rehydrated_override.txt"))
    system = _resource_text(str(AAB_RESOURCE_DIR / "judge_system_prompt.txt"))
    return f"{override}\n\n{system}"


@lru_cache(maxsize=None)
def _aab_turns(metadata_path: str, speaker: str) -> Tuple[Dict[str, Any], ...]:
    turns: List[Dict[str, Any]] = []
    with open(metadata_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if str(item.get("speaker") or "unknown") == speaker:
                turns.append(item)
    turns.sort(key=lambda item: int(item.get("turn_id") or 0))
    return tuple(turns)


def _aab_calls(value: Any) -> Any:
    if value in (None, ""):
        return None
    return json.loads(value) if isinstance(value, str) else value


def _aab_categories(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def aab_turn_prompt(
    *,
    turn_id: int,
    history: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
    assistant_text: str,
    score_dimensions: Sequence[str],
    kb_text: str,
    prompt_visible_kb_text: str = "",
) -> str:
    """Port of ``format_rehydrated_turns_for_judge`` + ``build_judge_user_prompt``."""
    lines: List[str] = []
    if prompt_visible_kb_text:
        lines += [
            "# Prompt-Visible Knowledge Base (What the Assistant Actually Saw)",
            "",
            prompt_visible_kb_text.strip(),
            "",
            "---",
            "",
        ]
    if kb_text:
        lines += [
            "# Full Benchmark Knowledge Base (Oracle / Tool-Only Facts)",
            "",
            kb_text.strip(),
            "",
            "---",
            "",
        ]
    lines += ["# Expected Function Calls Summary", ""]
    fc = _aab_calls(target.get("required_function_call"))
    if fc:
        if isinstance(fc, list):
            calls_str = " → ".join(f"{c['name']}({json.dumps(c['args'])})" for c in fc)
            lines.append(f"- Turn {turn_id}: [MULTI-STEP] {calls_str}")
        else:
            lines.append(f"- Turn {turn_id}: {fc['name']}({json.dumps(fc['args'])})")
    lines += ["", "---", ""]
    lines += [
        "# Hydrated Golden Conversation History",
        "",
        "This section is the golden prior context used to hydrate rehydrated benchmark turns.",
        "When judging target turn N, use only Golden Turns with index less than N as prior conversation state.",
        "Do NOT use actual transcript content from one target turn block as prior state for another target turn.",
        "",
    ]
    for index, expected in enumerate(history):
        lines.append(f"### Golden Turn {index}")
        lines.append(f"**User**: {expected.get('input_text', '')}")
        calls = _aab_calls(expected.get("required_function_call"))
        responses = _aab_calls(expected.get("function_call_response"))
        if calls is not None:
            calls = calls if isinstance(calls, list) else [calls]
            responses = (
                responses if isinstance(responses, list) else [responses] if responses is not None else []
            )
            for idx, call in enumerate(calls):
                lines.append(
                    f"  [Hydrated tool call: {call['name']}({json.dumps(call.get('args', {}))})]"
                )
                if idx < len(responses):
                    lines.append(f"  [Hydrated tool result: {json.dumps(responses[idx])}]")
        lines.append(f"**Assistant (Golden)**: {expected.get('golden_text', '')}")
        lines.append("")
    lines += ["---", "", "# Target Turn Evaluations", ""]
    lines.append(f"## Turn {turn_id}")
    lines.append("**Turn-Taking**: OK (no audio analysis)")
    if turn_id == 0:
        lines.append("**Hydrated Prior Context**: none (this is the first turn)")
    else:
        lines.append(
            f"**Hydrated Prior Context**: Golden Turns 0-{turn_id - 1} from the shared hydrated history above"
        )
    lines.append(f"**User**: {target.get('input_text', '')}")
    lines.append(f"**Assistant**: {assistant_text}")
    lines.append("")
    golden = target.get("golden_text", "")
    if golden:
        lines += [f"**Golden Response**: {golden}", ""]
    categories = _aab_categories(target.get("categories"))
    if categories:
        lines.append(f"**Category**: {', '.join(categories)}")
        if target.get("subcategory"):
            lines.append(f"**Subcategory**: {target['subcategory']}")
        lines.append(f"**Score Dimensions**: {', '.join(score_dimensions)}")
        lines.append("")
    lines.append(f"**Expected Function**: {json.dumps(fc)}" if fc else "**Expected Function**: none")
    lines.append("**Actual Functions**: none")
    lines.append("**Actual Function Results**: none")
    lines += ["", "---", ""]
    formatted = "\n".join(lines)
    instructions = f"""Please evaluate each turn independently:
1. Analyze each turn against its golden expectation
2. Use the "Hydrated Golden Conversation History" section as the ONLY source of prior-turn state
3. Do NOT use the actual transcript content from one target turn block as prior state for any other turn
4. Output the final JSON with judgments for ALL 1 turns

CRITICAL: Your final_judgments array MUST contain exactly 1 entries, one for each of these exact turn IDs:
[{turn_id}]

Do NOT renumber turns. Use the actual turn IDs shown above, even if they are non-contiguous or do not start at 0.

Remember:
- This is a rehydrated run. Each turn stands on its own with hydrated golden prior context
- For target turn N, only Golden Turns with index < N count as prior state. Golden Turns with index >= N are future turns for that target and must be ignored
- Do NOT mark tool_use_correct=TRUE because the expected call happened in another actual transcript turn
- Do NOT retroactively credit a turn because a function was called later in a different actual transcript turn
- **Penalty absorption rule**: When a tool call is missed due to a more specific root cause within the same turn, the penalty lands on the specific dimension (ambiguity_handling or state_tracking) if it's in Score Dimensions. If the specific dimension is NOT in Score Dimensions, fall back to tool_use_correct=FALSE. The penalty must always land somewhere.
- Missing/wrong tool call (not over-clarification or state failure) → tool_use_correct=FALSE only; do not fail instruction_following
- Words contradict actions (e.g. says "I'll wait" but calls in same turn) → tool_use_correct=FALSE and instruction_following=FALSE
- Be generous with kb_grounding unless there's a clear factual error
- Empty assistant_text with a valid tool call is still a valid turn - evaluate the tool call
- Set realignment_notes to "Cross-turn realignment disabled for rehydrated run."
"""
    suffix = (
        "\n(Keep private reasoning inside <think>...</think> and brief. After it, output only "
        "the JSON object; phase1_analysis may be an empty list.)"
    )
    return f"{formatted}\n\n{instructions}{suffix}"


def parse_aab_judgment(raw: Any, turn_id: int) -> Optional[Dict[str, Any]]:
    text = visible_text(raw)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    judgments = payload.get("final_judgments") if isinstance(payload, Mapping) else None
    if not isinstance(judgments, list) or not judgments:
        return None
    chosen = next(
        (item for item in judgments if isinstance(item, Mapping) and item.get("turn") == turn_id),
        judgments[0] if len(judgments) == 1 else None,
    )
    if not isinstance(chosen, Mapping):
        return None
    for key in ("instruction_following", "kb_grounding"):
        if not isinstance(chosen.get(key), bool):
            return None
    return dict(chosen)


# --------------------------------------------------------------------------
# MTalk rubric (src/audio_rubric_based.py)
# --------------------------------------------------------------------------


def mtalk_criteria_ids(rubric_prompt: str) -> List[str]:
    """Criterion ids from the criteria section (the JSON example may list extras)."""
    section = re.split(r"#+\s*Scoring Instructions", str(rubric_prompt or ""), maxsplit=1)[0]
    ids: List[str] = []
    for match in re.finditer(r"^\s*(?:-\s*)?(criteria_[a-z]+_\d+)\s*:", section, re.M):
        if match.group(1) not in ids:
            ids.append(match.group(1))
    return ids


def parse_mtalk_scores(raw: Any, criteria: Sequence[str]) -> Optional[Dict[str, int]]:
    text = visible_text(raw)
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        scores: Dict[str, int] = {}
        for key in criteria:
            value = payload.get(key)
            if isinstance(value, bool):
                value = int(value)
            if value not in (0, 1):
                break
            scores[key] = int(value)
        else:
            return scores
    return None


def _mtalk_suffix(criteria: Sequence[str], strict: bool = False) -> str:
    head = (
        "\n\nThe attached audio is the whole dialogue: for each turn a spoken turn marker, the user's "
        "audio, a spoken 'model response' marker, then the model's spoken response."
        f"\nScore exactly these criterion IDs: {', '.join(criteria)}."
    )
    if strict:
        from audio_evals.evaluator.llm_judge_prompts import OUTPUT_JSON_OBJECT

        return f"{head}\n\n{OUTPUT_JSON_OBJECT}"
    return head + (
        "\n(Keep private reasoning inside <think>...</think> and brief. After it, output only one "
        "JSON object mapping each criterion ID to 1 or 0.)"
    )


def concat_dialogue_audio(segments: Sequence[str], output_path: Path, silence_path: Optional[str]) -> Path:
    """Official concatenation: every segment followed by 1 s silence, first segment's format."""
    import numpy as np
    import soundfile as sf

    def load(path: str, rate: Optional[int]) -> Tuple[Any, int]:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        data = data.mean(axis=1)
        if rate is not None and sr != rate:
            from math import gcd

            from scipy.signal import resample_poly

            factor = gcd(int(rate), int(sr))
            data = resample_poly(data, rate // factor, sr // factor).astype("float32")
            sr = rate
        return data, sr

    target_rate: Optional[int] = None
    pieces = []
    for path in segments:
        data, sr = load(path, target_rate)
        target_rate = target_rate or sr
        pieces.append(data)
        if silence_path and Path(silence_path).is_file():
            pieces.append(load(silence_path, target_rate)[0])
        else:
            pieces.append(np.zeros(int(target_rate), dtype="float32"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.concatenate(pieces), int(target_rate or 16000))
    return output_path


# --------------------------------------------------------------------------
# Evaluator
# --------------------------------------------------------------------------


class MultiTurnDialogueEvaluator(Evaluator):
    def __init__(
        self,
        protocol: str,
        model_name: str = "qwen3-omni-thinking",
        judge: Optional[Callable[[List[Dict[str, Any]]], Any]] = None,
        max_attempts: int = 3,
        max_response_chars: int = 6000,
        work_dir: str = "",
        force_close_thinking_tokens: int = 0,
        strict_output_rules: bool = False,
    ):
        if protocol not in PROTOCOLS:
            raise ValueError(f"unsupported multi-turn protocol: {protocol}")
        self.protocol = protocol
        self.model_name = model_name
        self.judge = judge
        self.max_attempts = max(1, int(max_attempts))
        self.max_response_chars = int(max_response_chars)
        self.work_dir = work_dir
        self.force_close_thinking_tokens = int(force_close_thinking_tokens or 0)
        self.strict_output_rules = bool(strict_output_rules)

    # -- judge access -------------------------------------------------------

    def _judge(self, messages: List[Dict[str, Any]]) -> str:
        if self.judge is None:
            from audio_evals.evaluator.voice_bench import get_judge_model

            model = get_judge_model(self.model_name)
            self.judge = lambda payload, **options: model.inference(payload, **options)
        if self.force_close_thinking_tokens > 0:
            raw = self.judge(messages, force_close_thinking_tokens=self.force_close_thinking_tokens)
        else:
            raw = self.judge(messages)
        if isinstance(raw, Mapping):
            raw = raw.get("text", raw)
        return "" if raw is None else str(raw)

    def _ask(self, messages: List[Dict[str, Any]], parse: Callable[[str], Any]) -> Tuple[Any, str, int]:
        raw = ""
        for attempt in range(1, self.max_attempts + 1):
            raw = self._judge(messages)
            parsed = parse(raw)
            if parsed is not None:
                return parsed, raw, attempt
        return None, raw, self.max_attempts

    @staticmethod
    def _text_messages(user: str, system: str = "") -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "contents": [{"type": "text", "value": system}]})
        messages.append({"role": "user", "contents": [{"type": "text", "value": user}]})
        return messages

    @staticmethod
    def _failed(raw: str, **extra: Any) -> Dict[str, Any]:
        return {
            "status": "judge_parse_failed",
            "judge_parse_failed": 1,
            "judge_raw": raw[-2000:],
            **extra,
        }

    @staticmethod
    def _not_evaluated(reason: str) -> Dict[str, Any]:
        return {"status": "not_evaluated", "not_evaluated": 1, "not_evaluated_reason": reason}

    # -- protocols ----------------------------------------------------------

    def _vocalbench(self, pred: Any, reference: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        history = reference.get("context")
        if isinstance(history, str):
            try:
                history = ast.literal_eval(history)
            except (ValueError, SyntaxError):
                history = None
        if not isinstance(history, list):
            history = _history_from_messages((kwargs.get("input") or {}).get("messages") or [])
        response = _truncate(_responses(pred)[-1], self.max_response_chars)
        prompt = vocalbench_multi_round_prompt(
            question=str(reference.get("question") or ""),
            pred=response,
            reference=str(reference.get("answer") or ""),
            history=history,
            language=str(reference.get("language") or kwargs.get("language") or "en"),
        )
        label, raw, attempts = self._ask(self._text_messages(prompt), parse_correct_wrong)
        if label is None:
            return self._failed(raw)
        return {
            "status": "evaluated",
            "multi_round_acc": label,
            "judge_explain": visible_text(raw)[:1000],
            "judge_attempts": attempts,
        }

    def _mtbench(self, pred: Any, reference: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        questions = [str(item) for item in reference.get("turns") or kwargs.get("turns") or []]
        answers = [_truncate(item, self.max_response_chars) for item in _responses(pred)]
        if len(questions) != 2 or len(answers) != 2:
            return self._not_evaluated(
                f"MT-Bench needs 2 turns and 2 responses, got {len(questions)}/{len(answers)}"
            )
        category = str(reference.get("category") or "").lower()
        refs = [str(item) for item in reference.get("reference_answers") or []]
        if category not in MTBENCH_NEED_REF_CATS:
            refs = []
        scores: List[float] = []
        for turn in (1, 2):
            system, user = mtbench_turn_prompt(turn=turn, questions=questions, answers=answers, refs=refs)
            score, raw, _ = self._ask(self._text_messages(user, system), parse_mtbench_rating)
            if score is None:
                return self._failed(raw, failed_turn=turn)
            scores.append(score)
        return {
            "status": "evaluated",
            "mtbench_score": sum(scores) / len(scores),
            "mtbench_turn1": scores[0],
            "mtbench_turn2": scores[1],
            "reference_guided": int(bool(refs)),
        }

    def _audioagentbench(self, pred: Any, reference: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        metadata_path = str(reference.get("metadata_path") or "")
        speaker = str(reference.get("speaker") or "")
        turn_id = int(reference.get("turn_id") or 0)
        turns = _aab_turns(metadata_path, speaker)
        target = next((item for item in turns if int(item.get("turn_id") or 0) == turn_id), None)
        if target is None:
            raise ValueError(f"AudioAgentBench turn {turn_id} not found in {metadata_path}")
        history = [item for item in turns if int(item.get("turn_id") or 0) < turn_id]
        dims = list(reference.get("score_dimensions") or AAB_CORE_DIMENSIONS)
        visible_kb_path = str(reference.get("prompt_visible_kb_path") or "")
        prompt = aab_turn_prompt(
            turn_id=turn_id,
            history=history,
            target=target,
            assistant_text=_truncate(_responses(pred)[-1], self.max_response_chars),
            score_dimensions=dims,
            kb_text=_resource_text(str(reference["kb_path"])),
            prompt_visible_kb_text=_resource_text(visible_kb_path) if visible_kb_path else "",
        )
        judgment, raw, attempts = self._ask(
            self._text_messages(prompt, aab_judge_system_prompt()),
            lambda text: parse_aab_judgment(text, turn_id),
        )
        if judgment is None:
            return self._failed(raw)
        result: Dict[str, Any] = {"status": "evaluated", "judge_attempts": attempts}
        passed = True
        # No tool call is expected on dialogue turns and the offline SUT cannot
        # call tools, so the official _tool_pass treats a null score as a pass.
        tool = judgment.get("tool_use_correct")
        result["tool_use_correct"] = 1 if tool is None or tool is True else 0
        passed &= bool(result["tool_use_correct"])
        for key in ("instruction_following", "kb_grounding"):
            result[key] = int(bool(judgment.get(key)))
            passed &= bool(result[key])
        for key in ("state_tracking", "ambiguity_handling"):
            value = judgment.get(key)
            if key in dims and isinstance(value, bool):
                result[key] = int(value)
                passed &= value
        result["turn_pass"] = int(passed)
        result["judge_reasoning"] = str(judgment.get("reasoning") or "")[:1500]
        return result

    def _mtalk(self, pred: Any, reference: Mapping[str, Any], kwargs: Mapping[str, Any]) -> Dict[str, Any]:
        user_audio = [str(item) for item in reference.get("user_audio") or []]
        model_audio = _audio_responses(pred)
        if not user_audio:
            input_obj = kwargs.get("input") or {}
            user_audio = [
                str(item.get("uri") or item.get("path") or "")
                for item in input_obj.get("audio") or []
                if isinstance(item, Mapping)
            ]
        missing = [
            index
            for index in range(len(user_audio))
            if index >= len(model_audio) or not model_audio[index] or not Path(model_audio[index]).is_file()
        ]
        if not user_audio or missing:
            return self._not_evaluated(
                f"speech-to-speech rubric needs one generated audio per turn; missing turns {missing}"
            )
        hint_dir = Path(str(reference.get("hint_audio_dir") or ""))
        segments: List[str] = []
        for index, (user, model) in enumerate(zip(user_audio, model_audio), start=1):
            turn_hint = hint_dir / f"turn_{index}.wav"
            if turn_hint.is_file():
                segments.append(str(turn_hint))
            segments.append(user)
            response_hint = hint_dir / "models_response.wav"
            if response_hint.is_file():
                segments.append(str(response_hint))
            segments.append(str(model))
        base = Path(self.work_dir or os.environ.get("NATIVE_OUTPUT_DIR") or tempfile.gettempdir())
        digest = sha1("|".join(segments).encode("utf-8")).hexdigest()[:16]
        dialogue_wav = concat_dialogue_audio(
            segments,
            base / "judge" / "mtalk_dialogue_audio" / f"{digest}.wav",
            str(hint_dir / "silence_1s.wav"),
        )
        result: Dict[str, Any] = {"status": "evaluated", "dialogue_audio": str(dialogue_wav)}
        all_scores: List[int] = []
        for kind in ("general", "specific"):
            rubric = str(reference.get(f"rubric_{kind}") or "")
            criteria = mtalk_criteria_ids(rubric)
            if not criteria:
                raise ValueError(f"MTalk {kind} rubric has no criteria ids")
            messages = [
                {
                    "role": "user",
                    "contents": [
                        {"type": "text", "value": rubric + _mtalk_suffix(criteria, self.strict_output_rules)},
                        {"type": "audio", "value": str(dialogue_wav)},
                    ],
                }
            ]
            scores, raw, _ = self._ask(messages, lambda text: parse_mtalk_scores(text, criteria))
            forced = int(_FORCE_CLOSE_MARK in raw)
            if scores is None:
                return self._failed(
                    raw,
                    failed_rubric=kind,
                    dialogue_audio=str(dialogue_wav),
                    **{f"judge_forced_close_{kind}": forced},
                )
            result[f"judge_forced_close_{kind}"] = forced
            result[f"rubric_{kind}"] = sum(scores.values()) / len(scores)
            result[f"criteria_{kind}"] = json.dumps(scores, sort_keys=True)
            all_scores.extend(scores.values())
        result["rubric_score"] = sum(all_scores) / len(all_scores)
        return result

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = label if isinstance(label, Mapping) else reference_object(label, kwargs)
        handler = {
            "vocalbench_multi_round": self._vocalbench,
            "voicebench_mtbench": self._mtbench,
            "audioagentbench": self._audioagentbench,
            "mtalk_rubric": self._mtalk,
        }[self.protocol]
        result = handler(pred, reference, kwargs)
        result.setdefault("judge_protocol", self.protocol)
        result.setdefault("judge_model", self.model_name)
        return result
