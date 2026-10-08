"""Multi-turn dialogue judges: official prompt shapes and fail-closed parsing."""

import json
import re

import numpy as np
import pytest
import soundfile as sf

from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.dataset.external_benchmarks import (
    _aab_rehydrated_turn,
    audio_agent_score_dimensions,
)
from mesh_eval.evaluator.dialogue import (
    MultiTurnDialogueEvaluator,
    aab_judge_system_prompt,
    mtalk_criteria_ids,
    parse_correct_wrong,
    parse_mtbench_rating,
)


class RecordingJudge:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return reply(messages) if callable(reply) else reply


def _texts(messages):
    return "\n".join(
        item["value"] for message in messages for item in message["contents"] if item["type"] == "text"
    )


def test_vocalbench_multi_round_uses_official_prompt_and_first_token_label():
    judge = RecordingJudge("<think>compare</think>**Correct.** consistent with the answer")
    evaluator = MultiTurnDialogueEvaluator("vocalbench_multi_round", judge=judge)
    reference = {
        "answer": "E is the highest.",
        "question": "Who is the highest now?",
        "context": [
            {"from": "user", "value": "A is taller than B."},
            {"from": "assistant", "value": "A is the tallest."},
        ],
        "language": "en",
    }
    score = evaluator("E is the tallest.", reference, reference_obj=reference)

    assert score["multi_round_acc"] == 1
    prompt = _texts(judge.calls[0])
    assert "You are an objective question-answering evaluation assistant." in prompt
    assert "History: User: A is taller than B.\nModel: A is the tallest.\n" in prompt
    assert "Answer: E is the highest." in prompt

    zh_judge = RecordingJudge("<think>对比</think>错误 模型没有给出阿紫")
    zh = MultiTurnDialogueEvaluator("vocalbench_multi_round", judge=zh_judge)(
        "小明最高", {**reference, "language": "zh"}
    )
    assert zh["multi_round_acc"] == 0
    assert "你是一个客观、严谨的语音问答评估助手" in _texts(zh_judge.calls[0])


def test_unparseable_judge_retries_then_omits_numeric_metric():
    judge = RecordingJudge("<think>never ends")
    evaluator = MultiTurnDialogueEvaluator("vocalbench_multi_round", judge=judge, max_attempts=2)
    score = evaluator("x", {"answer": "y", "question": "q", "context": []})

    assert len(judge.calls) == 2
    assert score["status"] == "judge_parse_failed"
    assert "multi_round_acc" not in score
    assert "overall/multi_round_acc" not in MeshAgg(group_by=[], min_slice_size=1)([score])
    assert parse_correct_wrong("Final answer:\nWrong. mismatch") == 0


def test_mtbench_uses_fastchat_templates_and_reference_only_for_need_ref_categories():
    judge = RecordingJudge("<think>..</think>Solid. Rating: [[8]]", "Weak. Rating: [[4]]")
    evaluator = MultiTurnDialogueEvaluator("voicebench_mtbench", judge=judge)
    reference = {
        "turns": ["Q1?", "Q2?"],
        "reference_answers": ["R1", "R2"],
        "category": "reasoning",
    }
    score = evaluator(json.dumps({"responses": ["A1", "A2"]}), reference)

    assert score["mtbench_score"] == pytest.approx(6.0)
    assert (score["mtbench_turn1"], score["mtbench_turn2"]) == (8.0, 4.0)
    assert score["reference_guided"] == 1
    turn1, turn2 = (_texts(call) for call in judge.calls)
    assert "[The Start of Reference Answer]\nR1\n[The End of Reference Answer]" in turn1
    assert "focus on the assistant's answer to the second question" in turn2
    assert "### Reference answer:\nR2" in turn2

    stem_judge = RecordingJudge("Rating: [[7]]")
    stem = MultiTurnDialogueEvaluator("voicebench_mtbench", judge=stem_judge)(
        json.dumps({"responses": ["A1", "A2"]}), {**reference, "category": "stem"}
    )
    assert stem["reference_guided"] == 0
    assert "Reference" not in _texts(stem_judge.calls[0])

    short = MultiTurnDialogueEvaluator("voicebench_mtbench", judge=judge)(
        json.dumps({"responses": ["only one"]}), reference
    )
    assert short["status"] == "not_evaluated"
    assert parse_mtbench_rating("score [11]") is None


def _write_aab(tmp_path):
    bench = tmp_path / "grocery-bench"
    (bench / "benchmark").mkdir(parents=True)
    (bench / "benchmark" / "knowledge_base.txt").write_text("KB: eggs cost $5.", encoding="utf-8")
    turns = [
        {"turn_id": 0, "speaker": "p1", "input_text": "Hi, I'm Ann.", "golden_text": "Hi Ann!",
         "required_function_call": None, "function_call_response": None, "categories": "basic_qa"},
        {"turn_id": 1, "speaker": "p1", "input_text": "Order eggs.", "golden_text": "Ordered.",
         "required_function_call": json.dumps([{"name": "process_order", "args": {"item": "eggs"}}]),
         "function_call_response": json.dumps([{"status": "ok", "order_id": "O-1"}]),
         "categories": "tool_use"},
        {"turn_id": 2, "speaker": "p1", "input_text": "What's my name and order id?",
         "golden_text": "Ann, order O-1.", "required_function_call": None,
         "function_call_response": None, "categories": "long_range_memory"},
    ]
    metadata = bench / "metadata.jsonl"
    metadata.write_text("\n".join(json.dumps(item) for item in turns), encoding="utf-8")
    return bench, metadata, turns


def test_audioagentbench_rehydrated_prompt_and_official_pass_rule(tmp_path):
    bench, metadata, turns = _write_aab(tmp_path)
    dims = audio_agent_score_dimensions("grocery-bench", "long_range_memory")
    assert dims == ["kb_grounding", "instruction_following", "tool_use_correct", "state_tracking"]
    assert audio_agent_score_dimensions("grocery-bench", "implicit_correction")[-1] == "tool_use_correct"

    history = _aab_rehydrated_turn(turns[1])
    assert history[0]["contents"][0]["value"] == "Order eggs."
    assert '[Tool call: process_order({"item": "eggs"})]' in history[1]["contents"][0]["value"]
    assert '[Tool result: {"status": "ok", "order_id": "O-1"}]' in history[1]["contents"][0]["value"]

    reply = (
        '<think>ok</think>{"phase1_analysis": [], "final_judgments": [{"turn": 2, "reasoning": "recalled",'
        ' "turn_taking": true, "tool_use_correct": null, "instruction_following": true,'
        ' "kb_grounding": true, "ambiguity_handling": null, "state_tracking": false}]}'
    )
    judge = RecordingJudge(reply)
    reference = {
        "metadata_path": str(metadata),
        "speaker": "p1",
        "turn_id": 2,
        "score_dimensions": dims,
        "kb_path": str(bench / "benchmark" / "knowledge_base.txt"),
    }
    score = MultiTurnDialogueEvaluator("audioagentbench", judge=judge)("Ann.", reference)

    assert score["tool_use_correct"] == 1
    assert score["instruction_following"] == 1
    assert score["state_tracking"] == 0
    assert "ambiguity_handling" not in score
    assert score["turn_pass"] == 0
    system = judge.calls[0][0]["contents"][0]["value"]
    assert system == aab_judge_system_prompt()
    assert system.startswith("# Rehydrated Run Override")
    prompt = judge.calls[0][1]["contents"][0]["value"]
    assert "# Full Benchmark Knowledge Base (Oracle / Tool-Only Facts)\n\nKB: eggs cost $5." in prompt
    assert "### Golden Turn 1\n**User**: Order eggs." in prompt
    assert '[Hydrated tool result: {"status": "ok", "order_id": "O-1"}]' in prompt
    assert "**Score Dimensions**: kb_grounding, instruction_following, tool_use_correct, state_tracking" in prompt
    assert "Golden Turn 2" not in prompt


def _tone(path, seconds, rate):
    sf.write(str(path), np.zeros(int(seconds * rate), dtype="float32"), rate)
    return str(path)


def test_mtalk_rubric_scores_both_rubrics_on_concatenated_dialogue(tmp_path):
    specific = (
        "## Sample-Level Evaluation Criteria:\n- criteria_d_1: a\n- criteria_d_2: b\n\n"
        '### Scoring Instructions\n```json\n{"criteria_d_1": 1, "criteria_d_2": 0, "criteria_d_3": 1}\n```'
    )
    general = "## Evaluation Criteria:\ncriteria_s_1: x\ncriteria_m_1: y\n\n### Scoring Instructions\n"
    assert mtalk_criteria_ids(specific) == ["criteria_d_1", "criteria_d_2"]

    hints = tmp_path / "asset"
    hints.mkdir()
    for name in ("turn_1", "turn_2", "models_response"):
        _tone(hints / f"{name}.wav", 0.5, 24000)
    _tone(hints / "silence_1s.wav", 1.0, 16000)
    users = [_tone(tmp_path / f"u{i}.wav", 2.0, 24000) for i in (1, 2)]
    models = [_tone(tmp_path / f"m{i}.wav", 1.0, 16000) for i in (1, 2)]

    def reply(messages):
        ids = re.findall(r"Score exactly these criterion IDs: (.*)\.", messages[0]["contents"][0]["value"])[0]
        return "<think>ok</think>" + json.dumps({key: 1 for key in ids.split(", ")})

    judge = RecordingJudge(reply)
    reference = {
        "user_audio": users,
        "rubric_general": general,
        "rubric_specific": specific,
        "hint_audio_dir": str(hints),
    }
    evaluator = MultiTurnDialogueEvaluator("mtalk_rubric", judge=judge, work_dir=str(tmp_path / "run"))
    score = evaluator(json.dumps({"responses": ["a", "b"], "audio_responses": models}), reference)

    assert score["rubric_general"] == 1.0
    assert score["rubric_specific"] == 1.0
    assert score["rubric_score"] == 1.0
    assert len(judge.calls) == 2
    assert judge.calls[0][0]["contents"][1] == {"type": "audio", "value": score["dialogue_audio"]}
    # 2 turns x (turn hint 0.5 + user 2 + response hint 0.5 + model 1) + 8 x 1 s silence
    assert sf.info(score["dialogue_audio"]).duration == pytest.approx(16.0, abs=0.01)

    partial = evaluator(json.dumps({"responses": ["a", "b"], "audio_responses": models[:1]}), reference)
    assert partial["status"] == "not_evaluated"
    assert "rubric_score" not in partial
    assert score["judge_forced_close_general"] == 0

    options = []

    def forced_judge(messages, **kwargs):
        options.append(kwargs)
        ids = re.findall(r"Score exactly these criterion IDs: (.*)\.", messages[0]["contents"][0]["value"])[0]
        return (
            "<think>Wait, let me check again"
            "\n\nThinking budget exhausted; giving the final answer now.\n</think>\n\n"
            + json.dumps({key: 0 for key in ids.split(", ")})
        )

    forced = MultiTurnDialogueEvaluator(
        "mtalk_rubric", judge=forced_judge, work_dir=str(tmp_path / "run"), force_close_thinking_tokens=512
    )(json.dumps({"responses": ["a", "b"], "audio_responses": models}), reference)
    assert options == [{"force_close_thinking_tokens": 512}] * 2
    assert forced["rubric_score"] == 0.0
    assert forced["judge_forced_close_general"] == 1
    assert forced["judge_forced_close_specific"] == 1
