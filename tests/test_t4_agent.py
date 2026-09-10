"""Contract and integration tests for the T4 spoken-agent audit."""

import csv
import json
import os
import wave
from pathlib import Path

import pytest

from audio_evals.registry import Registry
from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.core.benchmark_map import load_benchmark_map
from mesh_eval.dataset.external_benchmarks import (
    AudioAgentBenchDataset,
    FluentSpeechCommandsT4Dataset,
    SLURPT4Dataset,
)
from mesh_eval.evaluator.agent import (
    AgentRubricEvaluator,
    ClarificationEvaluator,
    IFEvalAdapter,
    IHBenchEvaluator,
    ToolTriggerEvaluator,
)
from mesh_eval.evaluator.structured import ToolCallEvaluator


REPO_ROOT = Path(__file__).resolve().parents[1]

T4_BENCHMARKS = {
    ("instruction_following", "fluent_speech_commands"): "t4-if-fluent_speech_commands",
    ("instruction_following", "slurp"): "t4-if-slurp",
    ("instruction_following", "vocalbench"): "t4-if-vocalbench",
    ("instruction_following", "vocalbench_zh"): "t4-if-vocalbench-zh",
    ("instruction_following", "voicebench"): "t4-if-voicebench",
    ("tool_call", "audioagentbench_suite"): "t4-tool-audioagentbench",
    ("tool_call", "fluent_speech_commands"): "t4-tool-fluent_speech_commands",
    ("tool_call", "slurp"): "t4-tool-slurp",
    ("tool_call", "stepeval_audio_toolcall"): "t4-tool-stepeval",
    ("multi_turn_dialogue", "audioagentbench_suite"): "t4-dialogue-audioagentbench",
    ("multi_turn_dialogue", "mtalk_bench"): "t4-dialogue-mtalk",
    ("multi_turn_dialogue", "vocalbench"): "t4-dialogue-vocalbench",
    ("multi_turn_dialogue", "vocalbench_zh"): "t4-dialogue-vocalbench-zh",
    ("multi_turn_dialogue", "voicebench"): "t4-dialogue-voicebench",
    ("clarification", "audioagentbench_suite"): "t4-clarification-audioagentbench",
    ("clarification", "ear_wdyl"): "t4-clarification-ear",
    ("interruption", "ihbench"): "t4-interruption-ihbench",
}


def mesh_registry() -> Registry:
    return Registry([REPO_ROOT / "registry", REPO_ROOT / "mesh_eval" / "registry"])


def test_audioagent_clarification_review_is_packaged_balanced_and_fail_closed(
    tmp_path,
):
    dataset = AudioAgentBenchDataset(
        benchmark_id="audio-agent-clarification-test",
        view="clarification",
        root=str(tmp_path),
    )
    cases = dataset._clarification_cases()

    assert len(cases) == 28
    assert sum(case["should_clarify"] for case in cases.values()) == 10
    assert cases[("grocery-bench", 5)]["should_clarify"] is True
    assert cases[("grocery-bench", 5)]["subcategory"] == "missing_quantity"

    missing = AudioAgentBenchDataset(
        benchmark_id="audio-agent-clarification-test",
        view="clarification",
        root=str(tmp_path),
        clarification_review_path=str(tmp_path / "missing-review.json"),
    )
    with pytest.raises(FileNotFoundError, match="clarification review is required"):
        missing.load(1)

    with pytest.raises(ValueError, match="digest mismatch"):
        dataset._validate_clarification_case(
            "grocery-bench",
            {
                "turn_id": 5,
                "input_text": "And some organic eggs.",
                "golden_text": "stale golden response",
            },
            cases[("grocery-bench", 5)],
        )


@pytest.mark.parametrize(
    "prediction",
    [
        '<tool_call>function\nsearch\n{"query":"VoxMatrix"}</tool_call>',
        '<tool_call>function search {"query":"VoxMatrix"}</tool_call>',
    ],
)
def test_qwen_tool_call_formats_are_parsed_without_claiming_execution(prediction):
    score = ToolCallEvaluator()(
        prediction,
        "",
        reference_obj={
            "expected_output": {
                "tool": "search",
                "arguments": {"query": "VoxMatrix"},
            }
        },
    )

    assert score["json_valid"] == 1
    assert score["tool_acc"] == 1
    assert score["parameter_acc"] == 1
    assert score["call_exact_match"] == 1
    assert "task_success" not in score


def test_clarification_decisions_and_sufficient_statistics_aggregation():
    evaluator = ClarificationEvaluator()
    rows = [
        evaluator('{"should_clarify": true, "response": "Which one?"}', {"should_clarify": True}),
        evaluator('{"should_clarify": true, "response": "Which one?"}', {"should_clarify": False}),
        evaluator("Which one?", {"should_clarify": True}),
        evaluator('{"should_clarify": false, "response": "Done."}', {"should_clarify": False}),
    ]

    assert rows[0]["clarification_tp"] == 1
    assert rows[1]["clarification_fp"] == 1
    assert rows[2]["clarification_parse_valid"] == 0
    assert rows[2]["clarification_accuracy"] == 0
    assert rows[2]["clarification_fn"] == 1
    assert rows[3]["clarification_tn"] == 1

    summary = MeshAgg(group_by=[], min_slice_size=1)(rows)
    assert summary["overall/clarification_precision"] == pytest.approx(0.5)
    assert summary["overall/clarification_recall"] == pytest.approx(0.5)
    assert summary["overall/clarification_f1"] == pytest.approx(0.5)
    assert summary["overall/clarification/aggregation"] == "sufficient_statistics"

    missing_reference = evaluator('{"should_clarify": true}', {})
    assert missing_reference["status"] == "not_evaluated"
    assert missing_reference["clarification_accuracy"] is None
    assert "overall/clarification_f1" not in MeshAgg(
        group_by=[], min_slice_size=1
    )([missing_reference])


def test_stepeval_trigger_type_aliases_and_parameter_judge_boundary():
    positive = ToolTriggerEvaluator()(
        '<tool_call>function search {"query":"最新政策"}</tool_call>',
        {
            "target_tool": "web_search",
            "polarity": "positive",
            "gold_call": {"name": "web_search", "arguments": {"query": "最新政策"}},
        },
    )
    assert positive["trigger_tp"] == 1
    assert positive["trigger_correct"] == 1
    assert positive["tool_type_accuracy"] == 1
    assert positive["parameter_exact_match"] == 1
    assert positive["parameter_judge_accuracy"] is None
    assert positive["parameter_evaluation_status"] == (
        "not_evaluated_official_judge_missing"
    )

    target_false_alarm = ToolTriggerEvaluator()(
        '{"tool":"timbre_rag","arguments":{"query":"robot"}}',
        {"target_tool": "timbre", "polarity": "negative"},
    )
    assert target_false_alarm["trigger_correct"] == 0
    assert target_false_alarm["trigger_fp"] == 1

    unrelated_call = ToolTriggerEvaluator()(
        '{"tool":"calculate","arguments":{"expression":"1+1"}}',
        {"target_tool": "get_weather", "polarity": "negative"},
    )
    assert unrelated_call["trigger_correct"] == 1
    assert unrelated_call["trigger_tn"] == 1

    judged = ToolTriggerEvaluator(parameter_judge=lambda payload: {"correct": True})(
        '{"tool":"get_weather","arguments":{"location":"chengdu"}}',
        {
            "target_tool": "get_weather",
            "polarity": "positive",
            "gold_call": {
                "name": "get_weather",
                "arguments": {"location": "chengdu"},
            },
        },
    )
    assert judged["parameter_judge_accuracy"] == 1
    assert judged["parameter_evaluation_status"] == "evaluated"

    missed = ToolTriggerEvaluator()(
        '{"tool": null, "arguments": {}}',
        {
            "target_tool": "get_weather",
            "polarity": "positive",
            "gold_call": {
                "name": "get_weather",
                "arguments": {"location": "chengdu"},
            },
        },
    )
    assert missed["trigger_fn"] == 1
    assert missed["tool_type_accuracy"] is None

    target_second = ToolTriggerEvaluator()(
        '[{"tool":"calculate","arguments":{}},'
        '{"tool":"get_weather","arguments":{"location":"chengdu"}}]',
        {
            "target_tool": "get_weather",
            "polarity": "positive",
            "gold_call": {
                "name": "get_weather",
                "arguments": {"location": "chengdu"},
            },
        },
    )
    assert target_second["tool_type_accuracy"] == 1
    assert target_second["parameter_exact_match"] == 1

    no_parameter = ToolTriggerEvaluator()(
        '{"tool":"get_date_time","arguments":{}}',
        {
            "target_tool": "get_date_time",
            "polarity": "positive",
            "gold_call": {"name": "get_date_time", "arguments": {}},
        },
    )
    assert no_parameter["parameter_judge_accuracy"] is None
    assert no_parameter["parameter_evaluation_status"] == (
        "not_applicable_no_parameters"
    )


def test_slurp_adapter_preserves_repeated_native_entity_types():
    slots = SLURPT4Dataset._slots(
        {
            "tokens": [
                {"surface": "from"},
                {"surface": "las"},
                {"surface": "vegas"},
                {"surface": "to"},
                {"surface": "los"},
                {"surface": "angeles"},
            ],
            "entities": [
                {"type": "place_name", "span": [1, 2]},
                {"type": "place_name", "span": [4, 5]},
            ],
        }
    )

    assert slots == {"place_name": ["las vegas", "los angeles"]}


def test_text_rubric_judge_fails_closed_without_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMCenterUserToken", raising=False)

    with pytest.raises(RuntimeError, match="requires OPENAI_API_KEY"):
        AgentRubricEvaluator()("candidate", {"rubric": "be relevant"})


def test_audio_rubric_is_not_scored_when_output_is_missing_or_incomplete(tmp_path):
    audio = tmp_path / "response.wav"
    with wave.open(str(audio), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)

    evaluator = AgentRubricEvaluator(
        judge=lambda payload: {"score": 1},
        require_audio=True,
    )
    missing = evaluator("text only", {"rubric": "speak clearly"})
    assert missing["status"] == "not_evaluated"
    assert missing["not_evaluated"] == 1
    long_text = evaluator("word " * 2000, {"rubric": "speak clearly"})
    assert long_text["status"] == "not_evaluated"

    incomplete = evaluator(
        json.dumps({"audio_responses": [str(audio), None]}),
        {
            "rubric": "answer both turns in speech",
            "required_audio_outputs": 2,
        },
    )
    assert incomplete["status"] == "not_evaluated"
    assert "1/2 outputs present" in incomplete["not_evaluated_reason"]

    no_audio_judge = evaluator(
        json.dumps({"audio_responses": [str(audio)]}),
        {"rubric": "speak clearly"},
    )
    assert no_audio_judge["status"] == "not_evaluated"
    assert "audio-capable" in no_audio_judge["not_evaluated_reason"]

    evaluated = AgentRubricEvaluator(
        judge=lambda payload: {"score": 0.75},
        require_audio=True,
        audio_judge=True,
    )(
        json.dumps({"audio_responses": [str(audio)]}),
        {"rubric": "speak clearly"},
    )
    assert evaluated["status"] == "evaluated"
    assert evaluated["rubric_score"] == pytest.approx(0.75)


def test_ifeval_adapter_reads_canonical_reference_object():
    score = IFEvalAdapter()(
        "#one #two #three #four",
        "",
        reference_obj={
            "prompt": "Write a tweet.",
            "instruction_id_list": ["keywords:letter_frequency"],
            "kwargs": [
                {"letter": "#", "let_frequency": 4, "let_relation": "at least"}
            ],
        },
    )

    assert score["status"] == "evaluated"
    assert score["constraint_satisfaction"] == 1
    assert score["strict_instruction"] == 1

    numeric_kwargs = IFEvalAdapter()(
        "send this first\n\nsecond paragraph\n\nthird paragraph",
        "",
        reference_obj={
            "prompt": "Write three paragraphs.",
            "instruction_id_list": [
                "length_constraints:nth_paragraph_first_word"
            ],
            "kwargs": [
                {
                    "num_paragraphs": 3.0,
                    "nth_paragraph": 1.0,
                    "first_word": "send",
                }
            ],
        },
    )
    assert numeric_kwargs["strict_instruction"] == 1

    numpy_kwargs = IFEvalAdapter()(
        "a safe response",
        "",
        reference_obj={
            "prompt": "Do not mention blocked words.",
            "instruction_id_list": ["keywords:forbidden_words"],
            "kwargs": [
                {"forbidden_words": __import__("numpy").array(["x", "y"])}
            ],
        },
    )
    assert numpy_kwargs["strict_instruction"] == 1


def test_ihbench_uses_tf_rq_rubric_fields_not_lexical_similarity():
    score = IHBenchEvaluator(
        judge=lambda payload: {"tf_win": 0.5, "rq_pass": 1, "reason": "tie"}
    )(
        "wording unrelated to the baseline",
        {
            "baseline": "baseline wording",
            "task_fulfillment_rubric": "continue the interrupted task",
            "response_quality_rubrics": ["do not repeat"],
        },
    )

    assert score["tf_win_score"] == pytest.approx(0.5)
    assert score["rq_pass"] == pytest.approx(1.0)
    assert "text_f1" not in score
    assert "token_f1" not in score

    with pytest.raises(ValueError, match="both TF and RQ"):
        IHBenchEvaluator(judge=lambda payload: {"tf_win": 1})(
            "candidate",
            {
                "baseline": "baseline",
                "task_fulfillment_rubric": "finish",
                "response_quality_rubrics": ["concise"],
            },
        )


def test_historical_smoke_replay_uses_production_metrics(monkeypatch):
    from tools.capability_t4_smoke import _score

    natural_language_clarification = _score(
        {
            "capability": "clarification",
            "benchmark": "audioagentbench_suite",
            "expected": {"should_clarify": True},
        },
        "Which session did you mean?",
    )
    assert natural_language_clarification["clarification_parse_valid"] == 0
    assert natural_language_clarification["clarification_accuracy"] == 0

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMCenterUserToken", raising=False)
    interruption = _score(
        {
            "capability": "interruption",
            "benchmark": "ihbench",
            "expected": {
                "tf_rubric": "continue the task",
                "rq_rubrics": ["do not repeat"],
            },
            "reference": {"baseline": "baseline answer"},
        },
        "baseline answer",
    )
    assert interruption["status"] == "not_evaluated"
    assert interruption["value"] is None
    assert "token_f1" not in interruption


def test_t4_benchmark_map_covers_every_assigned_grid_cell():
    benchmarks = load_benchmark_map()["benchmarks"]
    actual_ids = {key for key in benchmarks if key.startswith("t4-")}
    assert actual_ids == set(T4_BENCHMARKS.values())
    for key, benchmark_id in T4_BENCHMARKS.items():
        profile = benchmarks[benchmark_id]
        assert profile["capability"] == key[0]
        # The benchmark map keeps the public V1 task id; native adapters
        # normalize it to ``spoken_agentic_interaction`` at runtime.
        assert profile["task"] == "agent"
        assert profile["registry_dataset"]
        assert profile["metrics"]
        assert profile["evaluators"]
        assert profile["prompt"]
        assert profile["use_bucket"] == "diagnostic_evidence"


def _write_silent_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


@pytest.mark.parametrize(
    ("view", "prediction", "metric"),
    [
        (
            "instruction_following",
            '{"intent":"activate_lights","slots":{"action":"activate","object":"lights","location":"none"}}',
            "intent_acc",
        ),
        (
            "tool_call",
            '{"tool":"control_device","arguments":{"action":"activate","object":"lights","location":"none"}}',
            "call_exact_match",
        ),
    ],
)
def test_fsc_dataset_prompt_evaluator_aggregation_chain(tmp_path, view, prediction, metric):
    audio = tmp_path / "wavs" / "sample.wav"
    _write_silent_wav(audio)
    csv_path = tmp_path / "data" / "test_data.csv"
    csv_path.parent.mkdir(parents=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "transcription", "action", "object", "location"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "path": "wavs/sample.wav",
                "transcription": "turn on the lights",
                "action": "activate",
                "object": "lights",
                "location": "none",
            }
        )

    doc = FluentSpeechCommandsT4Dataset(
        benchmark_id=f"test-fsc-{view}",
        view=view,
        root=str(tmp_path),
    ).load(1)[0]
    registry = mesh_registry()
    prompt = registry.get_prompt("mesh-router-prompt").load(**doc)
    score = registry.get_evaluator("mesh-router-evaluator")(
        prediction, doc["text"], **doc
    )
    summary = MeshAgg(group_by=[], min_slice_size=1)([score])

    assert prompt
    assert doc["schema_version"] == "runtime-sample/2.0"
    assert not doc.get("schema_errors")
    assert score[metric] == 1
    assert summary[f"overall/{metric}"] == pytest.approx(1.0)


@pytest.mark.skipif(
    os.environ.get("VOXMATRIX_RUN_REAL_T4") != "1",
    reason="requires the shared native T4 benchmark roots",
)
def test_real_t4_adapters_load_render_and_validate_ten_samples_per_grid_cell():
    """Opt-in AFS integration smoke used for the T4 delivery audit."""
    registry = mesh_registry()
    router = registry.get_evaluator("mesh-router-evaluator")
    benchmarks = load_benchmark_map()["benchmarks"]
    total = 0
    locally_scored = []
    for (capability, _source), benchmark_id in T4_BENCHMARKS.items():
        profile = benchmarks[benchmark_id]
        dataset = registry.get_dataset(profile["registry_dataset"])
        rows = dataset.load(10)
        assert len(rows) == 10, benchmark_id
        for row in rows:
            total += 1
            assert row["schema_version"] == "runtime-sample/2.0"
            assert row["benchmark_id"] == profile["registry_dataset"]
            assert row["capability"] == capability
            assert row["metrics"] == profile["metrics"]
            assert row["evaluators"] == profile["evaluators"]
            assert row["prompt_name"] == profile["prompt"]
            assert not row.get("schema_errors")
            assert all(Path(item["uri"]).is_file() for item in row["input"]["audio"])
            assert registry.get_prompt("mesh-router-prompt").load(**row)
            if benchmark_id in {
                "t4-dialogue-mtalk",
                "t4-dialogue-voicebench",
            }:
                rendered = registry.get_prompt("mesh-router-prompt").load(**row)
                assert rendered["multi_turn"][0][0]["role"] == "system"
            evaluator_names = set(row["evaluators"])
            prediction = None
            if evaluator_names == {"mesh-intent", "mesh-slot-f1"}:
                prediction = json.dumps(
                    {
                        "intent": row["reference_obj"]["intent"],
                        "slots": row["reference_obj"].get("slots", {}),
                    },
                    ensure_ascii=False,
                )
            elif evaluator_names == {"mesh-tool-call"}:
                prediction = json.dumps(
                    row["reference_obj"]["expected_output"], ensure_ascii=False
                )
            elif evaluator_names == {"mesh-tool-trigger"}:
                reference = row["reference_obj"]
                prediction = (
                    json.dumps(reference.get("gold_call"), ensure_ascii=False)
                    if reference.get("polarity") == "positive"
                    else '{"tool": null, "arguments": {}}'
                )
            elif evaluator_names == {"mesh-clarification"}:
                prediction = json.dumps(
                    {
                        "should_clarify": row["reference_obj"]["should_clarify"],
                        "response": "Which value?",
                    }
                )
            elif evaluator_names == {"mesh-ifeval"}:
                # An empty response is a legal candidate and should receive a
                # checker score, not a parser/runtime failure.
                prediction = ""
            if prediction is not None:
                score = router(prediction, row["text"], **row)
                assert not score["mesh_unmapped_metrics"]
                for metric in row["metrics"]:
                    value = score.get(metric)
                    if value is not None:
                        assert isinstance(value, (int, float))
                        assert 0 <= value <= 1
                locally_scored.append(score)
    assert total == 170
    assert len(locally_scored) == 90
    assert MeshAgg(group_by=[], min_slice_size=1)(locally_scored)[
        "sample_count"
    ] == 90
