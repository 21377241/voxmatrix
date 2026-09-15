import json
import time
import wave
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
from datasets import Dataset, DatasetDict

from annotation.ai.collect_knowledge import load_source_card
from annotation.pipeline.common import load_mapping
from audio_evals.registry import Registry
from audio_evals.agg.original_benchmark import OriginalACC
from audio_evals.eval_task import EvalTask, _audio_duration_seconds, _measured_runtime
from audio_evals.utils import merge_data4view
from audio_evals.process.speech import Speech2text
from audio_evals.recorder import Recorder
from audio_evals.evaluator.coco import Coco
from audio_evals.evaluator.voice_bench import (
    VoiceBenchMultiTurnEvaluator,
    VoiceBenchQaOpenEvaluator,
    resolve_judge_model_name,
)
from audio_evals.evaluator.dnsmos import DNSMOS
from audio_evals.evaluator.bbh import BBH
from audio_evals.evaluator.harm import Harm
from audio_evals.evaluator.seed_tts_eval_asr_wer import SeedTTSEvalASRWER
from audio_evals.evaluator.simo import Simo
from audio_evals.evaluator.utmos import UTMOS
from audio_evals.models.qwen3_omni import Qwen3Omni
from audio_evals.models.fun_audio_chat import AUDIO_TEMPLATE, FunAudioChat
from audio_evals.models.qwen2_5 import QwenOmni
from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.core.benchmark_map import benchmark_names, benchmark_profile
from mesh_eval.core.prompt import AudioSegmentPrompt
from mesh_eval.core.schema import (
    default_metrics,
    normalize_record,
    normalize_split,
    validate_record,
)
from mesh_eval.core.schema_v2 import adapt_v1_to_v2
from mesh_eval.dataset.full_eval import FullEvalDataset
from mesh_eval.dataset.external_benchmarks import (
    AudioAgentBenchDataset,
    IHBenchDataset,
)
from mesh_eval.evaluator.edge import AuthorizationEvaluator, ResponseGateEvaluator
from mesh_eval.evaluator.external import BertScoreEvaluator
from mesh_eval.evaluator.structured import StructuredJsonEvaluator, ToolCallEvaluator
from mesh_eval.evaluator.speaker import SpeakerAttributionEvaluator
from mesh_eval.scripts.validate_benchmark_map import validate
from mesh_eval.scripts.attach_tts_candidates import attach_candidates
from mesh_eval.scripts.export_events_xlsx import export_events_xlsx
from mesh_eval.scripts.merge_event_shards import (
    aggregate_events,
    load_events,
    write_events,
)
from mesh_eval.scripts.slice_event_range import slice_event_range
from mesh_eval.scripts.validate_event_completion import (
    validate_event_completion,
    validate_overall_summary,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def mesh_registry() -> Registry:
    return Registry([REPO_ROOT / "registry", REPO_ROOT / "mesh_eval" / "registry"])


def test_benchmark_map_and_registry_are_consistent():
    report = validate()
    assert report["benchmark_count"] >= 26
    assert report["error_count"] == 0
    assert len(benchmark_names(groups=["current_asr"])) == 6
    assert benchmark_profile("WenetSpeech-test-meeting")["scenario"] == "meeting"
    seed_profile = benchmark_profile("seed_tts_eval_en")
    assert seed_profile["metrics"] == [
        "content_acc",
        "intelligibility",
        "sim",
        "mos",
        "dnsmos",
    ]


def test_taxonomy_is_canonical_and_rejects_cross_task_capability():
    record = normalize_record(
        {
            "sample_id": "contract-1",
            "dataset": "contract",
            "task": "speech_understanding",
            "capability": "tool_call",
            "scenario": "phone",
            "condition": {
                "acoustic": "clean",
                "spatial": "near_field",
                "speaker": "single_speaker",
                "device": "phone_mic",
                "interaction": "single_turn",
            },
            "input": {"audio_path": "assets/default.wav"},
            "reference": {"expected_output": {"tool": "alarm"}},
            "metrics": "em,f1,unknown_metric",
            "use_bucket": "diagnostic_evidence",
            "split": "validation",
        },
        ref_col="text",
    )

    assert record["metrics"] == ["exact_match", "text_f1", "unknown_metric"]
    assert record["split"] == "dev"
    assert record["source_split"] == "validation"
    assert default_metrics("agent", "qa") == ["exact_match", "text_f1"]
    errors = validate_record(record, ref_col="text")
    assert "capability tool_call does not belong to task speech_understanding" in errors
    assert "unknown metric: unknown_metric" in errors


def test_historical_replay_splits_normalize_to_test():
    assert normalize_split("replay_mini") == "test"
    assert normalize_split("event_reuse_mini") == "test"


def test_router_is_lazy_and_routes_asr_metric():
    router = mesh_registry().get_evaluator("mesh-router-evaluator")
    assert router.evaluators == {}
    score = router(
        "hello world",
        "hello world",
        metrics=["wer"],
        task="speech_understanding",
        capability="asr",
    )
    assert score["wer%"] == 0
    assert set(router.evaluators) == {"wer"}
    direct = router(
        "positive",
        "positive",
        evaluators=["mesh-classification"],
        metrics=["comet"],
        task="speech_understanding",
        capability="paralinguistic",
    )
    assert direct["accuracy"] == 1
    assert direct["mesh_evaluator"] == "mesh-classification"

    mixed = router(
        "hello",
        "hello",
        metrics=["bleu", "future_metric"],
        task="speech_understanding",
        capability="speech_translation",
    )
    assert mixed["mesh_unmapped_metrics"] == "future_metric"


def test_structured_and_edge_evaluators():
    structured = StructuredJsonEvaluator()
    score = structured(
        "```json\n{\"city\": \"Beijing\", \"days\": 2}\n```",
        "",
        reference_obj={"expected_output": {"city": "Beijing", "days": 2}},
    )
    assert score["json_valid"] == 1
    assert score["structured_exact_match"] == 1

    tool = ToolCallEvaluator()
    score = tool(
        '{"tool":"navigate","arguments":{"destination":"office"}}',
        "",
        edge={
            "expected_action": {
                "tool": "navigate",
                "arguments": {"destination": "office"},
            }
        },
    )
    assert score["tool_acc"] == 1
    assert score["parameter_acc"] == 1
    assert score["call_exact_match"] == 1
    assert "task_success" not in score

    calls = [
        {"tool": "register", "arguments": {"id": "one"}},
        {"tool": "register", "arguments": {"id": "two"}},
    ]
    tagged = "".join(
        f"<tool_call>{json.dumps({'name': call['tool'], 'arguments': call['arguments']})}</tool_call>"
        for call in calls
    )
    multi_score = tool(tagged, "", reference_obj={"expected_output": calls})
    assert multi_score["json_valid"] == 1
    assert multi_score["tool_acc"] == 1
    assert multi_score["parameter_acc"] == 1
    assert multi_score["call_exact_match"] == 1
    assert "task_success" not in multi_score

    response = ResponseGateEvaluator()("false", "", edge={"should_respond": False})
    assert response["response_acc"] == 1
    assert response["false_accept"] == 0
    authorization = AuthorizationEvaluator()("允许", "", edge={"authorized": False})
    assert authorization["authorization_acc"] == 0
    assert authorization["unauthorized_accept"] == 1


def test_full_eval_proxy_and_alimeeting_reconstruction(tmp_path):
    audio_path = tmp_path / "meeting.wav"
    second_audio_path = tmp_path / "second.wav"
    for path in (audio_path, second_audio_path):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 32000)

    slurp_manifest = tmp_path / "slurp" / "final" / "manifest.jsonl"
    slurp_manifest.parent.mkdir(parents=True)
    slurp_manifest.write_text(
        json.dumps(
            {
                "dataset": "slurp",
                "task": "agent",
                "capability": "instruction_following",
                "scenario": "home",
                "condition": {
                    "acoustic": "clean",
                    "spatial": "near_field",
                    "speaker": "single_speaker",
                    "device": "smart_speaker",
                    "interaction": "single_turn",
                },
                "metrics": "intent_acc,slot_f1",
                "use_bucket": "diagnostic_evidence",
                "split": "test",
                "input": {"audio_path": str(audio_path)},
                "reference": {"intent": "calendar_set", "slots": {"day": "monday"}},
                "sample_id": "slurp-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    proxy = FullEvalDataset(
        dataset="slurp",
        benchmark_id="slurp-tool",
        view="tool_call_proxy",
        root=str(tmp_path),
    ).load(1)[0]
    assert proxy["capability"] == "tool_call"
    assert proxy["schema_version"] == "runtime-sample/2.0"
    assert proxy["task"] == "spoken_agentic_interaction"
    assert proxy["capability_protocol"] == (
        "spoken_agentic_interaction/tool_call@1"
    )
    assert proxy["reference_obj"]["expected_output"] == {
        "tool": "execute_slurp_intent",
        "arguments": {
            "intent": "calendar_set",
            "slots": {"day": "monday"},
        },
    }

    alimeeting_manifest = tmp_path / "alimeeting" / "final" / "manifest.jsonl"
    alimeeting_manifest.parent.mkdir(parents=True)
    rows = []
    for index, (start, end, speaker, text) in enumerate(
        [(0.0, 1.0, "A", "hello"), (1.0, 2.0, "B", "world")]
    ):
        base = {
            "dataset": "alimeeting",
            "task": "speech_understanding",
            "capability": "asr",
            "scenario": "meeting",
            "condition": {
                "acoustic": "overlap_speech",
                "spatial": "far_field",
                "speaker": "multi_speaker",
                "device": "meeting_array",
                "interaction": "single_turn",
            },
            "metrics": ["cer", "wer"],
            "use_bucket": "formal_subscores",
            "split": "test",
            "input": {
                "audio_path": str(audio_path),
                "start_time": start,
                "end_time": end,
                "speaker_id": speaker,
            },
            "sample_id": f"ali-{index}",
        }
        asr = deepcopy(base)
        asr["reference"] = {"text": text}
        rows.append(asr)
        corrupt = deepcopy(base)
        corrupt.update({"task": "speaker", "reference": {"segments": []}})
        rows.append(corrupt)
    alimeeting_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    diarization = FullEvalDataset(
        dataset="alimeeting",
        benchmark_id="ali-diar",
        view="alimeeting_diarization",
        root=str(tmp_path),
    ).load(1)[0]
    assert diarization["capability"] == "diarization"
    assert diarization["task"] == "speaker_attribution"
    assert diarization["schema_version"] == "runtime-sample/2.0"
    assert len(diarization["reference_obj"]["segments"]) == 2
    assert diarization["reference_reconstruction"] == "grouped_from_asr_timestamps"


def test_full_eval_alimeeting_groups_canonical_v2_audio(tmp_path):
    audio_path = tmp_path / "canonical-meeting.wav"
    audio_path.touch()
    manifest = tmp_path / "alimeeting" / "final" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    rows = []
    for index, (start, end, speaker, text) in enumerate(
        [(0.0, 1.0, "A", "你好"), (1.0, 2.0, "B", "世界")]
    ):
        rows.append(
            adapt_v1_to_v2(
                {
                    "dataset": "alimeeting",
                    "task": "speech_understanding",
                    "capability": "asr",
                    "scenario": "meeting",
                    "condition": {
                        "acoustic": "overlap_speech",
                        "spatial": "far_field",
                        "speaker": "multi_speaker",
                        "device": "meeting_array",
                    },
                    "metrics": ["cer"],
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "input": {
                        "audio_path": str(audio_path),
                        "start_time": start,
                        "end_time": end,
                        "speaker_id": speaker,
                    },
                    "reference": {"text": text},
                    "sample_id": f"canonical-ali-{index}",
                }
            )
        )
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    diarization = FullEvalDataset(
        dataset="alimeeting",
        benchmark_id="canonical-ali-diar",
        view="alimeeting_diarization",
        root=str(tmp_path),
    ).load(1)[0]

    assert diarization["WavPath"] == str(audio_path)
    assert len(diarization["reference_obj"]["segments"]) == 2


def test_full_eval_covost_view_reads_canonical_v2_audio_and_language(tmp_path):
    audio_path = tmp_path / "audio" / "zh-CN" / "sample.wav"
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()
    manifest = tmp_path / "covost2" / "final" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    row = adapt_v1_to_v2(
        {
            "dataset": "covost2",
            "task": "speech_understanding",
            "capability": "speech_translation",
            "scenario": "public",
            "condition": {
                "acoustic": "clean",
                "spatial": "near_field",
                "speaker": "single_speaker",
                "device": "phone_mic",
            },
            "metrics": ["bleu"],
            "use_bucket": "diagnostic_evidence",
            "split": "test",
            "input": {"audio_path": str(audio_path)},
            "reference": {"text": "hello"},
            "metadata": {"language": "zh-CN", "target_language": "en"},
            "sample_id": "canonical-covost-zh-en",
        }
    )
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = FullEvalDataset(
        dataset="covost2",
        benchmark_id="canonical-covost-zh-en",
        view="covost_zh_en",
        root=str(tmp_path),
    ).load()

    assert len(loaded) == 1
    assert loaded[0]["WavPath"] == str(audio_path)
    assert loaded[0]["metadata"]["language"] == "zh"
    assert loaded[0]["metadata"]["target_language"] == "en"


def test_segment_prompt_and_runtime_measurement_preserve_replay(tmp_path):
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 32000)

    prompt = AudioSegmentPrompt(
        [
            {
                "role": "user",
                "contents": [{"type": "audio", "value": "{{WavPath}}"}],
            }
        ]
    ).load(
        WavPath=str(audio_path),
        input={"audio_path": str(audio_path), "start_time": 0.25, "end_time": 1.25},
    )
    assert prompt[0]["contents"][0]["value"] == {
        "path": str(audio_path),
        "start_time": 0.25,
        "end_time": 1.25,
    }

    class Predictor:
        calls = 0

        def inference(self, real_prompt):
            self.calls += 1
            return "ok"

    class Evaluator:
        def __call__(self, pred, ref, **kwargs):
            return {"match": int(pred == ref)}

    predictor = Predictor()
    recorder = Recorder(str(tmp_path / "runtime" / "events.jsonl"))
    task = EvalTask(
        dataset=SimpleNamespace(ref_col="text"),
        prompt=None,
        predictor=predictor,
        evaluator=Evaluator(),
        post_process=[],
        agg=None,
        recorder=recorder,
    )
    score, _ = task._eval(
        0,
        "prompt",
        "ok",
        WavPath=str(audio_path),
        input={"audio_path": str(audio_path)},
    )
    assert score["latency_ms"] >= 0
    assert score["rtf"] >= 0
    assert "first_token_latency_ms" not in score

    replay_score, _ = task._eval(
        1,
        "prompt",
        "ok",
        eval_info={
            "inference": {
                "content": "ok",
                "runtime": {"latency_ms": 123.0, "rtf": 0.5},
            }
        },
    )
    assert replay_score["latency_ms"] == 123.0
    assert replay_score["rtf"] == 0.5
    assert predictor.calls == 1


def test_mesh_agg_derives_macro_f1_runtime_and_eer():
    rows = [
        {
            "classification_pred": "a",
            "classification_ref": "a",
            "verification_score": 0.9,
            "verification_label": 1,
            "latency_ms": 10.0,
            "timeout": 0,
        },
        {
            "classification_pred": "a",
            "classification_ref": "b",
            "verification_score": 0.1,
            "verification_label": 0,
            "latency_ms": 30.0,
            "timeout": 1,
        },
    ]
    result = MeshAgg()._agg(rows)
    assert result["overall/macro_f1"] == pytest.approx(1 / 3)
    assert result["overall/eer"] == 0
    assert result["overall/auc"] == 1
    assert result["overall/latency_p50"] == 20
    assert result["overall/timeout_rate"] == 0.5
    assert "overall/verification_score" not in result
    assert "overall/verification_label" not in result


def test_voicebench_open_judge_accepts_bracketed_rating(monkeypatch):
    from audio_evals.registry import registry

    class FakeModel:
        def inference(self, prompt, **kwargs):
            return "[[4]]"

    class FakePrompt:
        def load(self, **kwargs):
            return kwargs["real_prompt"]

    monkeypatch.setattr(registry, "get_model", lambda name: FakeModel())
    monkeypatch.setattr(registry, "get_prompt", lambda name: FakePrompt())

    score = VoiceBenchQaOpenEvaluator("fake")(
        "candidate answer", "", question="question"
    )
    assert score["gpt_score"] == 4


def test_voicebench_judge_backend_can_fall_back_to_llmcenter(monkeypatch):
    monkeypatch.delenv("MESH_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLMCenterUserToken", "test-token")
    assert resolve_judge_model_name("gpt4o-mini") == "mb-gpt4o-mini"

    monkeypatch.setenv("MESH_JUDGE_MODEL", "custom-judge")
    assert resolve_judge_model_name("gpt4o-mini") == "custom-judge"


def test_attach_tts_candidates_refreshes_routes(tmp_path):
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    candidate_path = candidate_dir / "seed-output.wav"
    with wave.open(str(candidate_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)
    rows = [
        {
            "sample_id": "seed_tts_eval_en_00000",
            "dataset": "seed_tts_eval_en",
            "dataset_id": "seed_tts_eval_en",
            "filename": "seed-output",
            "ans": "wavs/seed-output.wav",
            "metadata": {"language": "en"},
        }
    ]

    attached, report = attach_candidates(rows, candidate_dir)

    assert report["attached_count"] == 1
    assert report["missing_count"] == 0
    assert attached[0]["eval_info"]["post_process"]["content"] == str(
        candidate_path.resolve()
    )
    assert attached[0]["evaluators"] == [
        "seed-tts-eval-asr-wer-en",
        "simo",
        "utmos",
        "dnsmos",
    ]


def test_event_completion_rejects_missing_eval(tmp_path):
    event_path = tmp_path / "events.jsonl"
    events = [
        {"type": event_type, "id": 0, "data": {}}
        for event_type in ("prompt", "inference", "post_process", "eval")
    ]
    events.extend(
        {"type": event_type, "id": 1, "data": {}}
        for event_type in ("prompt", "inference", "post_process")
    )
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    report = validate_event_completion(event_path, expected_count=2)

    assert report["error_count"] == 1
    assert report["missing"]["eval"] == [1]


def test_event_completion_supports_nonzero_start(tmp_path):
    event_path = tmp_path / "events.jsonl"
    events = [
        {"type": event_type, "id": event_id, "data": {}}
        for event_id in (5, 6)
        for event_type in ("prompt", "inference", "post_process", "eval")
    ]
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    report = validate_event_completion(
        event_path, expected_count=2, expected_start=5
    )

    assert report["error_count"] == 0
    assert report["observed_id_count"] == 2


def test_eval_task_slices_dataset_and_offsets_event_ids(tmp_path):
    class SliceDataset:
        ref_col = "text"

        def __init__(self):
            self.load_limit = None

        def load(self, limit=0):
            self.load_limit = limit
            rows = [{"text": str(index)} for index in range(6)]
            return rows[:limit] if limit else rows

    class Prompt:
        def load(self, **doc):
            return doc["text"]

    class Predictor:
        def inference(self, prompt):
            return prompt

    class Evaluator:
        def __call__(self, pred, ref, **kwargs):
            return {"pred": pred, "ref": ref}

    dataset = SliceDataset()
    event_path = tmp_path / "offset.jsonl"
    task = EvalTask(
        dataset=dataset,
        prompt=Prompt(),
        predictor=Predictor(),
        evaluator=Evaluator(),
        post_process=[],
        agg=lambda rows: {},
        recorder=Recorder(str(event_path)),
        event_id_offset=2,
    )

    rows = task._load_dataset(limit=2, offset=2)
    task._run(0, rows[0])

    assert dataset.load_limit == 4
    assert [row["text"] for row in rows] == ["2", "3"]
    assert {
        json.loads(line)["id"] for line in event_path.read_text().splitlines()
    } == {2}


def test_merge_event_shards_reaggregates_global_scores(tmp_path):
    shard_paths = []
    for event_id, prediction in ((0, "yes"), (1, "no")):
        path = tmp_path / f"shard-{event_id}.jsonl"
        events = [
            {"type": event_type, "id": event_id, "data": {}}
            for event_type in ("prompt", "inference", "post_process")
        ]
        events.append(
            {
                "type": "eval",
                "id": event_id,
                "data": {"pred": prediction, "ref": "yes"},
            }
        )
        path.write_text(
            "".join(json.dumps(event) + "\n" for event in events),
            encoding="utf-8",
        )
        shard_paths.append(path)

    events = load_events(shard_paths)
    merged_path = tmp_path / "merged.jsonl"
    write_events(events, merged_path)
    summary = aggregate_events(events, "original-benchmark-acc", "mesh_eval/registry")

    assert validate_event_completion(merged_path, 2)["error_count"] == 0
    assert summary["acc(%)"] == 50.0
    assert summary["sample_count"] == 2
    assert summary["failure_rate"] == 0.0


def test_slice_event_range_extracts_complete_inference_prefix(tmp_path):
    source = tmp_path / "source.jsonl"
    events = [
        {"type": event_type, "id": event_id, "data": {}}
        for event_id in range(4)
        for event_type in ("prompt", "inference")
    ]
    source.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    output = tmp_path / "prefix.jsonl"

    report = slice_event_range(
        source,
        output,
        start=0,
        stop=3,
        required_types=("prompt", "inference"),
    )

    assert report["error_count"] == 0
    assert {json.loads(line)["id"] for line in output.read_text().splitlines()} == {
        0,
        1,
        2,
    }


def test_merge_data4view_uses_global_event_offset(tmp_path):
    event_path = tmp_path / "events.jsonl"
    event_path.write_text(
        json.dumps(
            {"type": "inference", "id": 5, "data": {"content": "answer"}}
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "events.xlsx"

    merge_data4view([{"text": "reference"}], str(event_path), output, id_offset=5)

    import pandas as pd

    rows = pd.read_excel(output).to_dict(orient="records")
    assert rows == [{"text": "reference", "id": 5, "inference": "answer"}]


def test_export_events_xlsx_streams_sorted_global_ids(tmp_path):
    event_path = tmp_path / "merged.jsonl"
    events = [
        {"type": "prompt", "id": 5, "data": {"content": ["question"]}},
        {"type": "inference", "id": 5, "data": {"content": "answer"}},
        {"type": "post_process", "id": 5, "data": {"content": "answer"}},
        {
            "type": "eval",
            "id": 5,
            "data": {"pred": "answer", "ref": "answer", "acc(%)": 100.0},
        },
        {"type": "prompt", "id": 6, "data": {"content": ["next"]}},
        {"type": "inference", "id": 6, "data": {"content": "no"}},
        {"type": "post_process", "id": 6, "data": {"content": "no"}},
        {
            "type": "eval",
            "id": 6,
            "data": {"pred": "no", "ref": "yes", "acc(%)": 0.0},
        },
    ]
    event_path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    output = tmp_path / "events.xlsx"

    report = export_events_xlsx(
        event_path, output, expected_count=2, expected_start=5
    )

    import pandas as pd

    rows = pd.read_excel(output).to_dict(orient="records")
    assert report["row_count"] == 2
    assert [row["id"] for row in rows] == [5, 6]
    assert [row["inference"] for row in rows] == ["answer", "no"]
    assert [row["acc(%)"] for row in rows] == [100, 0]


def test_two_phase_releases_predictor_before_post_process(tmp_path):
    state = {"released": False}

    class Dataset:
        ref_col = "text"

        def load(self, limit=0):
            return [{"text": "processed"}]

    class Prompt:
        def load(self, **doc):
            return "prompt"

    class Predictor:
        def inference(self, prompt):
            return "raw"

        def release(self):
            state["released"] = True

    class PostProcess:
        def __call__(self, output):
            assert state["released"]
            return "processed"

    class Evaluator:
        def __call__(self, pred, ref, **kwargs):
            return {"pred": pred, "ref": ref}

    class Agg:
        def __call__(self, rows):
            return {"sample_count": len(rows)}

    event_path = tmp_path / "two_phase.jsonl"
    result = EvalTask(
        dataset=Dataset(),
        prompt=Prompt(),
        predictor=Predictor(),
        evaluator="fake-evaluator",
        post_process=[PostProcess()],
        agg=Agg(),
        recorder=Recorder(str(event_path)),
    )

    from audio_evals.registry import registry

    original_get_evaluator = registry.get_evaluator
    registry.get_evaluator = lambda name: Evaluator()
    try:
        overall, _, _ = result.run_two_phase(max_workers=1)
    finally:
        registry.get_evaluator = original_get_evaluator

    assert state["released"]
    assert overall["sample_count"] == 1
    event_types = [json.loads(line)["type"] for line in event_path.read_text().splitlines()]
    assert event_types == ["prompt", "inference", "post_process", "eval"]


def test_two_phase_resume_reuses_recorded_evaluation(tmp_path):
    class Evaluator:
        calls = 0

        def __call__(self, pred, ref, **kwargs):
            self.calls += 1
            raise AssertionError("recorded evaluation should be replayed")

    evaluator = Evaluator()
    task = EvalTask(
        dataset=SimpleNamespace(ref_col="text"),
        prompt=None,
        predictor=None,
        evaluator=evaluator,
        post_process=[],
        agg=None,
        recorder=Recorder(str(tmp_path / "two_phase_resume.jsonl")),
    )
    _, score, output, error, skipped = task._evaluate_only(
        0,
        "raw",
        {
            "text": "reference",
            "eval_info": {
                "post_process": {"content": "saved-output"},
                "eval": {"saved_score": 1},
            },
        },
    )

    assert evaluator.calls == 0
    assert output == "saved-output"
    assert score["saved_score"] == 1
    assert error == 0
    assert skipped == 0


def test_dataset_load_timeout_interrupts_stalled_loader(tmp_path):
    class StalledDataset:
        ref_col = "text"

        def load(self, limit=0):
            time.sleep(2)

    task = EvalTask(
        dataset=StalledDataset(),
        prompt=None,
        predictor=None,
        evaluator=None,
        post_process=[],
        agg=None,
        recorder=Recorder(str(tmp_path / "timeout.jsonl")),
    )

    with pytest.raises(TimeoutError, match="dataset loading exceeded 1 seconds"):
        task._load_dataset(0, timeout=1)


def test_speech_to_text_loads_model_lazily(tmp_path, monkeypatch):
    audio_path = tmp_path / "generated.wav"
    audio_path.write_bytes(b"audio")
    calls = []

    class Model:
        def inference(self, prompt):
            return "transcript"

    class Prompt:
        def load(self, **kwargs):
            return kwargs["WavPath"]

    from audio_evals.registry import registry

    monkeypatch.setattr(
        registry, "get_model", lambda name: calls.append(("model", name)) or Model()
    )
    monkeypatch.setattr(
        registry, "get_prompt", lambda name: calls.append(("prompt", name)) or Prompt()
    )

    process = Speech2text(model_name="whisper", prompt_name="whisper-asr")
    assert calls == []
    assert process(str(audio_path)) == "transcript"
    assert calls == [("model", "whisper"), ("prompt", "whisper-asr")]


def test_original_benchmark_accuracy_keeps_upstream_contract():
    metric = OriginalACC()
    assert metric(
        [
            {"pred": "neutral", "ref": "neutral", "match": 1},
            {"pred": "sadness", "ref": "joy", "match": 0},
        ]
    ) == {"acc(%)": 50.0}


def test_overall_validation_supports_original_metric_summaries(tmp_path):
    overall_path = tmp_path / "overall.json"
    overall_path.write_text(
        json.dumps({"wer(%)": 1.25, "failure_rate": 0}), encoding="utf-8"
    )
    report = validate_overall_summary(
        overall_path,
        expected_count=2619,
        required_keys=["wer(%)"],
        require_sample_count=False,
    )
    assert report["error_count"] == 0


def test_harm_emits_refusal_and_attack_success_rates():
    evaluator = Harm()
    refusal = evaluator("I'm sorry, I cannot help with that.", "")
    assert refusal["refusal_rate"] == 1
    assert refusal["attack_success_rate"] == 0

    compliance = evaluator("Here are the requested steps.", "")
    assert compliance["refusal_rate"] == 0
    assert compliance["attack_success_rate"] == 1


def test_ifeval_preserves_punctuation_letter_constraints():
    from audio_evals.evaluator.ifeval import IFEval

    score = IFEval(strict=True)(
        "#one #two #three #four",
        "tweet",
        instruction_id_list=["keywords:letter_frequency"],
        kwargs=[{"letter": "#", "let_frequency": 4, "let_relation": "at least"}],
    )
    assert score["strict_instruction"] == 1


def test_bbh_does_not_randomly_guess_unparseable_answers():
    score = BBH()(
        "The reasoning is inconclusive.",
        "No",
        id="bbh_web_of_lies_unknown",
    )
    assert score["match"] == 0
    assert score["extract_fail"] == 1


def test_qwen3_omni_multi_turn_preserves_assistant_history():
    model = object.__new__(Qwen3Omni)
    conversations = []

    def fake_request(conversation):
        conversations.append(deepcopy(conversation))
        return {"text": f"answer {len(conversations)}"}

    model._request = fake_request
    result = json.loads(
        model._inference(
            {
                "multi_turn": [
                    {
                        "role": "user",
                        "contents": [{"type": "audio", "value": "first.wav"}],
                    },
                    {
                        "role": "user",
                        "contents": [{"type": "audio", "value": "second.wav"}],
                    },
                ]
            }
        )
    )

    assert result["responses"] == ["answer 1", "answer 2"]
    assert [message["role"] for message in conversations[1]] == [
        "user",
        "assistant",
        "user",
    ]
    assert conversations[1][1]["content"] == "answer 1"


def test_voicebench_multi_turn_judge_receives_history(monkeypatch):
    from audio_evals.registry import registry

    judge_requests = []

    class FakeModel:
        def inference(self, prompt, **kwargs):
            judge_requests.append(prompt)
            return "Rating: [[8]]"

    class FakePrompt:
        def load(self, **kwargs):
            return kwargs

    monkeypatch.setattr(registry, "get_model", lambda name: FakeModel())
    monkeypatch.setattr(registry, "get_prompt", lambda name: FakePrompt())

    score = VoiceBenchMultiTurnEvaluator("fake")(
        json.dumps({"responses": ["first answer", "second answer"]}),
        "",
        turns=["first question", "revise the answer"],
    )
    assert score["gpt_score"] == 8
    assert "Assistant turn 1: first answer" in judge_requests[1]["instruction"]


def test_offline_audio_evaluators_emit_catalog_metric_names(tmp_path):
    audio_path = tmp_path / "audio.wav"
    reference_path = tmp_path / "reference.wav"
    audio_path.touch()
    reference_path.touch()

    class FakePrompt:
        def load(self, **kwargs):
            return kwargs

    class FakeModel:
        def __init__(self, result):
            self.result = result

        def inference(self, prompt, **kwargs):
            return self.result

    seed = object.__new__(SeedTTSEvalASRWER)
    seed.lang = "en"
    seed.prompt = FakePrompt()
    seed.model = FakeModel("hello world")
    seed_score = seed(
        str(audio_path), str(reference_path), text="hello world"
    )
    assert seed_score["content_acc"] == 1
    assert seed_score["intelligibility"] == 1

    simo = object.__new__(Simo)
    simo.model = FakeModel(0.75)
    assert simo(str(audio_path), str(reference_path))["sim"] == 0.75
    routed_score = simo(
        str(audio_path), "target transcript", WavPath=str(reference_path)
    )
    assert routed_score["sim"] == 0.75
    assert routed_score["ref"] == str(reference_path)

    utmos = object.__new__(UTMOS)
    utmos.model = FakeModel(4.25)
    assert utmos(str(audio_path), "")["mos"] == 4.25

    dnsmos = object.__new__(DNSMOS)
    dnsmos.model = FakeModel('{"OVRL": 3.5, "SIG": 4.0}')
    assert dnsmos(str(audio_path), "")["dnsmos"] == 3.5


def test_bert_score_clamps_floating_point_overshoot():
    class FakeModel:
        def inference(self, prompt):
            return {"f1": 1.000066, "precision": 1.00001, "recall": 0.99999}

    evaluator = object.__new__(BertScoreEvaluator)
    evaluator.model = FakeModel()
    score = evaluator("same text", "same text")
    assert score["bert_score"] == 1.0
    assert score["bert_score_precision"] == 1.0
    assert score["bert_score_recall"] == pytest.approx(0.99999)


def test_speaker_attribution_and_choice_normalization():
    evaluator = SpeakerAttributionEvaluator()
    score = evaluator(
        '{"utterances":[{"id":"u1","speaker":"p1","text":"hello"},'
        '{"id":"u2","speaker":"p2","text":"world"}]}',
        "",
        reference_obj={
            "utterances": [
                {"id": "u1", "speaker": "a", "text": "hello"},
                {"id": "u2", "speaker": "b", "text": "world"},
            ]
        },
        language="en",
    )
    assert score["attribution_acc"] == 1
    assert score["cpcer%"] == 0

    choices = mesh_registry().get_evaluator("kimi-mmau-choices")
    score = choices(
        "B",
        "A men's locker room post-exercise",
        choices=[
            "A classroom",
            "A men's locker room post exercise",
        ],
    )
    assert score["match"] == 1


def test_local_huggingface_loader_applies_limit_before_conversion(monkeypatch):
    import audio_evals.dataset.huggingface as hf

    source = DatasetDict(
        {
            "first": Dataset.from_dict({"prompt": ["a", "b", "c"]}),
            "second": Dataset.from_dict({"prompt": ["d", "e", "f"]}),
        }
    )
    monkeypatch.setattr(hf, "load_from_disk", lambda _: source)
    monkeypatch.setattr(
        hf, "save_audio_to_local", lambda dataset, _, **kwargs: dataset
    )
    rows = hf.load_audio_hf_dataset(
        "local/test",
        split="all",
        local_path="/unused",
        col_aliases={"prompt": "question"},
        limit=4,
    )
    assert len(rows) == 4
    assert [row["question"] for row in rows] == ["a", "b", "c", "d"]


def test_huggingface_loader_forwards_data_files(monkeypatch):
    import audio_evals.dataset.huggingface as hf

    captured = {}
    source = Dataset.from_dict({"sentence": ["one", "two"]})

    def fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return source

    monkeypatch.setattr(hf, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        hf, "save_audio_to_local", lambda dataset, _, **kwargs: dataset
    )
    rows = hf.load_audio_hf_dataset(
        "remote/test",
        split="en",
        data_files={"en": "data/en-*.parquet"},
        limit=1,
    )

    assert len(rows) == 1
    assert captured == {
        "path": "remote/test",
        "split": "en",
        "data_files": {"en": "data/en-*.parquet"},
        "trust_remote_code": True,
    }


def test_local_huggingface_loader_saves_numbered_audio_columns(
    monkeypatch, tmp_path
):
    import audio_evals.dataset.huggingface as hf

    source = Dataset.from_list(
        [
            {
                "audio1": {"array": [0.0, 0.1], "sampling_rate": 16000},
                "audio2": {"array": [0.1, 0.0], "sampling_rate": 16000},
            }
        ]
    )
    writes = []
    original_write = hf.sf.write

    def tracking_write(path, array, sampling_rate, **kwargs):
        writes.append((path, list(array), sampling_rate, kwargs))
        return original_write(path, array, sampling_rate, **kwargs)

    monkeypatch.setattr(
        hf.sf,
        "write",
        tracking_write,
    )

    rows = list(hf.save_audio_to_local(source, str(tmp_path)))
    assert rows[0]["WavPath1"].endswith("0_audio1.wav")
    assert rows[0]["WavPath2"].endswith("0_audio2.wav")
    assert len(writes) == 2


def test_annotation_mapping_inheritance_and_knowledge_override():
    mapping = load_mapping("wenetspeech-test-meeting")
    assert mapping["defaults"]["scenario"] == "meeting"
    assert mapping["defaults"]["metrics"] == ["cer"]
    card = load_source_card("wenetspeech-test-meeting")
    labels = card["seed_facts"]["typical_labels"]
    assert labels["scenario"] == "meeting"
    assert labels["condition"]["spatial"] == "far_field"


def test_mesh_agg_uses_corpus_error_rate_bleu_and_chrf():
    asr_rows = [
        {
            "pred": "alpha bravo charlie delta echo foxtrot golf hotel wrong",
            "ref": "alpha bravo charlie delta echo foxtrot golf hotel india",
            "wer%": 11.111,
            "language": "en",
            "dataset_id": "asr",
        },
        {
            "pred": "wrong",
            "ref": "juliet",
            "wer%": 100.0,
            "language": "en",
            "dataset_id": "asr",
        },
    ]
    asr_result = MeshAgg(group_by=["dataset_id"])._agg(asr_rows)
    assert asr_result["overall/wer%"] == pytest.approx(20.0)
    assert asr_result["overall/wer%/aggregation"] == "corpus"

    translation_rows = [
        {
            "pred": "the cat is on the mat",
            "ref": "the cat is on the mat",
            "bleu": 0.0,
            "chrf": 0.0,
            "metric_names": "bleu|chrf",
            "dataset_id": "translation",
        },
        {
            "pred": "there is a cat",
            "ref": "a cat is there",
            "bleu": 0.0,
            "chrf": 0.0,
            "metric_names": "bleu|chrf",
            "dataset_id": "translation",
        },
    ]
    import sacrebleu

    predictions = [row["pred"] for row in translation_rows]
    references = [row["ref"] for row in translation_rows]
    result = MeshAgg(group_by=["dataset_id"])._agg(translation_rows)
    assert result["overall/bleu"] == pytest.approx(
        sacrebleu.corpus_bleu(predictions, [references]).score
    )
    assert result["overall/chrf"] == pytest.approx(
        sacrebleu.corpus_chrf(predictions, [references]).score
    )


def test_coco_is_deferred_normalized_and_cached(monkeypatch):
    score = Coco()(
        "A short caption. A second sentence that must be removed.",
        "fallback",
        reference_obj={"captions": ["reference one", "reference two"]},
    )
    assert score["caption_prediction"] == "A short caption."
    assert score["caption_references"] == ["reference one", "reference two"]
    assert "cider" not in score

    from audio_evals.lib import coco as coco_lib

    calls = []

    def fake_compute_caption(references, predictions):
        calls.append((references, predictions))
        return {"CIDEr": 0.25, "SPICE": 0.5}

    monkeypatch.setattr(coco_lib, "compute_caption", fake_compute_caption)
    rows = [
        {
            **score,
            "metric_names": "cider|spice",
            "dataset_id": "caption",
            "benchmark_id": "caption-benchmark",
        }
    ]
    result = MeshAgg(group_by=["dataset_id", "benchmark_id"])._agg(rows)
    assert result["overall/cider"] == 0.25
    assert result["overall/spice"] == 0.5
    assert result["benchmark_id/caption-benchmark/cider"] == 0.25
    assert len(calls) == 1


@pytest.mark.parametrize("source_schema", ["1.0", "2.0"])
def test_full_eval_clotho_restores_all_caption_references(tmp_path, source_schema):
    audio_path = tmp_path / "clotho.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)

    dataset_root = tmp_path / "clotho"
    manifest = dataset_root / "final" / "manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    row = {
        "dataset": "clotho",
        "task": "speech_understanding",
        "capability": "audio_caption",
        "scenario": "outdoor",
        "condition": {
            "acoustic": "clean",
            "spatial": "far_field",
            "speaker": "single_speaker",
            "device": "phone_mic",
            "interaction": "single_turn",
        },
        "metrics": ["cider"],
        "use_bucket": "diagnostic_evidence",
        "split": "test",
        "input": {"audio_path": str(audio_path)},
        "reference": {"caption": "first caption"},
        "sample_id": "clotho-1",
    }
    if source_schema == "2.0":
        row = adapt_v1_to_v2(row)
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    captions = ["first caption", "second caption", "third caption"]
    (dataset_root / "raw_index.jsonl").write_text(
        json.dumps({"audio_path": str(audio_path), "captions": captions}) + "\n",
        encoding="utf-8",
    )

    dataset = FullEvalDataset(
        dataset="clotho", benchmark_id="full-clotho", root=str(tmp_path)
    )
    assert dataset.count() == 1
    loaded = dataset.load()[0]
    assert loaded["schema_version"] == "runtime-sample/2.0"
    assert loaded["task"] == "audio_speech_understanding"
    assert loaded["reference_obj"]["captions"] == captions


def test_runtime_duration_sums_inputs_and_uses_tts_output(tmp_path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    for path, frames in ((first, 32000), (second, 16000)):
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * frames)

    assert _audio_duration_seconds(
        {
            "WavPath": str(first),
            "WavPath2": str(second),
            "input": {"audio_path": str(first), "audio_path_b": str(second)},
        }
    ) == pytest.approx(3.0)

    runtime = _measured_runtime(
        time.perf_counter(),
        {"task": "speech_output", "WavPath": str(first)},
        json.dumps({"audio": str(second), "text": "generated"}),
    )
    assert runtime["audio_duration_seconds"] == pytest.approx(1.0)
    assert runtime["rtf"] >= 0


def test_qwen_maps_token_limit_to_thinker_and_exposes_timeouts():
    model = Qwen3Omni.__new__(Qwen3Omni)
    Qwen3Omni.__init__.__wrapped__(
        model,
        path="/model",
        max_new_tokens=2048,
        startup_timeout=12,
        request_timeout=34,
    )
    assert model.command_args["thinker-max-new-tokens"] == 2048
    assert "max-new-tokens" not in model.command_args
    assert model._startup_timeout == 12
    assert model._request_timeout == 34

    released = []
    model.process = SimpleNamespace(poll=lambda: None)
    model._terminate_isolated_process = lambda: released.append(True)
    model.release()
    assert released == [True]


def test_qwen2_5_adapter_is_non_mutating_and_maps_runtime_options(tmp_path):
    audio_path = tmp_path / "source.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    prompt = [
        {
            "role": "user",
            "contents": [
                {
                    "type": "audio",
                    "value": {
                        "path": str(audio_path),
                        "start_time": 0.0,
                        "end_time": 0.25,
                    },
                },
                {"type": "text", "value": "answer"},
            ],
        }
    ]
    original = deepcopy(prompt)
    model = QwenOmni.__new__(QwenOmni)
    QwenOmni.__init__.__wrapped__(
        model,
        path="/model",
        max_new_tokens=42,
        startup_timeout=12,
        request_timeout=34,
    )
    parsed = model._parse_role_content(prompt[0])

    assert prompt == original
    assert parsed["content"][0]["audio"].endswith(".wav")
    assert Path(parsed["content"][0]["audio"]).is_file()
    assert model.command_args["thinker-max-new-tokens"] == 42
    assert model._startup_timeout == 12
    assert model._request_timeout == 34


def test_fun_audio_chat_preserves_message_and_audio_order():
    prompt = [
        {
            "role": "system",
            "contents": [{"type": "text", "value": "Use the context."}],
        },
        {
            "role": "user",
            "contents": [
                {"type": "audio", "value": "/tmp/one.wav"},
                {"type": "text", "value": "first"},
            ],
        },
        {
            "role": "assistant",
            "contents": [{"type": "text", "value": "history"}],
        },
        {
            "role": "user",
            "contents": [
                {"type": "text", "value": "second"},
                {"type": "audio", "value": "/tmp/two.wav"},
            ],
        },
    ]
    original = deepcopy(prompt)
    model = FunAudioChat.__new__(FunAudioChat)
    FunAudioChat.__init__.__wrapped__(model, path="/model", max_new_tokens=23)
    request = model._prepare_request(prompt)

    assert prompt == original
    assert request["audio_paths"] == ["/tmp/one.wav", "/tmp/two.wav"]
    assert request["conversation"][0]["content"].startswith(
        "You are asked to generate text tokens."
    )
    assert request["conversation"][1]["content"] == f"{AUDIO_TEMPLATE}\nfirst"
    assert request["conversation"][3]["content"] == f"second\n{AUDIO_TEMPLATE}"
    assert model.command_args["max-new-tokens"] == 23


def test_audio_agent_loader_rebuilds_history_and_multiple_tool_calls(tmp_path):
    bench = tmp_path / "sample-bench"
    audio_dir = bench / "real_audio" / "person1"
    audio_dir.mkdir(parents=True)
    for turn in range(3):
        with wave.open(str(audio_dir / f"turn_{turn:03d}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 400)
    rows = [
        {
            "file_name": "real_audio/person1/turn_000.wav",
            "turn_id": 0,
            "speaker": "person1",
            "golden_text": "first response",
            "required_function_call": None,
        },
        {
            "file_name": "real_audio/person1/turn_001.wav",
            "turn_id": 1,
            "speaker": "person1",
            "golden_text": "second response",
            "required_function_call": None,
        },
        {
            "file_name": "real_audio/person1/turn_002.wav",
            "turn_id": 2,
            "speaker": "person1",
            "golden_text": "done",
            "required_function_call": json.dumps(
                [
                    {"name": "register", "args": {"id": "one"}},
                    {"name": "register", "args": {"id": "two"}},
                ]
            ),
        },
    ]
    (bench / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )
    dataset = AudioAgentBenchDataset(
        benchmark_id="audio-agent-test",
        view="tool_call",
        root=str(tmp_path),
        max_history_turns=1,
    )
    loaded = dataset.load()[0]

    assert loaded["capability"] == "tool_call"
    assert loaded["reference_obj"]["expected_output"] == [
        {"tool": "register", "arguments": {"id": "one"}},
        {"tool": "register", "arguments": {"id": "two"}},
    ]
    messages = loaded["input"]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-2]["contents"][0]["value"] == "second response"
    assert "JSON array" in loaded["input"]["question"]
    assert loaded["offline_proxy"] == "contextual_turn_without_live_tool_runtime"


def test_ihbench_loader_materializes_embedded_audio(monkeypatch, tmp_path):
    root = tmp_path / "ihbench"
    root.mkdir()
    (root / "baseline.parquet").touch()
    (root / "conversations.parquet").touch()
    payload = b"RIFF" + b"\x00" * 64
    baseline = __import__("pandas").DataFrame(
        [
            {
                "conversation_id": "conversation-1",
                "turn": 1,
                "response": "continue now",
                "interrupting_user_message_index": 2,
            }
        ]
    )
    conversations = __import__("pandas").DataFrame(
        [
            {
                "conversation_id": "conversation-1",
                "system_message": "system",
                "domain": "test",
                "assistant_turn_1_transcript": "partial answer",
                "user_turn_1_audio": {
                    "path": "conversation-1/turn.wav",
                    "bytes": payload,
                },
                "user_turn_1_interruption_type": "backchannel",
                "turn_1_tf_rubric": "continue coherently",
                "turn_1_rq_rubrics": ["no repetition"],
            }
        ]
    )

    def fake_read_parquet(path, columns=None):
        return baseline if Path(path).name == "baseline.parquet" else conversations

    monkeypatch.setattr("pandas.read_parquet", fake_read_parquet)
    dataset = IHBenchDataset(
        benchmark_id="ihbench-test",
        root=str(root),
        cache_dir=str(tmp_path / "cache"),
    )
    loaded = dataset.load()[0]

    assert Path(loaded["WavPath"]).read_bytes() == payload
    assert loaded["reference_obj"]["conversation_context"] == [
        {"role": "assistant", "content": "partial answer"},
        {"role": "user", "content": ""},
    ]
    assert loaded["input"]["messages"][1]["contents"][0]["value"] == "partial answer"
    assert loaded["reference_obj"]["answer"] == "continue now"
    assert loaded["offline_proxy"] == (
        "native_tf_rq_rubric_without_streaming_timing"
    )


def test_qwen_output_pump_preserves_buffered_ready_and_response():
    import subprocess
    import sys

    child_code = """
import json
import sys

print('__QWEN3_OMNI_READY__ ready', flush=True)
request = sys.stdin.readline()
prefix = request.split('->', 1)[0] + '->'
print('buffered child log', flush=True)
print(prefix + json.dumps({'text': 'assistant buffered response'}), flush=True)
sys.stdin.readline()
"""
    model = Qwen3Omni.__new__(Qwen3Omni)
    Qwen3Omni.__init__.__wrapped__(
        model,
        path="/model",
        startup_timeout=5,
        request_timeout=5,
        request_log_interval=0.1,
    )
    model.process = subprocess.Popen(
        [sys.executable, "-u", "-c", child_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        model._wait_until_ready()
        assert model._request([])["text"] == "buffered response"
    finally:
        model.release()


def test_overall_completion_gate_checks_sample_count_and_failures(tmp_path):
    overall = tmp_path / "overall.json"
    overall.write_text(
        json.dumps({"sample_count": 2, "failure_rate": 0.0}), encoding="utf-8"
    )
    assert validate_overall_summary(overall, 2)["error_count"] == 0

    overall.write_text(
        json.dumps({"sample_count": 1, "failure_rate": 0.5}), encoding="utf-8"
    )
    report = validate_overall_summary(overall, 2)
    assert report["error_count"] == 2
