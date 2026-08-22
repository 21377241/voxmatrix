import json
import os
import signal
import time
import traceback
import threading
import wave
from concurrent.futures import as_completed, ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Union

from tqdm import tqdm

from audio_evals.agg.base import AggPolicy
from audio_evals.base import ScoreUnit
from audio_evals.dataset.dataset import Dataset
from audio_evals.evaluator.base import Evaluator
from audio_evals.models.model import Model
from audio_evals.process.base import Process
from audio_evals.prompt.base import Prompt
from audio_evals.recorder import Recorder
from audio_evals.utils import logger, merge_data4view
from mesh_eval.core.measurement import (
    consume_reported_runtime_metrics,
    normalize_runtime_metrics,
)


def extract_score(s: str):
    d = json.loads(s)
    return d["score"]


RUNTIME_FIELDS = (
    "latency_ms",
    "first_token_latency_ms",
    "first_audio_chunk_latency_ms",
    "rtf",
    "memory_peak_mb",
    "cpu_usage",
    "npu_usage",
    "power_watts",
    "timeout",
    "failure",
    "failure_stage",
    "failure_type",
)


def _audio_duration_seconds(doc):
    input_obj = doc.get("input") or {}
    for value in (doc.get("audio_duration_seconds"), doc.get("audio_duration")):
        try:
            duration = float(value)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            continue

    descriptors = {}

    def add_descriptor(path, start=None, end=None, duration=None):
        if not path:
            return
        key = str(path)
        previous = descriptors.get(key, {})
        descriptors[key] = {
            "start": start if start is not None else previous.get("start"),
            "end": end if end is not None else previous.get("end"),
            "duration": (
                duration if duration is not None else previous.get("duration")
            ),
        }

    primary_path = doc.get("WavPath") or input_obj.get("audio_path")
    if primary_path:
        add_descriptor(
            primary_path, input_obj.get("start_time"), input_obj.get("end_time")
        )
    second_path = doc.get("WavPath2") or input_obj.get("audio_path_b")
    if second_path:
        add_descriptor(second_path)
    for audio in [
        *(input_obj.get("audio") or []),
        *(input_obj.get("audios") or []),
    ]:
        if not isinstance(audio, dict):
            continue
        for key in ("duration_seconds", "duration"):
            try:
                duration = float(audio.get(key))
            except (TypeError, ValueError):
                continue
            if duration > 0:
                break
        else:
            duration = None
        path = audio.get("uri") or audio.get("path") or audio.get("audio_path")
        add_descriptor(
            path,
            audio.get("start_time"),
            audio.get("end_time"),
            duration,
        )

    durations = []
    for path, descriptor in descriptors.items():
        start = descriptor.get("start")
        end = descriptor.get("end")
        explicit_duration = descriptor.get("duration")
        if explicit_duration:
            durations.append(explicit_duration)
            continue
        if start is not None and end is not None:
            try:
                duration = float(end) - float(start)
            except (TypeError, ValueError):
                duration = 0.0
            if duration > 0:
                durations.append(duration)
                continue
        duration = _path_audio_duration_seconds(path)
        if duration is None:
            return None
        durations.append(duration)
    if durations:
        return sum(durations)

    for value in (input_obj.get("duration_seconds"), input_obj.get("duration")):
        try:
            duration = float(value)
            if duration > 0:
                return duration
        except (TypeError, ValueError):
            continue
    return None


def _path_audio_duration_seconds(audio_path):
    if not audio_path or not os.path.isfile(audio_path):
        return None
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        if info.samplerate > 0 and info.frames > 0:
            return info.frames / float(info.samplerate)
    except Exception:
        pass
    try:
        with wave.open(str(audio_path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except (wave.Error, OSError, EOFError, ZeroDivisionError):
        return None


def _generated_audio_path(output):
    if isinstance(output, dict):
        value = output.get("audio")
        return str(value) if value else None
    if not isinstance(output, str):
        return None
    if os.path.isfile(output):
        return output
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("audio"):
        return str(payload["audio"])
    return None


def _recorded_runtime(doc):
    runtime = {}
    eval_info = doc.get("eval_info") or {}
    inference_info = eval_info.get("inference") or {}
    candidates = [
        doc.get("runtime"),
        inference_info.get("runtime") if isinstance(inference_info, dict) else None,
        inference_info,
        eval_info.get("eval"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        for field in RUNTIME_FIELDS:
            if candidate.get(field) is not None and field not in runtime:
                runtime[field] = candidate[field]
    return normalize_runtime_metrics(
        runtime, str(doc.get("measurement_protocol") or "ondevice_audio@1")
    )


def _is_audio_output_protocol(doc):
    resolved = doc.get("resolved_protocol") or {}
    return (
        resolved.get("output_schema") == "audio@1"
        or resolved.get("parser") == "generated_audio@1"
        or doc.get("answer_type") == "audio"
        # V1 compatibility only; V2 relies on the resolved protocol above.
        or doc.get("task") == "speech_output"
    )


def _measured_runtime(started, doc, output=None):
    latency_ms = (time.perf_counter() - started) * 1000
    runtime = {"latency_ms": latency_ms}
    if _is_audio_output_protocol(doc):
        duration = _path_audio_duration_seconds(_generated_audio_path(output))
    else:
        duration = _audio_duration_seconds(doc)
    if duration:
        runtime["audio_duration_seconds"] = duration
        runtime["rtf"] = latency_ms / 1000 / duration
    return normalize_runtime_metrics(
        runtime, str(doc.get("measurement_protocol") or "ondevice_audio@1")
    )


def _tag_exception(exc: Exception, stage: str, runtime: Dict[str, Any] = None):
    try:
        if not getattr(exc, "_ultraeval_failure_stage", ""):
            exc._ultraeval_failure_stage = stage
        if runtime:
            previous = getattr(exc, "_ultraeval_runtime", {})
            exc._ultraeval_runtime = {**previous, **runtime}
    except Exception:
        pass
    return exc


def _predict_with_runtime(predictor, prompt, doc):
    """Invoke one request under the versioned framework/adapter contract."""
    started = time.perf_counter()
    reported = {}
    try:
        measured_call = getattr(predictor, "inference_with_runtime", None)
        if callable(measured_call):
            result = measured_call(prompt)
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError(
                    "inference_with_runtime must return (output, runtime_metrics)"
                )
            output, reported = result
        else:
            output = predictor.inference(prompt)
            reported = consume_reported_runtime_metrics(predictor)
    except Exception as exc:
        runtime = _measured_runtime(started, doc)
        runtime.update(
            {
                "failure": 1,
                "timeout": int(isinstance(exc, TimeoutError)),
            }
        )
        _tag_exception(exc, "inference", runtime)
        raise

    profile_id = str(doc.get("measurement_protocol") or "ondevice_audio@1")
    runtime = normalize_runtime_metrics(reported, profile_id)
    # The framework wall-clock boundary is authoritative for latency/derived RTF.
    runtime.update(_measured_runtime(started, doc, output))
    return output, runtime


def _failure_type(exc: Exception, stage: str) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, MemoryError):
        return "resource_exhausted"
    if isinstance(exc, FileNotFoundError):
        return "input_unavailable"
    if isinstance(exc, OSError):
        return "infrastructure"
    if isinstance(exc, (ValueError, TypeError, KeyError)) and stage in {
        "prompt",
        "post_process",
        "evaluation",
    }:
        return "contract_violation"
    if stage == "inference":
        return "model_runtime"
    return "internal"


def _event_dimensions(doc):
    fields = (
        "sample_id",
        "benchmark_id",
        "dataset_id",
        "dataset",
        "task",
        "capability",
        "scenario",
        "use_bucket",
        "split",
        "language",
        "measurement_protocol",
        "measurement_protocol_hash",
    )
    data = {field: doc[field] for field in fields if doc.get(field) not in (None, "", [])}
    if doc.get("schema_version") == "runtime-sample/2.0":
        data["sample_schema_version"] = doc["schema_version"]
        scenario = doc.get("scenario") or {}
        if isinstance(scenario, dict):
            for field in ("primary", "secondary"):
                if scenario.get(field) not in (None, "", []):
                    data[f"scenario__{field}"] = scenario[field]
        for field, value in (doc.get("metadata") or {}).items():
            if value not in (None, "", []):
                data[f"metadata__{field}"] = (
                    "|".join(str(item) for item in value)
                    if isinstance(value, list)
                    else value
                )
        if doc.get("provenance"):
            data["_slice_provenance"] = doc["provenance"]
        from mesh_eval.core.subset_manifest import flatten_subset_profile

        data.update(flatten_subset_profile(doc.get("subset_profile")))
        for field in (
            "subset_manifest_record_id",
            "subset_manifest_schema_version",
            "subset_manifest_sha256",
            "source_benchmark_id",
            "subset_id",
            "source_protocol_id",
            "source_metric_names",
            "mapping_status",
            "resource_status",
            "annotation_confidence",
        ):
            if doc.get(field) not in (None, "", []):
                data[field] = doc[field]
    for group, value in (doc.get("condition") or {}).items():
        data[f"condition__{group}"] = value
    return data


def _failure_score(doc, exc: Exception, error_traceback: str) -> Dict[str, Any]:
    stage = str(getattr(exc, "_ultraeval_failure_stage", "") or "unknown")
    runtime = dict(getattr(exc, "_ultraeval_runtime", {}) or {})
    runtime.update(
        {
            "failure": 1,
            "timeout": int(isinstance(exc, TimeoutError)),
            "failure_stage": stage,
            "failure_type": _failure_type(exc, stage),
        }
    )
    runtime = normalize_runtime_metrics(
        runtime, str(doc.get("measurement_protocol") or "ondevice_audio@1")
    )
    return {
        **_event_dimensions(doc),
        **runtime,
        "error_class": type(exc).__name__,
        "error_message": str(exc),
        "info": error_traceback,
    }


def load_dataset_slice(dataset, limit=0, timeout=0, offset=0):
    """Load one dataset shard, preferring an implementation-level seek."""
    # Historical callers use ``None`` for an unbounded load.  Normalize it at
    # this boundary so both the old ``Dataset.load`` API and the new slice API
    # retain the same behaviour.
    limit = 0 if limit is None else limit
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit < 0:
        raise ValueError("limit must be non-negative")

    def load():
        loader = getattr(dataset, "load_slice", None)
        if callable(loader):
            return loader(offset=offset, limit=limit)
        load_limit = offset + limit if limit else 0
        data = dataset.load(load_limit)
        return data[offset : offset + limit if limit else None]

    if not timeout:
        return load()
    if timeout < 0:
        raise ValueError("dataset load timeout must be non-negative")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("dataset load timeout requires the main thread")
    if not hasattr(signal, "SIGALRM"):
        logger.warning("dataset load timeout is not supported on this platform")
        return load()

    def handle_timeout(_signum, _frame):
        raise TimeoutError(f"dataset loading exceeded {timeout} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return load()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] or previous_timer[1]:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


class EvalTask:

    def __init__(
        self,
        dataset: Dataset,
        prompt: Prompt,
        predictor: Model,
        evaluator: Evaluator,
        post_process: List[Process],
        agg: AggPolicy,
        recorder: Recorder,
        event_id_offset: int = 0,
        run_context: Dict[str, Any] = None,
    ):
        self.dataset = dataset
        self.prompt = prompt
        self.predictor = predictor
        self.evaluator = evaluator
        self.post_process = post_process
        self.agg = agg
        self.recorder = recorder
        self.event_id_offset = event_id_offset
        from mesh_eval.core.measurement import resolve_measurement_protocol

        measurement = resolve_measurement_protocol("ondevice_audio@1")
        self.run_context = {
            "measurement_protocol": measurement["id"],
            "measurement_protocol_hash": measurement["hash"],
            "request_mode": "unspecified",
            **dict(run_context or {}),
        }

    def _event_id(self, index):
        return self.event_id_offset + index

    def _record_event(self, event_type, event_id, data, doc=None):
        if isinstance(doc, dict) and doc.get("schema_version") == "runtime-sample/2.0":
            from mesh_eval.core.schema_v2 import result_event_v2

            event = result_event_v2(event_type, event_id, doc, data)
        else:
            event = {"type": event_type, "id": event_id, "data": data}
            if isinstance(doc, dict) and doc.get("run_context"):
                event["run_context"] = dict(doc["run_context"])
        self.recorder.add(event)

    def _load_dataset(self, limit, timeout=0, offset=0):
        return load_dataset_slice(dataset=self.dataset, limit=limit, timeout=timeout, offset=offset)

    def _eval(
        self, idx, prompt: Union[str, List[Dict[str, str]]], reference: str, **kwargs
    ) -> Tuple[ScoreUnit, str]:
        event_id = self._event_id(idx)
        self._record_event("prompt", event_id, {"content": prompt}, kwargs)
        is_replay = "eval_info" in kwargs and "inference" in kwargs["eval_info"]
        runtime = _recorded_runtime(kwargs)
        if is_replay:
            try:
                output = kwargs["eval_info"]["inference"]["content"]
            except Exception as exc:
                _tag_exception(exc, "inference")
                raise
        else:
            output, measured = _predict_with_runtime(self.predictor, prompt, kwargs)
            runtime.update(measured)
        inference_data = {"content": output}
        if runtime:
            inference_data["runtime"] = runtime
            inference_data.update(runtime)
        self._record_event("inference", event_id, inference_data, kwargs)

        if "eval_info" in kwargs and "post_process" in kwargs["eval_info"]:
            try:
                output = kwargs["eval_info"]["post_process"]["content"]
            except Exception as exc:
                _tag_exception(exc, "post_process")
                raise
        else:
            for p in self.post_process:
                try:
                    output = p(output)
                except Exception as exc:
                    _tag_exception(exc, "post_process")
                    raise
        self._record_event("post_process", event_id, {"content": output}, kwargs)

        if "eval_info" in kwargs and "eval" in kwargs["eval_info"]:
            score = dict(kwargs["eval_info"]["eval"])
        else:
            eval_kwargs = dict(kwargs)
            if runtime:
                eval_kwargs["runtime"] = runtime
            try:
                score = dict(self.evaluator(output, reference, **eval_kwargs))
            except Exception as exc:
                _tag_exception(exc, "evaluation")
                raise
        for field, value in runtime.items():
            score.setdefault(field, value)
        for field, value in _event_dimensions(kwargs).items():
            score.setdefault(field, value)
        score.setdefault("failure", 0)
        score.setdefault("timeout", 0)
        self._record_event("eval", event_id, score, kwargs)
        return score, output

    def _run(self, i, doc):
        """单个任务处理逻辑"""
        try:
            try:
                real_prompt = self.prompt.load(**doc)
            except Exception as exc:
                _tag_exception(exc, "prompt")
                raise
            score, ans = self._eval(
                i, real_prompt, doc.get(self.dataset.ref_col, ""), **doc
            )
            return i, score, ans, 0
        except Exception as exc:
            error_traceback = traceback.format_exc()
            error_data = _failure_score(doc, exc, error_traceback)
            self._record_event("error", self._event_id(i), error_data, doc)
            print(error_traceback)
            return i, error_data, None, 1

    def _inference_only(self, i, doc):
        """仅执行推理；后处理延迟到模型释放后的第二阶段。"""
        try:
            try:
                real_prompt = self.prompt.load(**doc)
            except Exception as exc:
                _tag_exception(exc, "prompt")
                raise
            event_id = self._event_id(i)
            self._record_event("prompt", event_id, {"content": real_prompt}, doc)
            
            is_replay = "eval_info" in doc and "inference" in doc["eval_info"]
            runtime = _recorded_runtime(doc)
            if is_replay:
                try:
                    output = doc["eval_info"]["inference"]["content"]
                except Exception as exc:
                    _tag_exception(exc, "inference")
                    raise
            else:
                output, measured = _predict_with_runtime(
                    self.predictor, real_prompt, doc
                )
                runtime.update(measured)
            if runtime:
                doc["runtime"] = runtime
            inference_data = {"content": output}
            if runtime:
                inference_data["runtime"] = runtime
                inference_data.update(runtime)
            self._record_event("inference", event_id, inference_data, doc)
            
            return i, output, doc, 0, None
        except Exception as exc:
            error_traceback = traceback.format_exc()
            error_data = _failure_score(doc, exc, error_traceback)
            self._record_event(
                "error",
                self._event_id(i),
                error_data,
                doc,
            )
            print(error_traceback)
            return i, None, doc, 1, error_data

    def _evaluate_only(self, i, output, doc):
        """在推理模型释放后执行后处理和评测。"""
        try:
            if "eval_info" in doc and "post_process" in doc["eval_info"]:
                try:
                    output = doc["eval_info"]["post_process"]["content"]
                except Exception as exc:
                    _tag_exception(exc, "post_process")
                    raise
            else:
                for process in self.post_process:
                    try:
                        output = process(output)
                    except Exception as exc:
                        _tag_exception(exc, "post_process")
                        raise
            self._record_event(
                "post_process", self._event_id(i), {"content": output}, doc
            )

            reference = doc.get(self.dataset.ref_col, "")
            if "eval_info" in doc and "eval" in doc["eval_info"]:
                score = dict(doc["eval_info"]["eval"])
            else:
                try:
                    score = dict(self.evaluator(output, reference, **doc))
                except Exception as exc:
                    _tag_exception(exc, "evaluation")
                    raise
            for field, value in _recorded_runtime(doc).items():
                score.setdefault(field, value)
            for field, value in _event_dimensions(doc).items():
                score.setdefault(field, value)
            score.setdefault("failure", 0)
            score.setdefault("timeout", 0)
            self._record_event("eval", self._event_id(i), score, doc)
            return i, score, output, 0
        except Exception as exc:
            error_traceback = traceback.format_exc()
            error_data = _failure_score(doc, exc, error_traceback)
            self._record_event(
                "error",
                self._event_id(i),
                error_data,
                doc,
            )
            print(error_traceback)
            return i, error_data, output, 1

    def _release_predictor(self):
        """释放推理模型占用的 GPU 显存"""
        try:
            # 尝试调用模型的释放方法（如果有的话）
            if hasattr(self.predictor, 'release') and callable(self.predictor.release):
                self.predictor.release()
                # 删除模型引用
                del self.predictor
                self.predictor = None
                print("Predictor released successfully, GPU memory freed.")
            else:
                predictor_type = type(self.predictor).__name__
                print(f"Predictor ({predictor_type}) does not have a release method, skipping GPU memory release.")
        except Exception as e:
            print(f"Warning: Failed to release GPU memory: {e}")

    def run_two_phase(
        self,
        limit=None,
        rand_size=None,
        max_workers=1,
        dataset_load_timeout=0,
        dataset_offset=0,
        evaluation_workers=1,
    ) -> Tuple[ScoreUnit, List[ScoreUnit], List[str]]:
        """
        两阶段执行：先并发推理，后串行评测
        :param limit: 限制数据条数
        :param rand_size: 随机采样数量
        :param max_workers: 推理阶段的并发数
        :return: 聚合结果, 各条评分, 各条输出
        """
        quiz = self.load_quiz(
            limit, rand_size, dataset_load_timeout, dataset_offset
        )

        inference_state = self.run_inference_phase(quiz, max_workers)

        # The standalone two-phase path owns its predictor.  A multi-benchmark
        # EvaluationSession calls the two public phase methods directly and
        # releases the shared predictor only after every benchmark inferred.
        print("Releasing model GPU memory...")
        self._release_predictor()

        return self.run_evaluation_phase(
            quiz,
            inference_state,
            evaluation_workers=evaluation_workers,
        )

    def load_quiz(
        self,
        limit=None,
        rand_size=None,
        dataset_load_timeout=0,
        dataset_offset=0,
    ):
        """Load and sample the documents used by either execution mode."""
        quiz = self._load_dataset(limit, dataset_load_timeout, dataset_offset)
        from mesh_eval.core.schema import normalize_record

        runtime_quiz = []
        for raw_doc in quiz:
            doc = dict(raw_doc)
            if doc.get("schema_version") == "2.0":
                doc = normalize_record(
                    doc,
                    ref_col=self.dataset.ref_col,
                    default_task=self.dataset.task_name,
                    run_context=self.run_context,
                )
            else:
                context = dict(doc.get("run_context") or {})
                context.update(self.run_context)
                doc["run_context"] = context
                doc.setdefault(
                    "measurement_protocol", context["measurement_protocol"]
                )
                doc.setdefault(
                    "measurement_protocol_hash",
                    context["measurement_protocol_hash"],
                )
            runtime_quiz.append(doc)
        quiz = runtime_quiz
        if limit:
            quiz = quiz[:limit]
        if rand_size:
            import random
            quiz = random.sample(quiz, rand_size)
        return quiz

    def run_inference_phase(self, quiz, max_workers=1):
        """Run prompts/inference while keeping post-processing deferred."""
        print("Phase 1: Running inference...")
        inference_results = [None] * len(quiz)
        inference_docs = [None] * len(quiz)
        inference_error_scores = [None] * len(quiz)
        inference_error_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = [
                executor.submit(self._inference_only, i, doc) for i, doc in enumerate(quiz)
            ]
            for future in tqdm(as_completed(future_to_index), total=len(quiz), desc="Inference"):
                index, output, doc, has_error, error_score = future.result()
                inference_error_count += has_error
                if has_error:
                    inference_error_scores[index] = error_score
                else:
                    inference_results[index] = output
                    inference_docs[index] = doc
        return {
            "results": inference_results,
            "docs": inference_docs,
            "error_scores": inference_error_scores,
            "error_count": inference_error_count,
        }

    def run_evaluation_phase(self, quiz, inference_state, evaluation_workers=1):
        """Post-process and score a completed inference phase."""
        if evaluation_workers < 1:
            raise ValueError("evaluation_workers must be >= 1")
        print("Phase 2: Running evaluation...")
        res = list(inference_state.get("error_scores") or [None] * len(quiz))
        answers = [None] * len(quiz)
        eval_error_count = 0
        inference_results = inference_state["results"]
        inference_docs = inference_state["docs"]
        inference_error_count = inference_state["error_count"]
        
        from audio_evals.registry import registry
        self.evaluator = registry.get_evaluator(self.evaluator)

        # 构建需要评测的任务列表
        eval_tasks = [
            (i, inference_results[i], inference_docs[i])
            for i in range(len(quiz))
            if inference_docs[i] is not None and res[i] is None
        ]
        
        with ThreadPoolExecutor(max_workers=evaluation_workers) as executor:
            future_to_index = [
                executor.submit(self._evaluate_only, i, output, doc)
                for i, output, doc in eval_tasks
            ]
            for future in tqdm(as_completed(future_to_index), total=len(eval_tasks), desc="Evaluation"):
                index, score, output, has_error = future.result()
                eval_error_count += has_error
                if score is not None:
                    res[index] = score
                    if not has_error and output is not None:
                        answers[index] = output

        res, answers = [item for item in res if item is not None], [
            item for item in answers if item is not None
        ]
        merge_data4view(
            quiz,
            self.recorder.name,
            self.recorder.name.replace(".jsonl", ".xlsx"),
            id_offset=self.event_id_offset,
        )
        final_res = self.agg(res)
        total_error = inference_error_count + eval_error_count
        failure_rate = total_error / len(quiz) if quiz else 0.0
        final_res["failure_rate"] = failure_rate
        final_res["fail_rate(%d)"] = failure_rate * 100
        final_res["inference_fail_count"] = inference_error_count
        final_res["eval_fail_count"] = eval_error_count
        return final_res, res, answers

    @lru_cache(maxsize=None)
    def run(
        self,
        limit=None,
        rand_size=None,
        max_workers=1,
        two_phase=False,
        dataset_load_timeout=0,
        dataset_offset=0,
        evaluation_workers=1,
    ) -> Tuple[ScoreUnit, List[ScoreUnit], List[str]]:
        """
        eval
        :param limit: 限制数据条数
        :param rand_size: 随机采样数量
        :param max_workers: 并发数
        :param two_phase: 是否使用两阶段模式（先推理后评测）
        :return: 聚合结果, 各条评分, 各条输出
        """
        if two_phase:
            return self.run_two_phase(
                limit,
                rand_size,
                max_workers,
                dataset_load_timeout,
                dataset_offset,
                evaluation_workers,
            )

        quiz = self.load_quiz(
            limit, rand_size, dataset_load_timeout, dataset_offset
        )

        res = [None] * len(quiz)
        answers = [None] * len(quiz)
        error_count = 0
        from audio_evals.registry import registry
        self.evaluator = registry.get_evaluator(self.evaluator)

        # 使用进程池并控制最大并发量
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = [
                executor.submit(self._run, i, doc) for i, doc in enumerate(quiz)
            ]
            for future in tqdm(as_completed(future_to_index), total=len(quiz)):
                index, score, ans, has_error = future.result()
                error_count += has_error
                if score is not None:
                    res[index] = score
                if not has_error and ans is not None:
                    answers[index] = ans

        res, answers = [item for item in res if item is not None], [
            item for item in answers if item is not None
        ]
        merge_data4view(
            quiz,
            self.recorder.name,
            self.recorder.name.replace(".jsonl", ".xlsx"),
            id_offset=self.event_id_offset,
        )
        final_res = self.agg(res)
        failure_rate = error_count / len(quiz) if quiz else 0.0
        final_res["failure_rate"] = failure_rate
        final_res["fail_rate(%d)"] = failure_rate * 100
        return final_res, res, answers
