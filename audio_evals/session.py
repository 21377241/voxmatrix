"""Run multiple compatible benchmarks under one long-lived predictor."""

import argparse
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import yaml

from audio_evals.dataset.dataset import InMemoryDataset
from audio_evals.dataset.prepared import PreparedAudioJsonl
from audio_evals.eval_task import EvalTask
from audio_evals.main import build_run_context, create_predictor, preload_dataset
from audio_evals.recorder import Recorder
from audio_evals.registry import registry

logger = logging.getLogger(__name__)


@dataclass
class PreparedBenchmark:
    benchmark_id: str
    dataset: Any
    task_cfg: Any
    save_path: str
    overall_path: str
    event_id_offset: int = 0
    two_phase: bool = False
    evaluation_workers: int = 1


def _atomic_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "benchmark"


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        if Path(path).suffix.lower() == ".json":
            config = json.load(handle)
        else:
            config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("suite config must be an object")
    return config


def _subset_manifest_options(config: Dict[str, Any]):
    raw = config.get("subset_manifest")
    if raw in (None, "", False):
        return None, None
    if isinstance(raw, str):
        raw = {"path": raw}
    if not isinstance(raw, dict):
        raise ValueError("subset_manifest must be a path or object")
    path = str(raw.get("path") or "")
    if not path:
        raise ValueError("subset_manifest.path is required")
    aliases = raw.get("benchmark_aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError("subset_manifest.benchmark_aliases must be an object")
    excluded = {"meeting_summary", "todo_extraction"}
    configured_exclusions = raw.get("exclude_capabilities") or []
    if not isinstance(configured_exclusions, list) or not all(
        isinstance(item, str) and item for item in configured_exclusions
    ):
        raise ValueError("subset_manifest.exclude_capabilities must be a string list")
    excluded.update(configured_exclusions)

    from mesh_eval.core.subset_manifest import SubsetManifestCatalog

    catalog = SubsetManifestCatalog(
        path,
        excluded_capabilities=excluded,
        benchmark_aliases=aliases,
    )
    return catalog, {
        "excluded_capabilities": excluded,
        "allow_unmatched": bool(raw.get("allow_unmatched", False)),
    }


def _enrich_preloaded_dataset(dataset, benchmark_id, spec, catalog, options):
    from mesh_eval.core.subset_manifest import (
        SubsetManifestError,
        enrich_canonical_sample,
    )

    rows = []
    excluded_count = 0
    unmatched_count = 0
    fixed_record_id = str(spec.get("subset_record_id") or "")
    fixed_record = catalog.find(record_id=fixed_record_id) if fixed_record_id else None
    for index, sample in enumerate(dataset.load()):
        capability = str(sample.get("capability") or "")
        if capability in options["excluded_capabilities"]:
            excluded_count += 1
            continue
        try:
            record = fixed_record or catalog.find_for_sample(
                sample,
                benchmark_id=str(
                    spec.get("manifest_benchmark_id")
                    or spec.get("source_benchmark_id")
                    or ""
                ),
                subset_id=str(spec.get("subset_id") or ""),
                source_protocol_id=str(spec.get("source_protocol_id") or ""),
            )
            rows.append(enrich_canonical_sample(sample, record, catalog))
        except SubsetManifestError as exc:
            if not options["allow_unmatched"]:
                raise ValueError(
                    f"benchmark {benchmark_id} sample {index} could not be joined "
                    f"to subset manifest: {exc}"
                ) from exc
            rows.append(sample)
            unmatched_count += 1
    return (
        InMemoryDataset.from_dataset(dataset, rows),
        excluded_count,
        unmatched_count,
    )


def _resolve_save_path(spec: Dict[str, Any], output_root: str, benchmark_id: str) -> str:
    save_path = str(spec.get("save") or "")
    if not save_path:
        save_path = os.path.join(output_root, f"{_safe_name(benchmark_id)}.jsonl")
    elif not os.path.isabs(save_path):
        save_path = os.path.join(output_root, save_path)
    if not save_path.endswith(".jsonl"):
        save_path += ".jsonl"
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    return os.path.abspath(save_path)


def prepare_benchmarks(
    config: Dict[str, Any], model_name: str, output_root: str
) -> List[PreparedBenchmark]:
    """Load/validate every benchmark before the shared predictor occupies GPUs."""
    raw_benchmarks = config.get("benchmarks")
    if not isinstance(raw_benchmarks, list) or not raw_benchmarks:
        raise ValueError("suite config requires a non-empty benchmarks list")
    prepared: List[PreparedBenchmark] = []
    subset_catalog, subset_options = _subset_manifest_options(config)
    seen_ids = set()
    for index, spec in enumerate(raw_benchmarks):
        if not isinstance(spec, dict):
            raise ValueError(f"benchmarks[{index}] must be an object")
        dataset_name = str(spec.get("dataset") or "")
        if not dataset_name:
            raise ValueError(f"benchmarks[{index}] is missing dataset")
        benchmark_id = str(spec.get("benchmark_id") or dataset_name)
        if benchmark_id in seen_ids:
            raise ValueError(f"duplicate benchmark_id: {benchmark_id}")
        seen_ids.add(benchmark_id)
        if spec.get("model") and spec["model"] != model_name:
            raise ValueError(
                f"benchmark {benchmark_id} requests model {spec['model']!r}, "
                f"but the session model is {model_name!r}"
            )

        dataset = registry.get_dataset(dataset_name)
        if dataset is None:
            raise KeyError(f"dataset is not registered: {dataset_name}")
        if spec.get("prepared_manifest"):
            dataset = PreparedAudioJsonl(
                str(spec["prepared_manifest"]),
                default_task=dataset.task_name,
                ref_col=dataset.ref_col,
                col_aliases=dataset.col_aliases,
            )
        if spec.get("resume"):
            dataset = dataset.resume_from(str(spec["resume"]))
        if spec.get("dataset_ref_col"):
            dataset.reset_ref_col(str(spec["dataset_ref_col"]))
        if spec.get("inf_file"):
            dataset = dataset.load_inf_file(str(spec["inf_file"]))

        task_name = str(spec.get("task") or dataset.task_name)
        task_cfg = registry.get_eval_task(task_name)
        if task_cfg is None:
            raise KeyError(f"eval task is not registered: {task_name}")
        task_cfg.model = model_name
        for field in ("prompt", "evaluator", "agg", "post_process"):
            if field in spec:
                setattr(task_cfg, field, spec[field])

        offset = int(spec.get("offset") or 0)
        limit = int(spec.get("limit") or 0)
        rand_size = int(spec.get("rand") or 0)
        timeout = int(spec.get("dataset_load_timeout") or 0)
        if min(offset, limit, rand_size, timeout) < 0:
            raise ValueError(f"benchmark {benchmark_id} has a negative load option")
        evaluation_workers = int(spec.get("evaluation_workers") or 1)
        if evaluation_workers < 1:
            raise ValueError(
                f"benchmark {benchmark_id} evaluation_workers must be >= 1"
            )
        dataset = preload_dataset(
            dataset,
            offset=offset,
            limit=limit,
            rand_size=rand_size,
            timeout=timeout,
        )
        if subset_catalog is not None and spec.get("attach_subset_profile", True):
            dataset, excluded_count, unmatched_count = _enrich_preloaded_dataset(
                dataset,
                benchmark_id,
                spec,
                subset_catalog,
                subset_options,
            )
            logger.info(
                "Attached subset profiles for benchmark %s: rows=%d excluded=%d "
                "unmatched=%d",
                benchmark_id,
                len(dataset.rows),
                excluded_count,
                unmatched_count,
            )
            if not dataset.rows:
                logger.info(
                    "Skipping benchmark %s because no rows remain after subset filtering",
                    benchmark_id,
                )
                continue
        save_path = _resolve_save_path(spec, output_root, benchmark_id)
        prepared.append(
            PreparedBenchmark(
                benchmark_id=benchmark_id,
                dataset=dataset,
                task_cfg=task_cfg,
                save_path=save_path,
                overall_path=save_path[: -len(".jsonl")] + "-overall.json",
                event_id_offset=offset,
                two_phase=bool(spec.get("two_phase", False)),
                evaluation_workers=evaluation_workers,
            )
        )
    if not prepared:
        raise ValueError("suite has no runnable benchmarks after subset filtering")
    return prepared


class EvaluationSession:
    """Own a predictor above individual benchmark recorder/report lifetimes."""

    def __init__(self, predictor, inference_workers: int):
        if inference_workers < 1:
            raise ValueError("inference_workers must be >= 1")
        self.predictor = predictor
        self.inference_workers = inference_workers
        self._released = False

    def run(self, benchmarks: List[PreparedBenchmark]) -> Dict[str, Any]:
        results = {}

        def create_task(benchmark):
            cfg = benchmark.task_cfg
            return EvalTask(
                dataset=benchmark.dataset,
                prompt=registry.get_prompt(cfg.prompt),
                predictor=self.predictor,
                evaluator=cfg.evaluator,
                post_process=[registry.get_process(name) for name in cfg.post_process],
                agg=registry.get_agg(cfg.agg),
                recorder=Recorder(benchmark.save_path),
                event_id_offset=benchmark.event_id_offset,
                run_context=build_run_context(
                    benchmark_id=benchmark.benchmark_id,
                    model_name=str(
                        getattr(cfg, "model", type(self.predictor).__name__)
                    ),
                    predictor=self.predictor,
                    inference_workers=self.inference_workers,
                    evaluation_workers=benchmark.evaluation_workers,
                ),
            )

        def save_result(benchmark, result):
            _atomic_json(benchmark.overall_path, result[0])
            results[benchmark.benchmark_id] = {
                "status": "ok",
                "save": benchmark.save_path,
                "overall": benchmark.overall_path,
                "result": result[0],
            }

        try:
            if any(benchmark.two_phase for benchmark in benchmarks):
                # Resource-phased suite: complete every model inference first,
                # release the shared GPU predictor once, then load/run all
                # post-processors and evaluators (for example speech ASR).
                pending = []
                for benchmark in benchmarks:
                    logger.info(
                        "Running deferred inference for benchmark %s",
                        benchmark.benchmark_id,
                    )
                    task = create_task(benchmark)
                    quiz = task.load_quiz(0, 0, 0, 0)
                    state = task.run_inference_phase(quiz, self.inference_workers)
                    pending.append((benchmark, task, quiz, state))
                self.release()
                for benchmark, task, quiz, state in pending:
                    result = task.run_evaluation_phase(
                        quiz,
                        state,
                        evaluation_workers=benchmark.evaluation_workers,
                    )
                    save_result(benchmark, result)
                return results

            for benchmark in benchmarks:
                logger.info("Running benchmark %s in shared session", benchmark.benchmark_id)
                task = create_task(benchmark)
                result = task.run(
                    limit=0,
                    rand_size=0,
                    max_workers=self.inference_workers,
                    two_phase=False,
                    dataset_load_timeout=0,
                    dataset_offset=0,
                )
                save_result(benchmark, result)
            return results
        finally:
            self.release()

    def release(self):
        if self._released:
            return
        self._released = True
        release = getattr(self.predictor, "release", None)
        if callable(release):
            release()


def _pool_options(config: Dict[str, Any]) -> SimpleNamespace:
    options = config.get("model_pool") or {}
    if not isinstance(options, dict):
        raise ValueError("model_pool must be an object")
    mode = str(options.get("use_model_pool", config.get("use_model_pool", "auto")))
    if mode not in {"auto", "on", "off"}:
        raise ValueError("use_model_pool must be auto, on, or off")
    return SimpleNamespace(
        use_model_pool=mode,
        workers=options.get("workers"),
        replicas=int(options.get("replicas") or 0),
        gpus_per_replica=int(options.get("gpus_per_replica") or 1),
        inference_workers=int(options.get("inference_workers") or 0),
        allow_gpu_sharing=bool(options.get("allow_gpu_sharing", False)),
        model_startup_workers=int(options.get("model_startup_workers") or 0),
        model_ready_policy=str(options.get("model_ready_policy") or "auto"),
    )


def _suite_aggregation_options(
    config: Dict[str, Any], output_root: str
) -> Optional[Dict[str, Any]]:
    raw = config.get("suite_aggregation")
    if raw in (None, False):
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("suite_aggregation must be a boolean or object")
    if raw.get("enabled", True) is False:
        return None
    report_format = str(raw.get("report_format") or "both")
    if report_format not in {"flat", "nested", "both"}:
        raise ValueError("suite_aggregation.report_format must be flat, nested, or both")
    min_slice_size = int(raw.get("min_slice_size", 5))
    if min_slice_size < 1:
        raise ValueError("suite_aggregation.min_slice_size must be >= 1")
    excluded = raw.get("exclude_capabilities") or [
        "meeting_summary",
        "todo_extraction",
    ]
    metric_fields = raw.get("metric_fields") or []
    if not isinstance(excluded, list) or not all(
        isinstance(item, str) and item for item in excluded
    ):
        raise ValueError("suite_aggregation.exclude_capabilities must be a string list")
    if not isinstance(metric_fields, list) or not all(
        isinstance(item, str) and item for item in metric_fields
    ):
        raise ValueError("suite_aggregation.metric_fields must be a string list")
    output = str(raw.get("output") or "suite-overall.json")
    if not os.path.isabs(output):
        output = os.path.join(output_root, output)
    return {
        "output": os.path.abspath(output),
        "report_format": report_format,
        "min_slice_size": min_slice_size,
        "exclude_capabilities": excluded,
        "metric_fields": metric_fields,
    }


def run_suite(config: Dict[str, Any]) -> Dict[str, Any]:
    for path in config.get("registry_paths") or []:
        registry.add_registry_paths([path])
    model_name = str(config.get("model") or "")
    if not model_name:
        raise ValueError("suite config requires model")
    run_id = str(
        config.get("run_id")
        or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    )
    output_root = os.path.abspath(
        str(config.get("output_root") or os.path.join("outputs", "suite", run_id))
    )
    os.makedirs(output_root, exist_ok=True)

    aggregation_options = _suite_aggregation_options(config, output_root)
    benchmarks = prepare_benchmarks(config, model_name, output_root)
    # This call happens exactly once after all datasets are ready.
    predictor, inference_workers, using_pool = create_predictor(
        model_name, _pool_options(config), logger
    )
    session = EvaluationSession(predictor, inference_workers)
    started_at = datetime.now(timezone.utc).isoformat()
    results = session.run(benchmarks)
    suite_aggregation = None
    if aggregation_options is not None:
        from mesh_eval.scripts.reaggregate_events import aggregate_event_files

        event_files = [
            item["save"]
            for item in results.values()
            if item.get("status") == "ok" and item.get("save")
        ]
        aggregate = aggregate_event_files(
            event_files,
            excluded_capabilities=aggregation_options["exclude_capabilities"],
            report_format=aggregation_options["report_format"],
            min_slice_size=aggregation_options["min_slice_size"],
            metric_fields=aggregation_options["metric_fields"],
        )
        _atomic_json(aggregation_options["output"], aggregate)
        suite_aggregation = {
            "status": "ok",
            "output": aggregation_options["output"],
            "report_format": aggregation_options["report_format"],
            "sample_count": int(aggregate.get("sample_count") or 0),
            "input_file_count": len(event_files),
            "excluded_capabilities": aggregation_options["exclude_capabilities"],
        }
    manifest = {
        "schema_version": "evaluation-session/1.0",
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "model_pool": using_pool,
        "inference_workers": inference_workers,
        "deferred_openai_judging": list(
            config.get("deferred_openai_judging") or []
        ),
        "benchmarks": results,
    }
    if suite_aggregation is not None:
        manifest["suite_aggregation"] = suite_aggregation
    _atomic_json(os.path.join(output_root, "session.json"), manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compatible benchmarks while loading the predictor only once"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    result = run_suite(_load_config(args.config))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
