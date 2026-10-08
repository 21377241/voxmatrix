#!/usr/bin/env python3
"""CPU-only native-suite preflight.

Loads the catalog, checks each capability, writes a runtime suite that only
includes datasets that actually loaded, and records blocked capabilities.
Does not load Qwen3-Omni.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault(
    "VOXMATRIX_FULL_EVAL_ROOT",
    "/mnt/afs/users/wangyl/vocal_bench/work/full_eval",
)
os.environ.setdefault("MESH_FULL_EVAL_ROOT", os.environ["VOXMATRIX_FULL_EVAL_ROOT"])
os.environ.setdefault(
    "VOXMATRIX_SUBSET_MANIFEST",
    str(ROOT / "annotation" / "benchmark_subset_manifest.jsonl"),
)

from audio_evals.main import preload_dataset  # noqa: E402
from audio_evals.registry import Registry  # noqa: E402
from mesh_eval.core.subset_manifest import (  # noqa: E402
    default_subset_manifest_path,
    load_annotation_baseline,
)

MODEL_PATH = Path("/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct")
MODEL_ENV = Path("/mnt/afs/users/wangyl/VoxMatrix/envs/qwen3-omni")
if not os.environ.get("NATIVE_OUTPUT_DIR"):
    from datetime import datetime

    os.environ["NATIVE_OUTPUT_DIR"] = str(
        Path(os.environ.get("NATIVE_RUNS_ROOT") or ROOT / "outputs" / "native_runs")
        / f"{datetime.now():%Y%m%d_%H%M%S}_{os.uname().nodename}_{os.getpid()}"
    )
OUTPUT = Path(os.environ["NATIVE_OUTPUT_DIR"]).resolve()
RUNTIME_REGISTRY = OUTPUT / "registry" / "eval_task"
# NATIVE_DATA=full: clip slots read their Full index (slot_map.yaml), all slots run unlimited.
FULL_MODE = str(os.environ.get("NATIVE_DATA") or "native").strip().lower() == "full"
SLOT_MAP = ROOT / "mesh_eval" / "data" / "full" / "slot_map.yaml"
FULL_DATASET_SUFFIX = "--full"

# Catalog capability may be renamed while annotation still uses the historical
# capability id (spoken_slu ← instruction_following for FSC/SLURP).
ANNOTATION_CAPABILITY_ALIASES: dict[str, tuple[str, ...]] = {
    "spoken_slu": ("spoken_slu", "instruction_following"),
}


def _audio_path(row: dict) -> str:
    input_obj = row.get("input") or {}
    return str(
        row.get("WavPath")
        or input_obj.get("audio_path")
        or input_obj.get("audio")
        or ""
    )


def _load_catalog() -> dict:
    return yaml.safe_load((HERE / "catalog.yaml").read_text())


def _slot_id(item: dict) -> str:
    return str(item.get("id") or item.get("capability") or "").strip()


def _capability_id(item: dict) -> str:
    return str(item.get("capability") or item.get("id") or "").strip()


def _slot_limit(item: dict, limit: int) -> int:
    """Round down to whole units (e.g. 4 rows = one S/M/L/XL family)."""
    step = int(item.get("limit_step") or 1)
    if step <= 1 or limit <= 0:
        return limit
    return max(step, limit - limit % step)


def _task_name(item: dict) -> str:
    return f"native-task-{_slot_id(item).replace('_', '-')}"


def _benchmark_id(item: dict) -> str:
    return f"native-{_slot_id(item).replace('_', '-')}"


def _write_full_datasets(catalog: dict) -> dict[str, str]:
    """Register ``<dataset>--full`` copies whose f_name is the slot's Full index."""
    slot_map = yaml.safe_load(SLOT_MAP.read_text(encoding="utf-8")) or {}
    specs: dict = {}
    for path in sorted((ROOT / "mesh_eval" / "registry" / "dataset").glob("*.yaml")):
        specs.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    renamed: dict[str, str] = {}
    full_registry: dict = {}
    for item in catalog["capabilities"]:
        entry = slot_map.get(_slot_id(item))
        spec = specs.get(item.get("dataset"))
        if not entry or not spec:
            continue
        args = dict(spec.get("args") or {})
        if args.get("f_name") != entry["native"]:
            raise ValueError(
                f"{_slot_id(item)}: registry f_name {args.get('f_name')} != slot_map native {entry['native']}"
            )
        args["f_name"] = entry["full"]
        name = f"{item['dataset']}{FULL_DATASET_SUFFIX}"
        full_registry[name] = {"class": spec["class"], "args": args}
        renamed[item["dataset"]] = name
    _dump_yaml(OUTPUT / "registry" / "dataset" / "native_full.yaml", full_registry)
    return renamed


def _write_concurrent_model(model_name: str, concurrency: int) -> str:
    """Register ``<model>--c<N>``: same args, engine and wrapper serve N requests at once."""
    specs: dict = {}
    for root in (ROOT / "mesh_eval" / "registry" / "model", HERE / "registry" / "model"):
        for path in sorted(root.rglob("*.yaml")):
            specs.update(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    spec = specs.get(model_name)
    if not spec or not str(spec.get("class", "")).endswith("Qwen3OmniVLLMOmni"):
        raise ValueError(f"NATIVE_CONCURRENCY needs a Qwen3OmniVLLMOmni model, got {model_name}")
    if (spec.get("args") or {}).get("speech"):
        raise ValueError(f"NATIVE_CONCURRENCY only applies to text models, got {model_name}")
    name = f"{model_name}--c{concurrency}"
    args = dict(spec.get("args") or {}, max_num_seqs=concurrency)
    _dump_yaml(
        OUTPUT / "registry" / "model" / "native_concurrency.yaml",
        {name: {"class": spec["class"], "args": args}},
    )
    return name


def _dump_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _make_task_yaml(item: dict, model_name: str) -> dict:
    args = {
        "dataset": item["dataset"],
        "prompt": item["prompt"],
        "model": item.get("model") or model_name,
        "post_process": item.get("post_process") or [],
        "evaluator": item["evaluator"],
        "agg": item["agg"],
    }
    return {"class": "audio_evals.base.EvalTaskCfg", "args": args}


def _make_suite(
    *,
    run_id: str,
    model_name: str,
    output_root: Path,
    registry_paths: list[str],
    rows: list[dict],
    limit: int,
    replicas: int = 1,
    concurrency: int = 1,
) -> dict:
    slots = replicas * concurrency
    benchmarks = []
    for item in rows:
        spec = {
            "benchmark_id": _benchmark_id(item),
            "dataset": item["dataset"],
            "task": _task_name(item),
            "prompt": item["prompt"],
            "evaluator": item["evaluator"],
            "agg": item["agg"],
            "limit": _slot_limit(item, limit),
            "evaluation_workers": 1,
        }
        if item.get("two_phase"):
            spec["two_phase"] = True
        if item.get("subset_record_id"):
            spec["subset_record_id"] = item["subset_record_id"]
        benchmarks.append(spec)
    return {
        "schema_version": "evaluation-suite-config/1.0",
        "run_id": run_id,
        "model": model_name,
        "output_root": str(output_root),
        "registry_paths": registry_paths,
        "subset_manifest": {
            "path": str(default_subset_manifest_path()),
            "exclude_capabilities": ["meeting_summary", "todo_extraction"],
            "allow_unmatched": True,
        },
        "model_pool": {
            "use_model_pool": "on" if slots > 1 else "off",
            "replicas": replicas,
            "gpus_per_replica": 1,
            "inference_workers": slots,
            "allow_gpu_sharing": False,
            "model_ready_policy": "required" if slots > 1 else "auto",
        },
        "suite_aggregation": {
            "enabled": True,
            "report_format": "both",
            "min_slice_size": 1,
            "exclude_capabilities": ["meeting_summary", "todo_extraction"],
        },
        "benchmarks": benchmarks,
    }


def main() -> int:
    catalog = _load_catalog()
    limit = int(os.environ.get("NATIVE_LIMIT") or os.environ.get("W1_LIMIT") or catalog.get("limit") or 10)
    full_datasets: dict[str, str] = {}
    if FULL_MODE:
        # NATIVE_FULL_LIMIT is only for timing probes; 0 = every row.
        limit = int(os.environ.get("NATIVE_FULL_LIMIT") or 0)
        full_datasets = _write_full_datasets(catalog)
    # Comma-separated capability ids, e.g. long_form_asr. Empty = all catalog slots.
    capability_filter = {
        part.strip()
        for part in str(os.environ.get("NATIVE_CAPABILITY_FILTER") or "").split(",")
        if part.strip()
    }
    # Comma-separated catalog slot ids, e.g. acoustic_scene__mmsu. Empty = all slots.
    slot_filter = {
        part.strip()
        for part in str(os.environ.get("NATIVE_SLOT_FILTER") or "").split(",")
        if part.strip()
    }
    default_model = str(
        os.environ.get("NATIVE_MODEL") or catalog["model"]
    )
    speech_model = str(
        os.environ.get("NATIVE_SPEECH_MODEL") or catalog["speech_model"]
    )
    concurrency = max(1, int(os.environ.get("NATIVE_CONCURRENCY") or 1))
    if concurrency > 1:
        default_model = _write_concurrent_model(default_model, concurrency)
    baseline = load_annotation_baseline()
    errors: list[str] = []
    report = {
        "catalog": str(HERE / "catalog.yaml"),
        "data_mode": "full" if FULL_MODE else "native",
        "limit": limit,
        "model": default_model,
        "concurrency": concurrency,
        "capability_filter": sorted(capability_filter),
        "slot_filter": sorted(slot_filter),
        "annotation_baseline": baseline.path,
        "annotation_baseline_sha256": baseline.sha256,
        "annotation_baseline_records": len(baseline),
        "full_eval_root": os.environ.get("VOXMATRIX_FULL_EVAL_ROOT"),
        "model_path": str(MODEL_PATH),
        "model_env": str(MODEL_ENV),
        "capabilities": {},
        "runnable_main": [],
        "runnable_tts": [],
        "blocked": [],
    }

    if not MODEL_PATH.is_dir():
        errors.append(f"missing checkpoint: {MODEL_PATH}")
    if not MODEL_ENV.is_dir():
        errors.append(f"missing isolated env: {MODEL_ENV}")

    registry = Registry()
    registry.add_registry_paths(
        [
            ROOT / "mesh_eval" / "registry",
            HERE / "registry",
        ]
        + ([OUTPUT / "registry"] if FULL_MODE or concurrency > 1 else [])
    )
    for name in (default_model, speech_model):
        if name not in registry._model:
            errors.append(f"model not registered: {name}")

    runnable = {"main": [], "tts": []}
    task_registry: dict = {}

    for item in catalog["capabilities"]:
        if item.get("dataset") in full_datasets:
            item = dict(item, dataset=full_datasets[item["dataset"]])
        slot_id = _slot_id(item)
        cap_id = _capability_id(item)
        bench_id = str(item.get("benchmark") or "")
        entry = {
            "id": slot_id,
            "capability": cap_id,
            "benchmark": bench_id,
            "catalog_status": item.get("status"),
            "suite": item.get("suite"),
            "dataset": item.get("dataset"),
            "note": item.get("note") or item.get("reason") or "",
        }
        if capability_filter and cap_id not in capability_filter:
            continue
        if slot_filter and slot_id not in slot_filter:
            continue
        coverage = []
        for ann_cap in ANNOTATION_CAPABILITY_ALIASES.get(cap_id, (cap_id,)):
            coverage.extend(baseline.executable_records(ann_cap) if ann_cap else [])
        if bench_id:
            coverage = [
                row
                for row in coverage
                if str((row.get("benchmark") or {}).get("id") or "") == bench_id
            ]
        entry["annotation_records"] = len(coverage)
        entry["annotation_benchmarks"] = sorted(
            {row["benchmark"]["id"] for row in coverage}
        )
        if item.get("subset_record_id"):
            entry["subset_record_id"] = item["subset_record_id"]
        if item.get("suite") == "none" or item.get("status") == "blocked":
            entry["runtime_status"] = "blocked"
            entry["reason"] = item.get("reason") or "catalog blocked"
            report["blocked"].append(slot_id)
            report["capabilities"][slot_id] = entry
            continue
        if not coverage:
            entry.update(
                {
                    "runtime_status": "blocked",
                    "reason": (
                        f"annotation baseline has no executable {cap_id} records"
                    ),
                }
            )
            report["blocked"].append(slot_id)
            report["capabilities"][slot_id] = entry
            continue
        if bench_id and bench_id not in entry["annotation_benchmarks"]:
            entry.update(
                {
                    "runtime_status": "blocked",
                    "reason": (
                        f"annotation baseline has no executable {cap_id}/{bench_id}"
                    ),
                }
            )
            report["blocked"].append(slot_id)
            report["capabilities"][slot_id] = entry
            continue

        dataset_name = item["dataset"]
        dataset = registry.get_dataset(dataset_name)
        if dataset is None:
            entry.update(
                {
                    "runtime_status": "blocked",
                    "reason": f"dataset not registered: {dataset_name}",
                }
            )
            report["blocked"].append(slot_id)
            report["capabilities"][slot_id] = entry
            continue

        for kind, key in (
            ("prompt", item["prompt"]),
            ("evaluator", item["evaluator"]),
            ("agg", item["agg"]),
        ):
            registered = getattr(registry, f"_{kind}")
            if key not in registered:
                errors.append(f"{slot_id}: {kind} not registered: {key}")

        try:
            loaded = preload_dataset(dataset, limit=_slot_limit(item, limit))
            rows = loaded.rows
        except Exception as exc:
            entry.update(
                {
                    "runtime_status": "blocked",
                    "reason": f"load failed: {type(exc).__name__}: {exc}",
                    "loaded": 0,
                }
            )
            report["blocked"].append(slot_id)
            report["capabilities"][slot_id] = entry
            continue

        missing_audio = []
        for row in rows:
            audio = _audio_path(row)
            if audio and not Path(audio).is_file():
                missing_audio.append(audio)
        entry.update(
            {
                "runtime_status": "runnable",
                "loaded": len(rows),
                "sample0_id": rows[0].get("sample_id") if rows else None,
                "sample0_audio": _audio_path(rows[0]) if rows else None,
                "missing_audio": len(missing_audio),
            }
        )
        if not rows:
            entry["runtime_status"] = "blocked"
            entry["reason"] = "loaded 0 rows"
            report["blocked"].append(slot_id)
            report["capabilities"][slot_id] = entry
            continue
        if missing_audio:
            errors.append(
                f"{slot_id}: {len(missing_audio)}/{len(rows)} audio files missing"
            )
        suite_name = item.get("suite") or "main"
        model_name = item.get("model") or (
            speech_model if suite_name == "tts" else default_model
        )
        task_registry[_task_name(item)] = _make_task_yaml(item, model_name)
        runnable[suite_name].append(item)
        report["runnable_" + suite_name].append(slot_id)
        report["capabilities"][slot_id] = entry

    OUTPUT.mkdir(parents=True, exist_ok=True)
    RUNTIME_REGISTRY.mkdir(parents=True, exist_ok=True)
    _dump_yaml(RUNTIME_REGISTRY / "native_runtime.yaml", task_registry)

    registry_paths = [
        str(ROOT / "mesh_eval" / "registry"),
        str(HERE / "registry"),
        str(OUTPUT / "registry"),
    ]
    suite_main = OUTPUT / "suite.runtime.yaml"
    suite_tts = OUTPUT / "suite_tts.runtime.yaml"
    # Drop stale suites so a filtered preflight cannot accidentally re-run old slots.
    for stale in (suite_main, suite_tts):
        if stale.is_file():
            stale.unlink()
    suite_dir = str(os.environ.get("NATIVE_SUITE_DIR") or "main").strip() or "main"
    if runnable["main"]:
        _dump_yaml(
            suite_main,
            _make_suite(
                run_id=f"native-{suite_dir}",
                model_name=default_model,
                output_root=OUTPUT / suite_dir,
                registry_paths=registry_paths,
                rows=runnable["main"],
                limit=limit,
                replicas=int(os.environ.get("NATIVE_REPLICAS") or 1),
                concurrency=concurrency,
            ),
        )
    if runnable["tts"]:
        _dump_yaml(
            suite_tts,
            _make_suite(
                run_id="native-tts",
                model_name=speech_model,
                output_root=OUTPUT / "tts",
                registry_paths=registry_paths,
                rows=runnable["tts"],
                limit=limit,
                replicas=int(os.environ.get("NATIVE_TTS_REPLICAS") or 1),
            ),
        )

    fatal = [err for err in errors if not err.endswith("audio files missing")]
    audio_errors = [err for err in errors if err.endswith("audio files missing")]
    report["ok"] = not fatal and not audio_errors
    report["errors"] = errors
    report["runnable_count"] = len(report["runnable_main"]) + len(report["runnable_tts"])
    report["blocked_count"] = len(report["blocked"])
    out_path = OUTPUT / "preflight.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    summary = {
        "ok": report["ok"],
        "runnable_count": report["runnable_count"],
        "blocked_count": report["blocked_count"],
        "runnable_main": report["runnable_main"],
        "runnable_tts": report["runnable_tts"],
        "blocked": report["blocked"],
        "errors": errors,
        "output_dir": str(OUTPUT),
        "preflight": str(out_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if fatal or audio_errors:
        print(
            f"\npreflight FAILED ({len(errors)} issues) -> {out_path}",
            file=sys.stderr,
        )
        return 1
    print(
        f"\npreflight OK runnable={report['runnable_count']} "
        f"blocked={report['blocked_count']} -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
