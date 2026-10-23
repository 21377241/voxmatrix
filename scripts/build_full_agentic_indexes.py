#!/usr/bin/env python3
"""Build Full (+ Native) indexes for IF / multi-turn / spoken_slu.

These slots historically loaded via external Dataset classes (no
``mesh_eval/data/full`` jsonl). This script materializes the same normalized
runtime rows the loaders already produce, relocates ephemeral ``/tmp`` audio
onto AFS, and writes:

  mesh_eval/data/full/{instruction_following,multi_turn_dialogue,spoken_slu}/
  mesh_eval/data/native/<same basename>.jsonl   (10-row smoke subsets)

Re-run is idempotent. VoiceBench / MTalk audio lands under
``/mnt/afs/eval_data/voxmatrix_full_audio/{voicebench,mtalk}/``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "mesh_eval" / "data" / "full"
NATIVE = ROOT / "mesh_eval" / "data" / "native"
AUDIO_ROOT = Path("/mnt/afs/eval_data/voxmatrix_full_audio")

# Drop ephemeral / rebuildable bookkeeping; keep everything prompts + judges need.
DROP_KEYS = {
    "canonical",
    "canonical_schema_version",
    "run_context",
    "provenance",
    "resolved_protocol",
    "measurement_protocol",
    "measurement_protocol_hash",
    "protocol_hash",
    "evaluators",
    "capability_protocol",
    "label_meta",
    "parser",
    "prompt_name",
    "metrics",
    "use_bucket",
    "offline_proxy",
    "answer_type",
    "source_manifest",
    "full_eval_view",
    "source_dataset_id",
}

TMP_PREFIXES = (
    "/tmp/voxmatrix_voicebench_audio",
    "/tmp/voxmatrix_mtalk_audio",
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _slot_id(capability: str, benchmark: str) -> str:
    return f"{capability}__{benchmark}"


def _rewrite_paths(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [_rewrite_paths(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, mapping) for key, item in value.items()}
    return value


def _collect_paths(value: Any, found: set[str]) -> None:
    if isinstance(value, str):
        if value.startswith(TMP_PREFIXES) or "/tmp/voxmatrix_" in value:
            found.add(value)
        return
    if isinstance(value, list):
        for item in value:
            _collect_paths(item, found)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_paths(item, found)


def _durable_target(src: str, kind: str) -> Path:
    name = Path(src).name
    return AUDIO_ROOT / kind / name


def _relocate_tmp_audio(rows: list[dict], kind: str) -> list[dict]:
    """Copy /tmp wavs onto AFS and rewrite every nested path reference."""
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIO_ROOT / kind).mkdir(parents=True, exist_ok=True)
    mapping: dict[str, str] = {}
    found: set[str] = set()
    for row in rows:
        _collect_paths(row, found)
    for src in sorted(found):
        src_path = Path(src)
        dst = _durable_target(src, kind)
        if src_path.is_file():
            if not dst.is_file() or dst.stat().st_size != src_path.stat().st_size:
                shutil.copy2(src_path, dst)
        elif not dst.is_file():
            raise FileNotFoundError(f"missing tmp audio and no durable copy: {src}")
        mapping[src] = str(dst)
    return [_rewrite_paths(row, mapping) for row in rows]


def _compact(row: dict, *, capability: str, benchmark: str) -> dict:
    out = {key: value for key, value in row.items() if key not in DROP_KEYS}
    out["schema_version"] = "full-index/agentic/1.0"
    out["capability"] = capability
    out["protocol"] = capability
    out["benchmark"] = benchmark
    out.setdefault("dataset", row.get("dataset_id") or row.get("dataset") or benchmark)
    # Prefer structured gold for judges; keep text for ref_col compatibility.
    if isinstance(out.get("reference"), dict) and "reference_obj" not in out:
        out["reference_obj"] = out.pop("reference")
    else:
        out.pop("reference", None)
    return out


def _spread(rows: list[dict], count: int = 10) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    # Prefer category round-robin when present.
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        key = str(
            row.get("dialogue_category")
            or ((row.get("reference_obj") or {}).get("rubric") or {}).get("category")
            or ((row.get("reference_obj") or {}).get("category"))
            or "all"
        )
        buckets.setdefault(key, []).append(row)
    keys = sorted(buckets)
    picked: list[dict] = []
    idx = {key: 0 for key in keys}
    while len(picked) < count:
        progressed = False
        for key in keys:
            i = idx[key]
            if i < len(buckets[key]):
                picked.append(buckets[key][i])
                idx[key] = i + 1
                progressed = True
                if len(picked) >= count:
                    break
        if not progressed:
            break
    if len(picked) < count:
        # fill evenly from full list
        step = max(1, len(rows) // count)
        for i in range(0, len(rows), step):
            if rows[i] not in picked:
                picked.append(rows[i])
            if len(picked) >= count:
                break
    return picked[:count]


def _export(
    *,
    name: str,
    capability: str,
    benchmark: str,
    loader: Callable[[], list[dict]],
    relocate_kind: str | None = None,
) -> tuple[Path, Path, int]:
    print(f"[build] {name}: loading…", flush=True)
    rows = loader()
    if relocate_kind:
        print(f"[build] {name}: relocating {relocate_kind} audio…", flush=True)
        rows = _relocate_tmp_audio(rows, relocate_kind)
    compact = [_compact(row, capability=capability, benchmark=benchmark) for row in rows]
    # Attach subset-manifest language/scenario/metadata before writing.
    from mesh_eval.data.full_index_meta import finalize_rows

    slot_id = _slot_id(capability, benchmark)
    try:
        compact = finalize_rows(compact, slot_id)
    except KeyError as exc:
        print(f"[build] {name}: warn subset meta skip ({exc})", flush=True)
    full_path = FULL / capability / f"{name}.jsonl"
    native_path = NATIVE / f"{name}.jsonl"
    _write_jsonl(full_path, compact)
    _write_jsonl(native_path, _spread(compact, 10))
    print(f"[build] {name}: full={len(compact)} -> {full_path}", flush=True)
    print(f"[build] {name}: native={sum(1 for _ in native_path.open())} -> {native_path}", flush=True)
    return full_path, native_path, len(compact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated slot basenames, e.g. instruction_following_voicebench",
    )
    args = parser.parse_args()
    only = {item.strip() for item in args.only.split(",") if item.strip()}

    # Import after path setup so audio_evals resolves from this repo.
    import sys

    sys.path.insert(0, str(ROOT))
    from mesh_eval.dataset.external_benchmarks import (
        AudioAgentBenchDataset,
        FluentSpeechCommandsT4Dataset,
        MTalkBenchDataset,
        SLURPT4Dataset,
        VocalBenchDataset,
        VoiceBenchT4Dataset,
    )

    voicebench_cache = str(AUDIO_ROOT / "voicebench")
    mtalk_cache = str(AUDIO_ROOT / "mtalk")
    Path(voicebench_cache).mkdir(parents=True, exist_ok=True)
    Path(mtalk_cache).mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str, str, Callable[[], list[dict]], str | None]] = [
        (
            "instruction_following_voicebench",
            "instruction_following",
            "voicebench",
            lambda: VoiceBenchT4Dataset(
                benchmark_id="t4-voicebench-ifeval",
                view="ifeval",
                cache_dir=voicebench_cache,
            ).load(),
            "voicebench",
        ),
        (
            "instruction_following_vocalbench",
            "instruction_following",
            "vocalbench",
            lambda: VocalBenchDataset(
                benchmark_id="t4-vocalbench-instruction-following",
                view="instruction_following",
                language="en",
            ).load(),
            None,
        ),
        (
            "instruction_following_vocalbench_zh",
            "instruction_following",
            "vocalbench_zh",
            lambda: VocalBenchDataset(
                benchmark_id="t4-vocalbench-zh-instruction-following",
                view="instruction_following",
                language="zh",
            ).load(),
            None,
        ),
        (
            "multi_turn_dialogue_audioagentbench_suite",
            "multi_turn_dialogue",
            "audioagentbench_suite",
            lambda: AudioAgentBenchDataset(
                benchmark_id="audio-agent-bench-dialogue",
                view="dialogue",
            ).load(),
            None,
        ),
        (
            "multi_turn_dialogue_vocalbench",
            "multi_turn_dialogue",
            "vocalbench",
            lambda: VocalBenchDataset(
                benchmark_id="t4-vocalbench-multi-turn",
                view="multi_turn",
                language="en",
            ).load(),
            None,
        ),
        (
            "multi_turn_dialogue_vocalbench_zh",
            "multi_turn_dialogue",
            "vocalbench_zh",
            lambda: VocalBenchDataset(
                benchmark_id="t4-vocalbench-zh-multi-turn",
                view="multi_turn",
                language="zh",
            ).load(),
            None,
        ),
        (
            "multi_turn_dialogue_voicebench",
            "multi_turn_dialogue",
            "voicebench",
            lambda: VoiceBenchT4Dataset(
                benchmark_id="t4-voicebench-mtbench",
                view="mtbench",
                cache_dir=voicebench_cache,
            ).load(),
            "voicebench",
        ),
        (
            "multi_turn_dialogue_mtalk_bench",
            "multi_turn_dialogue",
            "mtalk_bench",
            lambda: MTalkBenchDataset(
                benchmark_id="t4-mtalk-bench",
                cache_dir=mtalk_cache,
            ).load(),
            "mtalk",
        ),
        (
            "spoken_slu_fluent_speech_commands",
            "spoken_slu",
            "fluent_speech_commands",
            lambda: FluentSpeechCommandsT4Dataset(
                benchmark_id="t4-fsc-instruction-following",
                view="instruction_following",
            ).load(),
            None,
        ),
        (
            "spoken_slu_slurp",
            "spoken_slu",
            "slurp",
            lambda: SLURPT4Dataset(
                benchmark_id="t4-slurp-instruction-following",
                view="instruction_following",
            ).load(),
            None,
        ),
    ]

    summary = []
    for name, capability, benchmark, loader, relocate in jobs:
        if only and name not in only:
            continue
        full_path, native_path, n = _export(
            name=name,
            capability=capability,
            benchmark=benchmark,
            loader=loader,
            relocate_kind=relocate,
        )
        summary.append((name, n, str(full_path.relative_to(ROOT)), str(native_path.relative_to(ROOT))))

    print("\n=== summary ===")
    for name, n, full_rel, native_rel in summary:
        print(f"{name:48s} n={n:5d}  full={full_rel}  native={native_rel}")


if __name__ == "__main__":
    main()
