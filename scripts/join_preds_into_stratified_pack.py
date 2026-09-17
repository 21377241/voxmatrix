#!/usr/bin/env python3
"""Join existing smoke/results (+ undertest) preds into a stratified semantic-judge pack.

Does not invent preds from references. Unfilled rows keep pred=\"\".
Also writes a residual pack for undertest inference.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path("/mnt/afs/users/wangyl")
DEFAULT_UNDERTEST = (
    ROOT / "VoxMatrix/scripts/data/semantic_judge_undertest_preds.jsonl"
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _ingest_row(
    by_id: Dict[str, str],
    by_wav: Dict[str, str],
    sid_to_wav: Dict[str, str],
    data: Dict[str, Any],
    *,
    prefer: bool = False,
) -> None:
    sid = str(data.get("sample_id") or "")
    pred = str(data.get("pred") or "").strip()
    if not pred:
        return
    if sid:
        if prefer or sid not in by_id:
            by_id[sid] = pred
        if sid.startswith("ar-open-"):
            base = sid[len("ar-open-") :]
            if prefer or base not in by_id:
                by_id[base] = pred
        if sid.startswith("cs-semi-"):
            base = sid[len("cs-semi-") :]
            if prefer or base not in by_id:
                by_id[base] = pred
    wav = str(data.get("WavPath") or sid_to_wav.get(sid) or "")
    if wav:
        key = os.path.abspath(wav)
        if prefer or key not in by_wav:
            by_wav[key] = pred


def _extract_pred_maps(
    undertest_paths: Sequence[Path],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return (by_sample_id, by_wavpath) pred maps from known results + undertest."""
    sources = [
        ROOT / "8p31/runs/qa_uro_bench_smoke/cluster_run_20260911_062245/results.jsonl",
        ROOT / "8p31/runs/qa_vocalbench_smoke/cluster_run_20260831_114828/results.jsonl",
        ROOT / "8p31/runs/qa_vocalbench_zh_smoke/cluster_run_20260911_062247/results.jsonl",
        ROOT / "8p31/runs/qa_spoken_squad_smoke/cluster_run_20260911_062226/results.jsonl",
        ROOT / "8p31/runs/qa_ear_wdyl_smoke/cluster_run_20260831_055552/results.jsonl",
        ROOT
        / "8p31/runs/audio_reasoning_vocalbench_smoke/cluster_run_20260831_104516/results.jsonl",
        ROOT
        / "8p31/runs/audio_reasoning_vocalbench_zh_smoke/cluster_run_20260831_104455/results.jsonl",
        ROOT / "8p31/runs/audio_reasoning_mmsu_smoke/cluster_run_20260911_061531/results.jsonl",
        ROOT / "VoxMatrix/smoke_st_all/results/covost2_en_zh.jsonl",
        ROOT / "VoxMatrix/smoke_st_all/results/covost2_zh_en.jsonl",
        ROOT / "VoxMatrix/smoke_st_all/results_mmsu_open/mmsu_open.jsonl",
        ROOT / "VoxMatrix/smoke_cs_all/results_mmsu_open/mmsu_open.jsonl",
        ROOT / "VoxMatrix/smoke_cs_all/results/vocalbench_zh_knowledge.jsonl",
        ROOT / "VoxMatrix/smoke_cs_all/results/vocalbench_zh_open_ended.jsonl",
        ROOT / "VoxMatrix/scripts/data/semantic_judge_compare_pack.jsonl",
    ]
    manifests = [
        ROOT / "8p31/runs/qa_uro_bench_smoke/qa_uro_bench_10.jsonl",
        ROOT / "8p31/runs/qa_vocalbench_smoke/qa_vocalbench_10.jsonl",
        ROOT / "8p31/runs/qa_vocalbench_zh_smoke/qa_vocalbench_zh_10.jsonl",
        ROOT / "8p31/runs/qa_spoken_squad_smoke/qa_spoken_squad_10.jsonl",
        ROOT / "8p31/runs/qa_ear_wdyl_smoke/qa_ear_wdyl_10.jsonl",
        ROOT / "8p31/runs/audio_reasoning_vocalbench_smoke/audio_reasoning_vocalbench_10.jsonl",
        ROOT
        / "8p31/runs/audio_reasoning_vocalbench_zh_smoke/audio_reasoning_vocalbench_zh_10.jsonl",
        ROOT / "8p31/runs/audio_reasoning_mmsu_smoke/audio_reasoning_mmsu_open_10.jsonl",
        ROOT / "VoxMatrix/smoke_st_all/covost2_en_zh/manifest.jsonl",
        ROOT / "VoxMatrix/smoke_st_all/covost2_zh_en/manifest.jsonl",
        ROOT / "VoxMatrix/smoke_st_all/mmsu_open/manifest.jsonl",
        ROOT / "VoxMatrix/smoke_cs_all/mmsu_open/manifest.jsonl",
        ROOT / "VoxMatrix/smoke_cs_all/vocalbench_zh_knowledge/manifest.jsonl",
        ROOT / "VoxMatrix/smoke_cs_all/vocalbench_zh_open_ended/manifest.jsonl",
    ]
    by_id: Dict[str, str] = {}
    sid_to_wav: Dict[str, str] = {}
    for path in manifests:
        for obj in _load_jsonl(path):
            sid = str(obj.get("sample_id") or "")
            wav = str(obj.get("WavPath") or obj.get("audio_locator") or "")
            if sid and wav:
                sid_to_wav[sid] = os.path.abspath(wav) if os.path.isabs(wav) else wav

    by_wav: Dict[str, str] = {}
    for path in sources:
        for obj in _load_jsonl(path):
            data = obj.get("data") if obj.get("type") == "eval" else obj
            if not isinstance(data, dict):
                continue
            _ingest_row(by_id, by_wav, sid_to_wav, data, prefer=False)

    # Undertest preds override / fill gaps (fresh Instruct runs for residual cells).
    for path in undertest_paths:
        for obj in _load_jsonl(path):
            if obj.get("error") and not str(obj.get("pred") or "").strip():
                continue
            _ingest_row(by_id, by_wav, sid_to_wav, obj, prefer=True)

    return by_id, by_wav


def join_pack(
    pack: List[Dict[str, Any]],
    by_id: Dict[str, str],
    by_wav: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    filled = 0
    for row in pack:
        if str(row.get("pred") or "").strip():
            filled += 1
            continue
        sid = str(row.get("sample_id") or "")
        wav = str(row.get("WavPath") or "")
        pred = ""
        source = ""
        for key in (
            sid,
            sid[len("ar-open-") :] if sid.startswith("ar-open-") else "",
            sid[len("cs-semi-") :] if sid.startswith("cs-semi-") else "",
        ):
            if key and key in by_id:
                pred = by_id[key]
                source = "sample_id"
                break
        if not pred and wav:
            pred = by_wav.get(os.path.abspath(wav), "")
            if pred:
                source = "wavpath"
        if pred:
            row["pred"] = pred
            row["pred_source"] = source
            filled += 1
    cells: Dict[str, Dict[str, int]] = {}
    for row in pack:
        key = f"{row.get('capability')}|{row.get('rubric')}"
        bucket = cells.setdefault(key, {"n": 0, "with_pred": 0})
        bucket["n"] += 1
        if str(row.get("pred") or "").strip():
            bucket["with_pred"] += 1
    return pack, {"filled": filled, "n": len(pack), "cells": cells}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pack",
        type=Path,
        default=ROOT / "VoxMatrix/scripts/data/semantic_judge_stratified_pack.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT
        / "VoxMatrix/scripts/data/semantic_judge_stratified_pack.with_pred.jsonl",
    )
    parser.add_argument(
        "--residual-out",
        type=Path,
        default=ROOT
        / "VoxMatrix/scripts/data/semantic_judge_stratified_pack.need_pred.jsonl",
    )
    parser.add_argument(
        "--undertest",
        type=Path,
        action="append",
        default=None,
        help="Undertest pred JSONL (repeatable). Default: semantic_judge_undertest_preds.jsonl if present.",
    )
    args = parser.parse_args()
    undertest_paths: List[Path] = list(args.undertest or [])
    if not undertest_paths and DEFAULT_UNDERTEST.is_file():
        undertest_paths = [DEFAULT_UNDERTEST]
    elif not undertest_paths:
        undertest_paths = [DEFAULT_UNDERTEST]  # may be missing; ingest no-ops

    pack = _load_jsonl(args.pack)
    by_id, by_wav = _extract_pred_maps(undertest_paths)
    pack, stats = join_pack(pack, by_id, by_wav)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in pack:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    residual = [r for r in pack if not str(r.get("pred") or "").strip()]
    with args.residual_out.open("w", encoding="utf-8") as handle:
        for row in residual:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "undertest_paths": [str(p) for p in undertest_paths],
                "undertest_exists": [p.is_file() for p in undertest_paths],
                "pred_by_id": len(by_id),
                "pred_by_wav": len(by_wav),
                **stats,
                "residual": len(residual),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"wrote {args.out}")
    print(f"wrote residual n={len(residual)} -> {args.residual_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
