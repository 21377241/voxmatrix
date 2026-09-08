#!/usr/bin/env python3
"""Offline-rescore smoke_diar_all results with current DiarizationEvaluator + normalize."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from mesh_eval.agg.mesh import MeshAgg
from mesh_eval.evaluator.speaker import DiarizationEvaluator, normalize_diarization_prediction
from mesh_eval.process.diarization import DiarizationNormalize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BENCHMARKS = ("aishell_4", "alimeeting", "ami", "chime_6", "misp")


def rescore_benchmark(name: str) -> dict:
    path = RESULTS / f"{name}.jsonl"
    recs: dict[int, dict] = defaultdict(dict)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        recs[int(event["id"])][event["type"]] = event

    evaluator = DiarizationEvaluator()
    normalizer = DiarizationNormalize()
    scores = []
    for rid in sorted(recs):
        types = recs[rid]
        if "eval" not in types or "inference" not in types:
            continue
        raw = types["inference"]["data"]["content"]
        dur = float(
            types["inference"]["data"].get("audio_duration_seconds")
            or types.get("eval", {}).get("data", {}).get("audio_duration_seconds")
            or 30.0
        )
        normalized = normalizer(raw)
        # Prefer clip with known duration inside evaluator via kwargs.
        ref = types["eval"]["data"]["ref"]
        score = evaluator(
            normalized,
            ref,
            audio_duration_seconds=dur,
            runtime={"audio_duration_seconds": dur},
        )
        meta = normalize_diarization_prediction(raw, max_time=dur)
        score["diarization_salvaged"] = meta["diarization_salvaged"]
        score["diarization_truncated"] = meta["diarization_truncated"]
        # Keep sample dimensions for agg if present.
        for key in ("sample_id", "dataset", "latency_ms", "rtf", "audio_duration_seconds"):
            if key in types["eval"]["data"]:
                score.setdefault(key, types["eval"]["data"][key])
        score.setdefault("audio_duration_seconds", dur)
        scores.append(score)

    overall = MeshAgg()(scores)
    out = RESULTS / f"{name}-overall.rescored.json"
    out.write_text(json.dumps(overall, ensure_ascii=False, indent=2) + "\n")
    return {
        "benchmark": name,
        "n": len(scores),
        "der": overall.get("overall/der"),
        "der_valid": overall.get("overall/der_valid"),
        "failure_rate": overall.get("overall/failure_rate"),
        "truncated": overall.get("overall/diarization_truncated/count", 0),
        "salvaged": overall.get("overall/diarization_salvaged/count", 0),
        "old_der": json.loads((RESULTS / f"{name}-overall.json").read_text()).get(
            "overall/der"
        ),
    }


def main() -> None:
    rows = [rescore_benchmark(name) for name in BENCHMARKS]
    print(f"{'benchmark':12} {'old_der':>8} {'new_der':>8} {'der_valid':>9} {'fail':>6} {'trunc':>5} {'salv':>5}")
    for row in rows:
        print(
            f"{row['benchmark']:12} {row['old_der']:8.4f} {row['der']:8.4f} "
            f"{(row['der_valid'] if row['der_valid'] is not None else float('nan')):9.4f} "
            f"{(row['failure_rate'] or 0):6.2f} {row['truncated']:5d} {row['salvaged']:5d}"
        )


if __name__ == "__main__":
    main()
