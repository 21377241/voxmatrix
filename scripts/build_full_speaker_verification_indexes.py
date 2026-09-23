#!/usr/bin/env python3
"""Build full speaker_verification indexes (VoxCeleb1 veri_test).

Protocol: dual-audio same/different speaker trials.
Native Catalog: speaker_verification__voxceleb1 → EER / AUC (+ acc@0.5 diagnostic).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/speaker_verification"

VERI = Path(
    "/mnt/afs/users/wangyl/benchmark_annotation_audit/native_v2_pipeline/references/voxceleb1/veri_test.txt"
)
WAV_ROOTS = [
    Path(
        "/mnt/afs/oss_data/datasets/03_multi_speaker/"
        "VoxCeleb_1_2_multi_speaker_multi_speaker/juliuscn/voxceleb/vox1/wav"
    ),
    Path("/mnt/afs/oss_data/datasets/03_multi_speaker/VoxCeleb1/wav"),
    Path("/mnt/afs/eval_data/03_multi_speaker/VoxCeleb1/wav"),
]


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    return len(rows), path


def _find_root() -> Optional[Path]:
    return next((r for r in WAV_ROOTS if r.is_dir()), None)


def build_voxceleb1() -> tuple[int, Path, dict]:
    root = _find_root()
    if root is None or not VERI.is_file():
        raise SystemExit(f"missing veri_test or wav root: veri={VERI} roots={WAV_ROOTS}")

    rows: list[dict] = []
    missing = 0
    n_same = 0
    n_diff = 0
    with VERI.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            label_s, p1, p2 = parts
            try:
                label = int(label_s)
            except ValueError:
                continue
            w1, w2 = root / p1, root / p2
            if not w1.is_file() or not w2.is_file():
                missing += 1
                continue
            if label == 1:
                n_same += 1
            else:
                n_diff += 1
            rows.append(
                {
                    "schema_version": "full-index/1.0",
                    "sample_id": f"voxceleb1_veri_{idx:05d}_{'same' if label == 1 else 'diff'}",
                    "dataset": "voxceleb1",
                    "benchmark": "voxceleb1",
                    "capability": "speaker_verification",
                    "protocol": "speaker_verification",
                    "subset": "veri_test",
                    "split": "test",
                    "WavPath": str(w1),
                    "WavPath2": str(w2),
                    "label": label,
                    "text": label,
                    "same_speaker": bool(label == 1),
                    "enroll_relpath": p1,
                    "test_relpath": p2,
                    "source_index": str(VERI),
                    "source_line": idx,
                }
            )
    n, path = _write(rows, "speaker_verification_voxceleb1.jsonl")
    return n, path, {
        "missing_audio": missing,
        "n_same": n_same,
        "n_diff": n_diff,
        "wav_root": str(root),
    }


def main() -> None:
    n, path, meta = build_voxceleb1()
    print(
        json.dumps(
            {"voxceleb1": {"n": n, "path": str(path), **meta}},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
