#!/usr/bin/env python3
"""Build full acoustic-scene indexes (closed-set classification).

Tracks align with Native catalog:
  - DCASE2013 scenes (10 official labels)
  - MMSU background_scene_recognition (per-item 4 scene options; return label)
  - TUT Sound Events 2017 acoustic_scene view (officially street-only)
  - TUT Urban Acoustic Scenes 2018 Mobile (10 official labels; audio may be partial)
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/acoustic_scene"

DCASE_MANIFEST = Path(
    "/mnt/afs/eval_data/benchmarks/DCASE2013_Scenes_Public/manifests/acoustic_scene.jsonl"
)
TUT2017_MANIFEST = Path(
    "/mnt/afs/eval_data/benchmarks/TUT-Sound-Events-2017/manifests/acoustic_scene.jsonl"
)
TUT2018_ROOT = Path(
    "/mnt/afs/eval_data/benchmarks/TUT_Urban_Acoustic_Scenes_2018_Mobile/raw/"
    "TUT-urban-acoustic-scenes-2018-mobile-development"
)
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
MMSU_JSONL = MMSU_ROOT / "mmsu_audioqa.jsonl"

TUT2018_OFFICIAL = [
    "airport",
    "bus",
    "metro",
    "metro_station",
    "park",
    "public_square",
    "shopping_mall",
    "street_pedestrian",
    "street_traffic",
    "tram",
]


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
        encoding="utf-8",
    )
    return len(rows), path


def build_dcase() -> tuple[int, Path, dict]:
    rows = []
    labels: set[str] = set()
    for line in DCASE_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        wav = Path(str(doc.get("audio") or doc.get("WavPath") or ""))
        label = str(
            doc.get("scene_label")
            or doc.get("scene_class")
            or doc.get("label")
            or ""
        ).strip()
        sid = str(doc.get("id") or "").strip()
        if not sid or not label or not wav.is_file():
            continue
        labels.add(label)
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "dcase2013_scenes",
                "benchmark": "dcase2013_scenes",
                "capability": "acoustic_scene",
                "protocol": "acoustic_scene",
                "subset": str(doc.get("split") or "development"),
                "split": str(doc.get("split") or "development"),
                "WavPath": str(wav),
                "text": label,
                "answer": label,
                "closed_labels": None,  # filled after pass
                "source_index": str(DCASE_MANIFEST),
            }
        )
    closed = sorted(labels)
    for row in rows:
        row["closed_labels"] = closed
        row["question"] = (
            "Identify the acoustic scene. Choose exactly one label from: "
            + ", ".join(closed)
            + ". Return only the scene label."
        )
    n, path = _write(rows, "acoustic_scene_dcase2013_scenes.jsonl")
    return n, path, {"labels": closed, "missing_audio": 0}


def build_tut2017() -> tuple[int, Path, dict]:
    rows = []
    for line in TUT2017_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        wav = Path(str(doc.get("audio") or ""))
        label = str(
            doc.get("official_scene_label")
            or doc.get("acoustic_scene_label")
            or doc.get("scene_label")
            or "street"
        ).strip()
        sid = str(doc.get("id") or "").strip()
        if not sid or not wav.is_file():
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "tut_sound_events_2017",
                "benchmark": "tut_sound_events_2017",
                "capability": "acoustic_scene",
                "protocol": "acoustic_scene",
                "subset": str(doc.get("split") or ""),
                "split": str(doc.get("split") or ""),
                "WavPath": str(wav),
                "text": label,
                "answer": label,
                "closed_labels": ["street"],
                "question": (
                    "Identify the acoustic scene. Choose exactly one label from: street. "
                    "Return only the scene label."
                ),
                "source_index": str(TUT2017_MANIFEST),
                "note": "TUT2017 ASC view is street-only in the L0 manifest.",
            }
        )
    n, path = _write(rows, "acoustic_scene_tut_sound_events_2017.jsonl")
    return n, path, {"labels": ["street"], "missing_audio": 0}


def build_tut2018() -> tuple[int, Path, dict]:
    meta = TUT2018_ROOT / "meta.csv"
    audio_dir = TUT2018_ROOT / "audio"
    # Prefer evaluate fold file list when present.
    eval_list = TUT2018_ROOT / "evaluation_setup" / "fold1_evaluate.txt"
    allow: set[str] | None = None
    if eval_list.is_file():
        allow = set()
        for line in eval_list.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lower().startswith("filename"):
                continue
            fn = line.split("\t")[0].strip()
            allow.add(Path(fn).name)

    rows = []
    missing = 0
    present_by_scene: dict[str, int] = {}
    with meta.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            fn = str(row.get("filename") or "").strip()
            scene = str(row.get("scene_label") or "").strip()
            if not fn or not scene:
                continue
            name = Path(fn).name
            if allow is not None and name not in allow:
                continue
            wav = audio_dir / name
            if not wav.is_file():
                alt = TUT2018_ROOT / fn
                wav = alt if alt.is_file() else wav
            if not wav.is_file():
                missing += 1
                continue
            present_by_scene[scene] = present_by_scene.get(scene, 0) + 1
            sid = f"tut2018_{wav.stem}"
            rows.append(
                {
                    "schema_version": "full-index/1.0",
                    "sample_id": sid,
                    "dataset": "tut_urban_acoustic_scenes_2018_mobile",
                    "benchmark": "tut_urban_acoustic_scenes_2018_mobile",
                    "capability": "acoustic_scene",
                    "protocol": "acoustic_scene",
                    "subset": "fold1_evaluate" if allow is not None else "development",
                    "split": "dev",
                    "WavPath": str(wav),
                    "text": scene,
                    "answer": scene,
                    "closed_labels": list(TUT2018_OFFICIAL),
                    "question": (
                        "Identify the acoustic scene. Choose exactly one label from: "
                        + ", ".join(TUT2018_OFFICIAL)
                        + ". Return only the scene label."
                    ),
                    "source_index": str(meta),
                    "note": (
                        "Local audio currently covers a subset of official scenes; "
                        "closed_labels still list all 10 official classes."
                    ),
                }
            )
    n, path = _write(rows, "acoustic_scene_tut_urban_acoustic_scenes_2018_mobile.jsonl")
    return n, path, {
        "labels": TUT2018_OFFICIAL,
        "present_by_scene": present_by_scene,
        "missing_audio": missing,
        "fold": "fold1_evaluate" if allow is not None else "all_meta",
    }


def _parse_mmsu_scene(doc: dict) -> tuple[str | None, str, list[str], str]:
    messages = doc.get("messages") or []
    if len(messages) < 2:
        return None, "", [], ""
    user = (messages[0].get("content") or [{}])[0]
    asst = (messages[1].get("content") or [{}])[0]
    rel = str(user.get("audio_path") or "").strip() or None
    raw_q = str(user.get("text") or "").strip()
    answer = str(asst.get("text") or "").strip()
    opts = re.findall(r"^([A-D])\.\s*(.+)$", raw_q, flags=re.M)
    stem = re.split(r"\nA\.\s*", raw_q, maxsplit=1)[0].strip()
    labels = [text.strip() for _, text in opts]
    choice_map = {k: v.strip() for k, v in opts}
    if answer in choice_map:
        answer = choice_map[answer]
    return rel, stem, labels, answer


def build_mmsu() -> tuple[int, Path, dict]:
    rows = []
    missing = 0
    for line in MMSU_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        if not sid.startswith("MMSU_background_scene_recognition_"):
            continue
        rel, stem, labels, answer = _parse_mmsu_scene(doc)
        if not rel or not answer or len(labels) < 2:
            continue
        wav = MMSU_ROOT / rel
        if not wav.is_file():
            alt = MMSU_ROOT / "audio" / Path(rel).name
            wav = alt if alt.is_file() else wav
        if not wav.is_file():
            missing += 1
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "mmsu",
                "benchmark": "mmsu",
                "capability": "acoustic_scene",
                "protocol": "acoustic_scene",
                "subset": "background_scene_recognition",
                "split": "test",
                "WavPath": str(wav),
                "text": answer,
                "answer": answer,
                "closed_labels": labels,
                "choice_a": labels[0] if len(labels) > 0 else "",
                "choice_b": labels[1] if len(labels) > 1 else "",
                "choice_c": labels[2] if len(labels) > 2 else "",
                "choice_d": labels[3] if len(labels) > 3 else "",
                "question": (
                    f"{stem} Choose exactly one label from: {', '.join(labels)}. "
                    f"Return only the scene label."
                ),
                "source_index": str(MMSU_JSONL),
            }
        )
    n, path = _write(rows, "acoustic_scene_mmsu.jsonl")
    return n, path, {"missing_audio": missing}


def main() -> int:
    summary = {}
    for name, fn in [
        ("dcase2013_scenes", build_dcase),
        ("mmsu", build_mmsu),
        ("tut_sound_events_2017", build_tut2017),
        ("tut_urban_acoustic_scenes_2018_mobile", build_tut2018),
    ]:
        n, path, meta = fn()
        summary[name] = {"records": n, "output": str(path), **meta}
        print(json.dumps({"benchmark": name, "records": n, "output": str(path), **meta}, ensure_ascii=False))
    print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
