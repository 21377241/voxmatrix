#!/usr/bin/env python3
"""Build full sound_event indexes (closed-set + multilabel).

Tracks align with Native catalog / 8p31 乙-SE:
  - ESC-50 fold5 closed-set (50 labels)
  - MMSU non_verbal_sound_detection (per-item 4 options → event label)
  - TUT2017 evaluation event clips + official 6-class closed-set
  - MAVD-traffic test event clips + window-overlap multilabel
  - SONYC-UST v2.3 test_verified multilabel tagging
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mesh_eval/data/full/sound_event"

ESC_ROOT = Path(
    "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/ESC-50/ESC-50-master"
)
MMSU_JSONL = Path("/mnt/afs/eval_data/benchmarks/MMSU/mmsu_audioqa.jsonl")
MMSU_ROOT = Path("/mnt/afs/eval_data/benchmarks/MMSU")
TUT_CLIPS = Path(
    "/mnt/afs/eval_data/benchmarks/TUT-Sound-Events-2017/event_clips/index_evaluation.jsonl"
)
MAVD_CLIPS = Path(
    "/mnt/afs/eval_data/benchmarks/MAVD-traffic/event_clips/index_test.jsonl"
)
MAVD_MANIFEST = Path(
    "/mnt/afs/eval_data/benchmarks/MAVD-traffic/manifests/sound_event.jsonl"
)
SONYC_MANIFEST = Path(
    "/mnt/afs/eval_data/benchmarks/SONYC-UST-v2.3/manifests/sound_event_tagging.jsonl"
)
SONYC_ANN = Path("/mnt/afs/eval_data/benchmarks/SONYC-UST-v2.3/annotations.csv")

TUT_CLOSED = [
    "brakes squeaking",
    "car",
    "children",
    "large vehicle",
    "people speaking",
    "people walking",
]


def _write(rows: list[dict], name: str) -> tuple[int, Path]:
    from mesh_eval.data.full_index_meta import slot_id_from_filename, write_index_jsonl

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    slot_id = slot_id_from_filename(OUT.name, name)
    n = write_index_jsonl(path, rows, slot_id)
    return n, path


def build_esc50() -> tuple[int, Path, dict]:
    meta = ESC_ROOT / "meta" / "esc50.csv"
    audio = ESC_ROOT / "audio"
    closed = sorted(
        {
            str(r["category"]).strip()
            for r in csv.DictReader(meta.open(encoding="utf-8"))
            if str(r.get("category") or "").strip()
        }
    )
    prompt = (
        "Return exactly one ESC-50 label from this list: "
        + ", ".join(closed)
        + ". Return only the event label."
    )
    rows = []
    missing = 0
    with meta.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("fold")) != "5":
                continue
            wav = audio / row["filename"]
            label = str(row["category"]).strip()
            if not wav.is_file():
                missing += 1
                continue
            rows.append(
                {
                    "schema_version": "full-index/1.0",
                    "sample_id": f"esc50_{Path(row['filename']).stem}",
                    "dataset": "esc50",
                    "benchmark": "esc50",
                    "capability": "sound_event",
                    "protocol": "sound_event",
                    "subset": "fold5",
                    "split": "test",
                    "WavPath": str(wav),
                    "text": label,
                    "answer": label,
                    "closed_labels": closed,
                    "question": prompt,
                    "source_index": str(meta),
                }
            )
    n, path = _write(rows, "sound_event_esc50.jsonl")
    return n, path, {"labels": closed, "missing_audio": missing}


def build_mmsu() -> tuple[int, Path, dict]:
    rows = []
    missing = 0
    for line in MMSU_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        if "non_verbal_sound_detection" not in sid:
            continue
        messages = doc.get("messages") or []
        if len(messages) < 2:
            continue
        user = (messages[0].get("content") or [{}])[0]
        asst = (messages[1].get("content") or [{}])[0]
        rel = str(user.get("audio_path") or "").strip()
        raw_q = str(user.get("text") or "").strip()
        answer = str(asst.get("text") or "").strip()
        opts = re.findall(r"^([A-D])\.\s*(.+)$", raw_q, flags=re.M)
        stem = re.split(r"\nA\.\s*", raw_q, maxsplit=1)[0].strip()
        labels = [t.strip() for _, t in opts]
        choice_map = {k: v.strip() for k, v in opts}
        if answer in choice_map:
            answer = choice_map[answer]
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
                "capability": "sound_event",
                "protocol": "sound_event",
                "subset": "non_verbal_sound_detection",
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
                    f"Return only the event label."
                ),
                "source_index": str(MMSU_JSONL),
            }
        )
    n, path = _write(rows, "sound_event_mmsu.jsonl")
    return n, path, {"n_options_mode": "per_item_4", "missing_audio": missing}


def build_tut2017() -> tuple[int, Path, dict]:
    prompt = (
        "Identify the labelled sound event. Choose exactly one label from: "
        + ", ".join(TUT_CLOSED)
        + ". Return only the event label."
    )
    rows = []
    missing = 0
    for line in TUT_CLIPS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        wav = Path(str(doc.get("WavPath") or ""))
        label = str(doc.get("event_label") or "").strip()
        sid = str(doc.get("id") or "").strip()
        if not sid or not label:
            continue
        if not wav.is_file():
            missing += 1
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "tut_sound_events_2017",
                "benchmark": "tut_sound_events_2017",
                "capability": "sound_event",
                "protocol": "evaluation_slice_closed_set",
                "subset": "evaluation",
                "split": "evaluation",
                "WavPath": str(wav),
                "text": label,
                "answer": label,
                "closed_labels": TUT_CLOSED,
                "question": prompt,
                "start_seconds": doc.get("start_seconds"),
                "end_seconds": doc.get("end_seconds"),
                "source_index": str(TUT_CLIPS),
            }
        )
    n, path = _write(rows, "sound_event_tut_sound_events_2017.jsonl")
    return n, path, {"labels": TUT_CLOSED, "missing_audio": missing}


def _mavd_overlap_labels() -> dict[str, list[str]]:
    """primary event id -> sorted unique labels overlapping that window."""
    events: list[dict] = []
    for line in MAVD_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if doc.get("split") != "test":
            continue
        sid = str(doc.get("id") or "")
        label = str(doc.get("event_label") or "").strip()
        start, end = doc.get("start_seconds"), doc.get("end_seconds")
        audio = str(doc.get("audio") or "")
        if not sid or not label or start is None or end is None:
            continue
        events.append(
            {
                "id": sid,
                "label": label,
                "start": float(start),
                "end": float(end),
                "audio": audio,
            }
        )
    by_audio: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_audio[ev["audio"]].append(ev)
    out: dict[str, list[str]] = {}
    for audio, group in by_audio.items():
        for ev in group:
            labs = {
                other["label"]
                for other in group
                if other["start"] < ev["end"] and other["end"] > ev["start"]
            }
            out[ev["id"]] = sorted(labs)
    return out


def build_mavd() -> tuple[int, Path, dict]:
    overlap = _mavd_overlap_labels()
    closed = sorted({lab for labs in overlap.values() for lab in labs})
    # Prefer Full ontology if we already know it; fall back to observed.
    if len(closed) < 10:
        closed = sorted(
            {
                str(doc.get("event_label") or "").strip()
                for line in MAVD_MANIFEST.read_text(encoding="utf-8").splitlines()
                if line.strip()
                for doc in [json.loads(line)]
                if doc.get("split") == "test" and doc.get("event_label")
            }
        )
    prompt = (
        "Identify all traffic sound events present in the audio. "
        "Choose zero or more labels from this closed list only: "
        + ", ".join(closed)
        + ". Return a comma-separated list of labels from the list, "
        "with no explanations."
    )
    rows = []
    missing = 0
    for line in MAVD_CLIPS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        sid = str(doc.get("id") or "")
        wav = Path(str(doc.get("WavPath") or ""))
        primary = str(doc.get("event_label") or "").strip()
        labels = overlap.get(sid) or ([primary] if primary else [])
        if not sid or not labels:
            continue
        if not wav.is_file():
            missing += 1
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": sid,
                "dataset": "mavd_traffic",
                "benchmark": "mavd_traffic",
                "capability": "sound_event",
                "protocol": "event_slice_multilabel",
                "subset": "test",
                "split": "test",
                "WavPath": str(wav),
                "labels": labels,
                "answer": labels,
                "text": ", ".join(labels),
                "primary_label": primary,
                "closed_labels": closed,
                "question": prompt,
                "start_seconds": doc.get("start_seconds"),
                "end_seconds": doc.get("end_seconds"),
                "source_index": str(MAVD_CLIPS),
            }
        )
    n, path = _write(rows, "sound_event_mavd_traffic.jsonl")
    return n, path, {"missing_audio": missing, "multilabel": True, "n_closed": len(closed)}


def _sonyc_col_to_label(col: str) -> str | None:
    # e.g. 7-1_person-or-small-group-talking_presence -> person or small group talking
    if not col.endswith("_presence"):
        return None
    body = col[: -len("_presence")]
    # drop leading taxonomy code like "7-1_" or "7_"
    body = re.sub(r"^\d+(?:-\d+|-[Xx])?_", "", body)
    body = body.replace("-", " ").replace("_", " ").strip()
    return body or None


def build_sonyc() -> tuple[int, Path, dict]:
    by_file: dict[str, set[str]] = defaultdict(set)
    if SONYC_ANN.is_file():
        with SONYC_ANN.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "test":
                    continue
                # Prefer city-consensus / verified annotator 0 when present.
                ann_id = str(row.get("annotator_id") or "")
                if ann_id not in ("0", "0.0"):
                    continue
                fn = row.get("audio_filename") or ""
                for col, val in row.items():
                    if str(val) != "1":
                        continue
                    lab = _sonyc_col_to_label(col)
                    if lab:
                        by_file[fn].add(lab)
    closed = sorted({lab for labs in by_file.values() for lab in labs})
    prompt = (
        "Name the prominent urban sound event(s) you hear. "
        "Choose zero or more labels from this closed list only: "
        + ", ".join(closed)
        + ". Return a comma-separated list of labels from the list, "
        "with no explanations."
    )
    rows = []
    missing = 0
    for line in SONYC_MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        if doc.get("split") != "test":
            continue
        if doc.get("verified") is False:
            continue
        audio = Path(str(doc.get("audio") or ""))
        fn = doc.get("audio_filename") or audio.name
        labels = sorted(by_file.get(fn) or [])
        if not labels or not audio.is_file():
            if not audio.is_file():
                missing += 1
            continue
        rows.append(
            {
                "schema_version": "full-index/1.0",
                "sample_id": str(doc.get("id") or f"sonyc_{fn}"),
                "dataset": "sonyc_ust_v23",
                "benchmark": "sonyc_ust_v23",
                "capability": "sound_event",
                "protocol": "multilabel_f1",
                "subset": "test_verified",
                "split": "test",
                "WavPath": str(audio),
                "labels": labels,
                "answer": labels,
                "text": ", ".join(labels),
                "closed_labels": closed,
                "question": prompt,
                "source_index": str(SONYC_MANIFEST),
            }
        )
    n, path = _write(rows, "sound_event_sonyc_ust_v23.jsonl")
    return n, path, {"missing_audio": missing, "multilabel": True, "n_closed": len(closed)}


def main() -> None:
    summary = {}
    for name, fn in [
        ("esc50", build_esc50),
        ("mmsu", build_mmsu),
        ("tut_sound_events_2017", build_tut2017),
        ("mavd_traffic", build_mavd),
        ("sonyc_ust_v23", build_sonyc),
    ]:
        n, path, meta = fn()
        summary[name] = {"n": n, "path": str(path), **meta}
        print(f"[full-sound-event] {name}: n={n} -> {path} meta={meta}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
