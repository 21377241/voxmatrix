#!/usr/bin/env python3
"""Make every Native slot a strict subset of its Full index.

Full is the source of truth.  For each runnable NativeClipDataset slot in
suites/native/catalog.yaml:

  1. resolve the Full file (``full/<capability>/<native basename>`` unless
     listed in FULL_OVERRIDES);
  2. match each Native row to a Full row (same audio path, else same id with
     the same question or reference);
  3. replace matched rows with the Full row, drop unmatched rows and top up
     with evenly spaced Full rows so the Native size is unchanged;
  4. keep the Native WavPath when the Full row points at a forbidden marker
     (utt_clips): those crops were verified sample-identical.

Writes mesh_eval/data/full/slot_map.yaml (slot -> native/full file + counts)
and leaves Native files untouched when nothing evaluation-relevant changes.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "suites" / "native" / "catalog.yaml"
REGISTRY_GLOB = str(ROOT / "mesh_eval" / "registry" / "dataset" / "*.yaml")
FULL = ROOT / "mesh_eval" / "data" / "full"
SLOT_MAP = FULL / "slot_map.yaml"

FULL_OVERRIDES = {
    "asr__wenetspeech": "asr/asr_wenetspeech_test_meeting.jsonl",
    "speech_translation__covost2": "speech_translation/speech_translation_covost2_zh_en.jsonl",
}
ID_PREFIXES = {"asr__aishell_5": "aishell5_"}
NATIVE_ONLY_PATH_MARKERS = ("utt_clips",)
EVAL_KEYS = (
    "WavPath",
    "WavPath2",
    "question",
    "text",
    "answer",
    "labels",
    "captions",
    "closed_labels",
    "protocol",
    "start_time",
    "end_time",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def runnable_slots() -> list[tuple[str, str, Path]]:
    registry: dict = {}
    for path in sorted(glob.glob(REGISTRY_GLOB)):
        registry.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    slots = []
    for item in catalog["capabilities"]:
        if item.get("status") not in ("ok", "diagnostic"):
            continue
        args = (registry.get(item["dataset"]) or {}).get("args") or {}
        f_name = str(args.get("f_name") or "")
        if not f_name.startswith("mesh_eval/data/native/"):
            continue
        slots.append((f"{item['capability']}__{item['benchmark']}", item["capability"], ROOT / f_name))
    return slots


def full_path_for(slot: str, capability: str, native: Path) -> Path:
    rel = FULL_OVERRIDES.get(slot)
    return FULL / rel if rel else FULL / capability / native.name


def spread(rows: list[dict], count: int) -> list[dict]:
    if count <= 0 or not rows:
        return []
    if len(rows) <= count:
        return list(rows)
    if count == 1:
        return [rows[0]]
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def match_rows(slot: str, native: list[dict], full: list[dict]) -> list[dict | None]:
    by_wav: dict[str, list[dict]] = {}
    by_id: dict[str, dict] = {}
    for row in full:
        key = (str(row.get("WavPath") or ""), str(row.get("WavPath2") or ""))
        by_wav.setdefault(key, []).append(row)
        by_id[str(row.get("sample_id"))] = row
    prefix = ID_PREFIXES.get(slot, "")
    out: list[dict | None] = []
    for row in native:
        hit = None
        wav2 = str(row.get("WavPath2") or "")
        for wav in (row.get("audio_locator"), row.get("WavPath")):
            cands = by_wav.get((str(wav or ""), wav2)) or []
            if len(cands) == 1:
                hit = cands[0]
                break
        if hit is None:
            sid = str(row.get("sample_id"))
            cand = by_id.get(prefix + sid) or by_id.get(sid)
            if cand is not None and (
                (row.get("question") and row.get("question") == cand.get("question"))
                or (row.get("text") not in (None, "") and row.get("text") == cand.get("text"))
            ):
                hit = cand
        out.append(hit)
    return out


def aligned_row(native_row: dict, full_row: dict) -> dict:
    row = dict(full_row)
    full_wav = str(full_row.get("WavPath") or "")
    if any(marker in full_wav for marker in NATIVE_ONLY_PATH_MARKERS):
        row["WavPath"] = native_row["WavPath"]
        row["full_WavPath"] = full_wav
    return row


def eval_view(row: dict) -> dict:
    return {key: row.get(key) for key in EVAL_KEYS} | {"sample_id": row.get("sample_id")}


def align_slot(slot: str, capability: str, native_path: Path, dry_run: bool) -> dict:
    full_path = full_path_for(slot, capability, native_path)
    entry = {
        "native": str(native_path.relative_to(ROOT)),
        "full": str(full_path.relative_to(ROOT)) if full_path.is_file() else None,
    }
    if not full_path.is_file():
        entry["status"] = "missing_full"
        return entry
    native = load_jsonl(native_path)
    full = load_jsonl(full_path)
    matches = match_rows(slot, native, full)
    kept = [aligned_row(n, f) for n, f in zip(native, matches) if f is not None]
    used = {str(r.get("sample_id")) for r in kept}
    pool = [r for r in full if str(r.get("sample_id")) not in used]
    extra = spread(pool, len(native) - len(kept))
    new_rows = kept + extra
    entry.update(
        native_rows=len(native),
        full_rows=len(full),
        kept=len(kept),
        replaced=len(extra),
    )
    changed = len(new_rows) != len(native) or any(
        eval_view(a) != eval_view(b) for a, b in zip(native, new_rows)
    )
    entry["status"] = "rewritten" if changed else "aligned"
    if changed and not dry_run:
        write_jsonl(native_path, new_rows)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = {}
    for slot, capability, native_path in runnable_slots():
        entry = align_slot(slot, capability, native_path, args.dry_run)
        report[slot] = entry
        print(
            f"{slot:58s} {entry['status']:12s} "
            f"kept={entry.get('kept', '-')}/{entry.get('native_rows', '-')} "
            f"full={entry.get('full_rows', '-')}"
        )
    missing = [s for s, e in report.items() if e["status"] == "missing_full"]
    if not args.dry_run:
        SLOT_MAP.write_text(
            yaml.safe_dump(
                {s: {k: e[k] for k in ("native", "full", "full_rows") if k in e} for s, e in report.items()},
                allow_unicode=True,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        print(f"wrote {SLOT_MAP.relative_to(ROOT)}")
    if missing:
        print("missing full:", ", ".join(missing), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
