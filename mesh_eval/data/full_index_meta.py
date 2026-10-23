"""Subset-manifest metadata for Full / Native indexes.

Full builders historically wrote scoring fields only. Annotation conditions
(language / scenario / device / …) live in
``annotation/benchmark_subset_manifest.jsonl`` at subset granularity. Call
``finalize_jsonl`` after writing an index so those fields are always present.

Rules
-----
- Manifest values may be ``unknown`` / ``none`` — still write them.
- Never overwrite a non-empty sample value (runtime loaders may be richer).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = REPO_ROOT / "suites" / "native" / "catalog.yaml"
SLOT_MAP_PATH = REPO_ROOT / "mesh_eval" / "data" / "full" / "slot_map.yaml"
MANIFEST_PATH = REPO_ROOT / "annotation" / "benchmark_subset_manifest.jsonl"

# spoken_slu catalog slots historically lack subset_record_id; reuse IF records.
FALLBACK_RECORD_IDS = {
    "spoken_slu__fluent_speech_commands": (
        "fluent_speech_commands__test__test__instruction_following__"
        "instruction_following__unknown__unknown__bfe63bb2e6"
    ),
    "spoken_slu__slurp": (
        "slurp__test__test__instruction_following__instruction_following__"
        "unknown__unknown__de3affb6d6"
    ),
}

REQUIRED_TOP = (
    "sample_id",
    "WavPath",
    "capability",
    "dataset",
    "language",
    "scenario",
    "metadata",
    "subset_manifest_record_id",
)
REQUIRED_SCENARIO = ("primary", "secondary")
REQUIRED_METADATA = (
    "language",
    "device_type",
    "recording_setup",
    "noise_type",
    "speaker_count",
    "overlap_level",
    "snr_bucket",
    "distance",
    "motion_state",
    "crowd_density",
    "weather",
    "window_state",
    "authorization_state",
    "speaker_age_group",
    "speaker_role",
    "language_mix",
)


def _absent(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, dict) and not value:
        return True
    return False


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["record_id"]] = row
    return out


@lru_cache(maxsize=1)
def load_catalog_subset_ids() -> dict[str, str]:
    catalog = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8")) or {}
    mapping: dict[str, str] = {}
    for item in catalog.get("capabilities") or []:
        slot_id = str(item.get("id") or "")
        record_id = item.get("subset_record_id") or FALLBACK_RECORD_IDS.get(slot_id)
        if slot_id and record_id:
            mapping[slot_id] = str(record_id)
    for slot_id, record_id in FALLBACK_RECORD_IDS.items():
        mapping.setdefault(slot_id, record_id)
    return mapping


def resolve_record_id(slot_id: str) -> Optional[str]:
    return load_catalog_subset_ids().get(slot_id)


def enrich_row(row: Mapping[str, Any], record: Mapping[str, Any]) -> dict:
    out = dict(row)
    meta = dict(out.get("metadata") or {})
    scen = (
        dict(out.get("scenario") or {})
        if isinstance(out.get("scenario"), dict)
        else {}
    )
    documented_meta = dict(record.get("metadata") or {})
    documented_scen = dict(record.get("scenario") or {})

    if _absent(out.get("language")):
        lang = documented_meta.get("language")
        if lang is not None and str(lang).strip() != "":
            out["language"] = lang

    if _absent(scen.get("primary")) and documented_scen.get("primary") is not None:
        scen["primary"] = documented_scen["primary"]
    if _absent(scen.get("secondary")) and documented_scen.get("secondary") is not None:
        scen["secondary"] = documented_scen["secondary"]
    if documented_scen or scen:
        scen.setdefault("primary", "unknown")
        scen.setdefault("secondary", "unknown")
        out["scenario"] = scen

    for key, value in documented_meta.items():
        if value is None:
            continue
        if _absent(meta.get(key)):
            meta[key] = value
    if meta:
        out["metadata"] = meta

    rid = record.get("record_id")
    if rid:
        out["subset_manifest_record_id"] = rid

    if not _absent(meta.get("language")) and _absent(out.get("metadata__language")):
        out["metadata__language"] = meta["language"]
    if not _absent(scen.get("primary")) and _absent(out.get("scenario__primary")):
        out["scenario__primary"] = scen["primary"]
    return out


def finalize_rows(rows: Iterable[Mapping[str, Any]], slot_id: str) -> list[dict]:
    record_id = resolve_record_id(slot_id)
    if not record_id:
        raise KeyError(f"no subset_record_id for slot {slot_id}")
    record = load_manifest().get(record_id)
    if not record:
        raise KeyError(f"subset manifest missing record {record_id} for {slot_id}")
    return [enrich_row(row, record) for row in rows]


def slot_id_from_filename(capability: str, filename: str) -> str:
    """Map ``{capability}_{benchmark}.jsonl`` -> ``{capability}__{benchmark}``."""
    stem = Path(filename).stem
    prefix = f"{capability}_"
    if stem.startswith(prefix):
        return f"{capability}__{stem[len(prefix):]}"
    return f"{capability}__{stem}"


def write_index_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    slot_id: str,
) -> int:
    """Write rows with subset-manifest fields attached (builders should use this)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enriched = finalize_rows(rows, slot_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)
    return len(enriched)


def finalize_jsonl(path: Path, slot_id: str) -> dict[str, int]:
    """Rewrite ``path`` in place with subset-manifest metadata attached."""
    path = Path(path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    enriched = finalize_rows(rows, slot_id)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in enriched:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)
    return {"rows": len(enriched), "path": str(path)}


def audit_row(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP:
        if key not in row or _absent(row.get(key)):
            errors.append(f"missing {key}")
    scen = row.get("scenario")
    if not isinstance(scen, dict):
        errors.append("scenario not object")
    else:
        for key in REQUIRED_SCENARIO:
            if key not in scen:
                errors.append(f"scenario missing {key}")
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        errors.append("metadata not object")
    else:
        for key in REQUIRED_METADATA:
            if key not in meta:
                errors.append(f"metadata missing {key}")
    return errors


def audit_slot_map(slot_map_path: Path = SLOT_MAP_PATH) -> list[str]:
    slot_map = yaml.safe_load(Path(slot_map_path).read_text(encoding="utf-8")) or {}
    problems: list[str] = []
    for slot_id, entry in sorted(slot_map.items()):
        for kind in ("full", "native"):
            rel = entry.get(kind)
            if not rel:
                continue
            path = REPO_ROOT / rel
            if not path.is_file():
                problems.append(f"{slot_id} {kind}: missing {rel}")
                continue
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            for err in audit_row(row):
                problems.append(f"{slot_id} {kind}: {err}")
    return problems


def finalize_all_slot_map_files(slot_map_path: Path = SLOT_MAP_PATH) -> list[str]:
    """Attach subset-manifest fields to every Full/Native file in slot_map."""
    slot_map = yaml.safe_load(Path(slot_map_path).read_text(encoding="utf-8")) or {}
    done: list[str] = []
    for slot_id, entry in sorted(slot_map.items()):
        if not resolve_record_id(slot_id):
            continue
        for kind in ("full", "native"):
            rel = entry.get(kind)
            if not rel:
                continue
            path = REPO_ROOT / rel
            if not path.is_file():
                continue
            finalize_jsonl(path, slot_id)
            done.append(f"{slot_id}:{kind}")
    return done
