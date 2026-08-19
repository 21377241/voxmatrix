import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import soundfile as sf

from audio_evals.dataset.huggingface import _exclusive_lock
from audio_evals.dataset.prepared import (
    PREPARED_FORMAT_VERSION,
    file_sha256,
    prepared_audio_paths,
)
from audio_evals.registry import registry

def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                default=_json_default,
            )
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


def _normalize_paths(record: Dict[str, Any], cwd: str) -> Dict[str, Any]:
    doc = dict(record)
    for key, value in list(doc.items()):
        if (key == "WavPath" or re.fullmatch(r"WavPath\d+", key)) and value:
            doc[key] = (
                str(value)
                if os.path.isabs(str(value))
                else os.path.abspath(os.path.join(cwd, str(value)))
            )
    input_obj = doc.get("input")
    if isinstance(input_obj, dict):
        input_obj = dict(input_obj)
        for key in ("audio_path", "audio_path_b"):
            if input_obj.get(key) and not os.path.isabs(str(input_obj[key])):
                input_obj[key] = os.path.abspath(
                    os.path.join(cwd, str(input_obj[key]))
                )
        audios = []
        for item in input_obj.get("audios") or []:
            if not isinstance(item, dict):
                audios.append(item)
                continue
            item = dict(item)
            if item.get("uri") and not os.path.isabs(str(item["uri"])):
                item["uri"] = os.path.abspath(
                    os.path.join(cwd, str(item["uri"]))
                )
            audios.append(item)
        if audios:
            input_obj["audios"] = audios
        audio = []
        for item in input_obj.get("audio") or []:
            if not isinstance(item, dict):
                audio.append(item)
                continue
            item = dict(item)
            if item.get("uri") and not os.path.isabs(str(item["uri"])):
                item["uri"] = os.path.abspath(
                    os.path.join(cwd, str(item["uri"]))
                )
            audio.append(item)
        if audio:
            input_obj["audio"] = audio
        doc["input"] = input_obj
    return doc


def _prepare_rows(
    rows: Iterable[Dict[str, Any]], dataset_name: str, offset: int, cwd: str
) -> List[Dict[str, Any]]:
    prepared = []
    for local_index, row in enumerate(rows):
        doc = _normalize_paths(dict(row), cwd)
        embedded = [
            key
            for key in doc
            if key == "audio" or (key.startswith("audio") and key[5:].isdigit())
        ]
        if embedded:
            raise ValueError(
                f"prepared row still contains decoded audio columns: {embedded}"
            )
        paths = prepared_audio_paths(doc)
        if not paths:
            raise ValueError(
                f"dataset {dataset_name} row {offset + local_index} has no audio path"
            )
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise ValueError(
                f"dataset {dataset_name} row {offset + local_index} has missing audio: {missing}"
            )
        doc.setdefault(
            "sample_id", f"{dataset_name}_{offset + local_index:08d}"
        )
        doc.setdefault("dataset", dataset_name)
        doc["asset_schema_version"] = PREPARED_FORMAT_VERSION
        prepared.append(doc)
    return prepared


def _write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=_json_default,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _asset_inventory(rows: List[Dict[str, Any]], checksum: str) -> List[Dict[str, Any]]:
    paths = sorted(
        {path for row in rows for path in prepared_audio_paths(row)}
    )
    inventory = []
    for path in paths:
        info = sf.info(path)
        item = {
            "path": path,
            "size_bytes": os.path.getsize(path),
            "frames": info.frames,
            "sampling_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
        }
        if checksum == "sha256":
            item["sha256"] = file_sha256(path)
        inventory.append(item)
    return inventory


def prepare_dataset(
    *,
    dataset_name: str,
    output_dir: str,
    limit: int = 0,
    offset: int = 0,
    checksum: str = "sha256",
) -> Dict[str, Any]:
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    metadata_path = output_root / "metadata.json"
    success_path = output_root / "_SUCCESS"
    lock_path = str(output_root / ".prepare.lock")

    with _exclusive_lock(lock_path):
        try:
            success_path.unlink()
        except FileNotFoundError:
            pass

        dataset = registry.get_dataset(dataset_name)
        if dataset is None:
            raise KeyError(f"dataset is not registered: {dataset_name}")
        if hasattr(dataset, "audio_cache_root"):
            dataset.audio_cache_root = str(output_root / "assets")
        loader = getattr(dataset, "load_slice", None)
        if callable(loader):
            rows = loader(offset=offset, limit=limit)
        else:
            load_limit = offset + limit if limit else 0
            loaded = dataset.load(load_limit)
            rows = loaded[offset : offset + limit if limit else None]
        prepared = _prepare_rows(rows, dataset_name, offset, os.getcwd())
        _write_manifest(manifest_path, prepared)
        assets = _asset_inventory(prepared, checksum)
        manifest_digest = file_sha256(str(manifest_path))
        source_metadata = getattr(dataset, "last_load_metadata", {})
        metadata = {
            "format_version": PREPARED_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset_name,
            "offset": offset,
            "limit": limit,
            "row_count": len(prepared),
            "audio_asset_count": len(assets),
            "checksum": checksum,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "source": source_metadata,
            "assets": assets,
        }
        _atomic_json(metadata_path, metadata)
        success = {
            "format_version": PREPARED_FORMAT_VERSION,
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "metadata_sha256": file_sha256(str(metadata_path)),
            "row_count": len(prepared),
            "audio_asset_count": len(assets),
        }
        _atomic_json(success_path, success)
        return success


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a registry dataset as stable WAV assets plus JSONL manifest"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registry-path", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--checksum", choices=["sha256", "none"], default="sha256")
    args = parser.parse_args()
    if args.registry_path:
        registry.add_registry_paths(args.registry_path.split())
    result = prepare_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        limit=args.limit,
        offset=args.offset,
        checksum=args.checksum,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
