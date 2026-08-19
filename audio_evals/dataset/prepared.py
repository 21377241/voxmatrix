import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from audio_evals.dataset.dataset import Dataset

PREPARED_FORMAT_VERSION = "prepared-audio/1.0"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepared_audio_paths(record: Dict[str, Any]) -> List[str]:
    paths: List[str] = []
    for key, value in record.items():
        if (key == "WavPath" or re.fullmatch(r"WavPath\d+", key)) and value:
            paths.append(str(value))
    input_obj = record.get("input") or {}
    for key in ("audio_path", "audio_path_b"):
        if input_obj.get(key):
            paths.append(str(input_obj[key]))
    for item in input_obj.get("audios") or []:
        if isinstance(item, dict) and item.get("uri"):
            paths.append(str(item["uri"]))
    for item in input_obj.get("audio") or []:
        if isinstance(item, dict) and item.get("uri"):
            paths.append(str(item["uri"]))
    return list(dict.fromkeys(paths))


class PreparedAudioJsonl(Dataset):
    """Read a completed prepared-audio manifest without HF audio decoding."""

    def __init__(
        self,
        f_name: str,
        default_task: str,
        ref_col: str,
        col_aliases=None,
        require_success: bool = True,
        check_audio_exists: bool = True,
        verify_asset_checksums: bool = False,
    ):
        super().__init__(default_task, ref_col, col_aliases)
        self.f_name = os.path.abspath(f_name)
        self.require_success = require_success
        self.check_audio_exists = check_audio_exists
        self.verify_asset_checksums = verify_asset_checksums
        self._validated = False

    @property
    def root(self) -> str:
        return os.path.dirname(self.f_name)

    def _validate_success(self) -> None:
        if self._validated:
            return
        success_path = os.path.join(self.root, "_SUCCESS")
        if self.require_success and not os.path.isfile(success_path):
            raise ValueError(f"prepared dataset is incomplete; missing {success_path}")
        if os.path.isfile(success_path):
            with open(success_path, encoding="utf-8") as handle:
                success = json.load(handle)
            if success.get("format_version") != PREPARED_FORMAT_VERSION:
                raise ValueError(
                    "unsupported prepared dataset format: "
                    f"{success.get('format_version')!r}"
                )
            expected = success.get("manifest_sha256")
            if expected and file_sha256(self.f_name) != expected:
                raise ValueError(
                    f"prepared manifest checksum mismatch: {self.f_name}"
                )
            metadata_path = os.path.join(self.root, "metadata.json")
            expected_metadata = success.get("metadata_sha256")
            if expected_metadata:
                if not os.path.isfile(metadata_path):
                    raise ValueError(
                        f"prepared dataset is incomplete; missing {metadata_path}"
                    )
                if file_sha256(metadata_path) != expected_metadata:
                    raise ValueError(
                        f"prepared metadata checksum mismatch: {metadata_path}"
                    )
            if self.verify_asset_checksums and os.path.isfile(metadata_path):
                with open(metadata_path, encoding="utf-8") as handle:
                    metadata = json.load(handle)
                for asset in metadata.get("assets") or []:
                    raw_path = str(asset.get("path") or "")
                    path = (
                        raw_path
                        if os.path.isabs(raw_path)
                        else os.path.abspath(os.path.join(self.root, raw_path))
                    )
                    if not os.path.isfile(path):
                        raise ValueError(f"prepared audio asset is missing: {path}")
                    expected_asset = asset.get("sha256")
                    if expected_asset and file_sha256(path) != expected_asset:
                        raise ValueError(
                            f"prepared audio checksum mismatch: {path}"
                        )
        self._validated = True

    def _resolve_paths(self, record: Dict[str, Any]) -> Dict[str, Any]:
        doc = dict(record)
        for key, value in list(doc.items()):
            if (key == "WavPath" or re.fullmatch(r"WavPath\d+", key)) and value:
                doc[key] = (
                    str(value)
                    if os.path.isabs(str(value))
                    else os.path.abspath(os.path.join(self.root, str(value)))
                )
        input_obj = doc.get("input")
        if isinstance(input_obj, dict):
            input_obj = dict(input_obj)
            for key in ("audio_path", "audio_path_b"):
                if input_obj.get(key) and not os.path.isabs(str(input_obj[key])):
                    input_obj[key] = os.path.abspath(
                        os.path.join(self.root, str(input_obj[key]))
                    )
            if isinstance(input_obj.get("audios"), list):
                audios = []
                for item in input_obj["audios"]:
                    if not isinstance(item, dict):
                        audios.append(item)
                        continue
                    item = dict(item)
                    if item.get("uri") and not os.path.isabs(str(item["uri"])):
                        item["uri"] = os.path.abspath(
                            os.path.join(self.root, str(item["uri"]))
                        )
                    audios.append(item)
                input_obj["audios"] = audios
            if isinstance(input_obj.get("audio"), list):
                audio = []
                for item in input_obj["audio"]:
                    if not isinstance(item, dict):
                        audio.append(item)
                        continue
                    item = dict(item)
                    if item.get("uri") and not os.path.isabs(str(item["uri"])):
                        item["uri"] = os.path.abspath(
                            os.path.join(self.root, str(item["uri"]))
                        )
                    audio.append(item)
                input_obj["audio"] = audio
            doc["input"] = input_obj
        return doc

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(offset=0, limit=limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        self._validate_success()
        rows: List[Dict[str, Any]] = []
        seen = 0
        with open(self.f_name, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                if seen < offset:
                    seen += 1
                    continue
                doc = self._resolve_paths(json.loads(line))
                for source, target in self.col_aliases.items():
                    if source in doc and target not in doc:
                        doc[target] = doc[source]
                if self.check_audio_exists:
                    missing = [
                        path
                        for path in prepared_audio_paths(doc)
                        if not os.path.isfile(path)
                    ]
                    if missing:
                        raise ValueError(
                            f"{self.f_name}:{line_no} references missing audio: {missing}"
                        )
                rows.append(doc)
                if limit and len(rows) >= limit:
                    break
        return rows
