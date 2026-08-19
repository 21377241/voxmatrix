import json
import os
from pathlib import Path
from typing import Any, Dict, List

from audio_evals.dataset.dataset import Dataset
from audio_evals.config import expand_environment
from mesh_eval.core.schema import normalize_record, validate_record


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def _resolve_bundled_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return str(path)
    candidate = PACKAGE_ROOT / path
    return str(candidate) if candidate.exists() else str(path)


def _resolve_audio_path(value: Any, manifest_dir: Path) -> Any:
    if not isinstance(value, (str, os.PathLike)) or not value:
        return value
    path = Path(value)
    if path.is_absolute() or path.exists():
        return str(path)
    sibling = manifest_dir / path
    if sibling.exists():
        return str(sibling)
    bundled = PACKAGE_ROOT / path
    return str(bundled) if bundled.exists() else str(path)


def _resolve_record_audio(record: Dict[str, Any], manifest_dir: Path) -> None:
    for key, value in list(record.items()):
        if key == "WavPath" or (
            isinstance(key, str)
            and key.startswith("WavPath")
            and key[7:].isdigit()
        ):
            record[key] = _resolve_audio_path(value, manifest_dir)
    input_obj = record.get("input")
    if not isinstance(input_obj, dict):
        return
    for key in ("audio_path", "audio_path_b"):
        if key in input_obj:
            input_obj[key] = _resolve_audio_path(input_obj[key], manifest_dir)
    for key in ("audio", "audios"):
        for item in input_obj.get(key) or []:
            if isinstance(item, dict) and "uri" in item:
                item["uri"] = _resolve_audio_path(item["uri"], manifest_dir)


class UnifiedSpeechJsonl(Dataset):
    def __init__(
        self,
        f_name: str,
        default_task: str,
        ref_col: str = "text",
        dataset_id: str = "",
        strict: bool = False,
        check_audio_exists: bool = False,
    ):
        super().__init__(default_task=default_task, ref_col=ref_col)
        self.f_name = _resolve_bundled_path(f_name)
        self.dataset_id = dataset_id
        self.strict = strict
        self.check_audio_exists = check_audio_exists

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(offset=0, limit=limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        records = []
        seen = 0
        with open(self.f_name, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                if seen < offset:
                    seen += 1
                    continue
                records.append(self._normalize(json.loads(line), line_no))
                if limit > 0 and len(records) >= limit:
                    break
        return records

    def _normalize(self, record: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        record = expand_environment(record)
        manifest_dir = Path(self.f_name).resolve().parent
        _resolve_record_audio(record, manifest_dir)
        doc = normalize_record(
            record,
            ref_col=self.ref_col,
            default_task=self.task_name,
            default_dataset_id=self.dataset_id,
        )
        for key, value in list(doc.items()):
            if not (
                key == "WavPath"
                or (key.startswith("WavPath") and key[7:].isdigit())
            ):
                continue
            doc[key] = _resolve_audio_path(value, manifest_dir)
        errors = validate_record(
            doc,
            ref_col=self.ref_col,
            check_audio_exists=self.check_audio_exists,
        )
        if errors:
            doc["schema_errors"] = errors
            if self.strict:
                raise ValueError(
                    f"{self.f_name}:{line_no} failed schema validation: {errors}"
                )
        return doc
