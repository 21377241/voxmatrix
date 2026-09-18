"""Smoke/8p31-closed sample loaders.

These read small JSONL manifests (or utt_clips indexes) that already follow
the evaluation unit smoke settled on. They are not a second annotation
pipeline and they do not rewrite official gold.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from audio_evals.dataset.dataset import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


class NativeClipDataset(Dataset):
    """JSONL of pre-cut clips / pairs with a bound capability protocol."""

    def __init__(
        self,
        f_name: str,
        default_task: str,
        ref_col: str = "text",
        protocol: str = "asr",
        benchmark_id: str = "",
        col_aliases=None,
    ):
        super().__init__(default_task, ref_col, col_aliases)
        raw = Path(f_name).expanduser()
        self.f_name = str(raw if raw.is_absolute() else (REPO_ROOT / raw))
        self.protocol = str(protocol or "asr").strip().lower()
        self.benchmark_id = benchmark_id or Path(self.f_name).stem

    def _iter_rows(self, offset: int = 0, limit: int = 0) -> List[Dict[str, Any]]:
        path = Path(self.f_name)
        if not path.is_file():
            raise FileNotFoundError(f"native clip manifest missing: {path}")
        rows: List[Dict[str, Any]] = []
        skipped = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if skipped < offset:
                    skipped += 1
                    continue
                rows.append(self._normalize(json.loads(line)))
                if limit and len(rows) >= limit:
                    break
        return rows

    def load(self, limit: int = 0) -> List[Dict[str, Any]]:
        return self._iter_rows(limit=limit)

    def load_slice(self, offset: int = 0, limit: int = 0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        return self._iter_rows(offset=offset, limit=limit)

    def _normalize(self, record: Dict[str, Any]) -> Dict[str, Any]:
        row = dict(record)
        row.setdefault("dataset", self.benchmark_id)
        row.setdefault("benchmark_id", self.benchmark_id)
        wav = row.get("WavPath") or row.get("audio_path") or ""
        if wav:
            row["WavPath"] = wav
            input_obj = dict(row.get("input") or {})
            input_obj.setdefault("audio_path", wav)
            row["input"] = input_obj
        if row.get("WavPath2"):
            row["WavPath2"] = row["WavPath2"]

        protocol = self.protocol
        if protocol in {"asr", "long_form_asr"}:
            row["capability"] = protocol
            row.setdefault("task", "speech_recognition_transcription")
        elif protocol == "target_speaker":
            row["capability"] = "target_speaker"
            row.setdefault("task", "speaker_attribution")
            reference = dict(row.get("reference") or {})
            should = row.get("should_execute")
            if should is None:
                should = row.get("text")
            reference["should_execute"] = bool(should)
            row["reference"] = reference
            row["text"] = bool(should)
        elif protocol == "diarization":
            row["capability"] = "diarization"
            row.setdefault("task", "speaker_attribution")
        elif protocol == "speaker_attribution":
            row["capability"] = "speaker_attribution"
            row.setdefault("task", "speaker_attribution")
        elif protocol == "code_switch":
            row["capability"] = "code_switch"
            row.setdefault("task", "audio_speech_understanding")
            row.setdefault("question", row.get("question") or "")
        elif protocol == "acoustic_scene":
            row["capability"] = "acoustic_scene"
            row.setdefault("task", "audio_speech_understanding")
            if not row.get("text") and row.get("answer") is not None:
                row["text"] = row["answer"]
            reference = dict(row.get("reference") or {})
            reference.setdefault("label", row.get("text") or row.get("answer"))
            row["reference"] = reference
        else:
            row.setdefault("capability", protocol)
        return row
