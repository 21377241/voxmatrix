"""Smoke/8p31-closed sample loaders.

These read small JSONL manifests that already follow the evaluation unit
settled for native coverage. ASR samples come from
``suites/native/build_asr_native_samples.py`` (L0 / timestamp crops), not
utt_clips indexes or FullEval. They do not rewrite official gold.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from audio_evals.dataset.dataset import Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]


def _parse_speaker_count(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and value.is_integer():
        return int(value) if value >= 0 else None
    text = str(value).strip()
    if not text:
        return None
    try:
        count = int(text)
        return count if count >= 0 else None
    except ValueError:
        pass
    match = re.search(r"-?\d+", text)
    if not match:
        return None
    try:
        count = int(match.group(0))
    except ValueError:
        return None
    return count if count >= 0 else None


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
        if row.get("bucket"):
            row.setdefault("window_bucket", row["bucket"])
        input_obj = dict(row.get("input") or {})
        wav = (
            row.get("WavPath")
            or row.get("audio_path")
            or row.get("audio_locator")
            or input_obj.get("audio_path")
            or ""
        )
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
            row["reference_obj"] = reference
            row["text"] = bool(should)
        elif protocol == "diarization":
            row["capability"] = "diarization"
            row.setdefault("task", "speaker_attribution")
        elif protocol == "speaker_attribution":
            row["capability"] = "speaker_attribution"
            row.setdefault("task", "speaker_attribution")
        elif protocol in {"acoustic_scene", "sound_event", "paralinguistic_recognition"}:
            row["capability"] = protocol
            row.setdefault("task", "audio_speech_understanding")
            if not row.get("text") and row.get("answer") is not None:
                answer = row["answer"]
                row["text"] = (
                    ", ".join(str(item) for item in answer)
                    if isinstance(answer, list)
                    else answer
                )
            reference = dict(row.get("reference") or {})
            reference.setdefault("label", row.get("text") or row.get("answer"))
            row["reference_obj"] = reference
            if row.get("question"):
                row.setdefault("question", row["question"])
        elif protocol in {"sound_event_multilabel"}:
            row["capability"] = "sound_event"
            row.setdefault("task", "audio_speech_understanding")
            labels = row.get("labels") or row.get("answer") or []
            if isinstance(labels, str):
                labels = [item.strip() for item in labels.split(",") if item.strip()]
            row["text"] = ", ".join(str(item) for item in labels)
            reference = dict(row.get("reference") or {})
            reference["labels"] = list(labels)
            row["reference_obj"] = reference
        elif protocol in {"qa", "audio_reasoning", "code_switch", "audio_caption", "speech_translation", "speech_translation_qa"}:
            row["capability"] = protocol
            row.setdefault(
                "task",
                "speech_recognition_transcription"
                if protocol == "speech_translation"
                else "audio_speech_understanding",
            )
            answer = row.get("answer")
            if not row.get("text") and answer not in (None, ""):
                row["text"] = (
                    ", ".join(str(item) for item in answer)
                    if isinstance(answer, list)
                    else answer
                )
            input_obj = dict(row.get("input") or {})
            question = row.get("question") or input_obj.get("question") or ""
            if question:
                row["question"] = question
                input_obj.setdefault("question", question)
                row["input"] = input_obj
            reference = dict(row.get("reference") or {})
            if row.get("text") not in (None, ""):
                if protocol == "audio_caption":
                    captions = row.get("captions")
                    if isinstance(captions, list) and captions:
                        reference.setdefault("captions", captions)
                        reference.setdefault("caption", captions[0])
                    else:
                        reference.setdefault("caption", row["text"])
                elif protocol == "speech_translation_qa":
                    reference.setdefault("answer", row["text"])
                else:
                    reference.setdefault("answer", row["text"])
            row["reference_obj"] = reference
        elif protocol == "speaker_verification":
            row["capability"] = "speaker_verification"
            row.setdefault("task", "speaker_attribution")
            label = row.get("label")
            if label is None:
                label = row.get("text")
            same = row.get("same_speaker")
            if same is None and label is not None:
                same = bool(int(label) == 1) if str(label).isdigit() or isinstance(label, int) else bool(label)
            row["text"] = label
            if same is not None:
                row["same_speaker"] = bool(same)
            reference = dict(row.get("reference") or {})
            reference["label"] = label
            if same is not None:
                reference["same_speaker"] = bool(same)
            row["reference_obj"] = reference
        elif protocol in {"spoof_detection", "speaker_count", "speech_generation"}:
            row["capability"] = protocol
            row.setdefault(
                "task",
                "speech_generation_delivery"
                if protocol == "speech_generation"
                else "speaker_attribution"
                if protocol == "spoof_detection"
                else "audio_speech_understanding",
            )
            if not row.get("text") and row.get("answer") is not None:
                row["text"] = row["answer"]
            reference = dict(row.get("reference") or {})
            if row.get("text") not in (None, ""):
                reference.setdefault("label", row["text"])
            if protocol == "speaker_count":
                count_raw = row.get("count")
                if count_raw in (None, ""):
                    count_raw = reference.get("count")
                if count_raw in (None, ""):
                    count_raw = row.get("answer")
                if count_raw in (None, ""):
                    count_raw = row.get("text")
                count = _parse_speaker_count(count_raw)
                if count is not None:
                    reference["count"] = count
                    row["text"] = count
                    row["answer"] = count
                    row["count"] = count
            row["reference_obj"] = reference
        else:
            row.setdefault("capability", protocol)
        if isinstance(row.get("reference"), dict):
            row.setdefault("reference_obj", row.pop("reference"))
        else:
            row.pop("reference", None)
        return row
