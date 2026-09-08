"""Post-process helpers for speaker diarization JSON outputs."""

from __future__ import annotations

from typing import Any, Optional

from audio_evals.process.base import Process
from mesh_eval.evaluator.speaker import normalize_diarization_prediction


class DiarizationNormalize(Process):
    """Salvage truncated diarization JSON into canonical {"segments":[...]}."""

    def __init__(self, audio_duration_seconds: Optional[float] = None):
        self.audio_duration_seconds = (
            float(audio_duration_seconds) if audio_duration_seconds else None
        )

    def __call__(self, answer: Any) -> str:
        normalized = normalize_diarization_prediction(
            answer, max_time=self.audio_duration_seconds
        )
        return normalized["content"]
