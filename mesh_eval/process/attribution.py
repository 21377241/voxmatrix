"""Post-process helpers for speaker-attribution JSON outputs."""

from __future__ import annotations

from typing import Any

from audio_evals.process.base import Process
from mesh_eval.evaluator.speaker import normalize_speaker_attribution_prediction


class SpeakerAttributionNormalize(Process):
    """Salvage truncated / illegal attribution JSON into {"utterances":[...]}."""

    def __call__(self, answer: Any) -> str:
        return normalize_speaker_attribution_prediction(answer)["content"]
