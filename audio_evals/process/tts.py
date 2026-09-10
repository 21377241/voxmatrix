"""Post-process helpers for speech-generation / TTS model outputs."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from audio_evals.process.base import Process

logger = logging.getLogger(__name__)

_WAV_PATH_RE = re.compile(r'(/[^\s"\']+\.(?:wav|flac|mp3|ogg))', re.IGNORECASE)


class ExtractOmniAudioPath(Process):
    """Extract a filesystem audio path from Omni-style JSON or bare path strings.

    Accepts:
      - bare path: ``/path/to/file.wav``
      - JSON: ``{"text": "...", "audio": "/path/to/file.wav"}``
      - noisy text containing an absolute ``.wav`` path
    """

    def __call__(self, answer: Any) -> str:
        if isinstance(answer, dict):
            for key in ("audio", "WavPath", "path", "audio_path"):
                value = answer.get(key)
                if value and os.path.isfile(str(value)):
                    return str(value)
            text = json.dumps(answer, ensure_ascii=False)
        else:
            text = str(answer).strip()

        if os.path.isfile(text):
            return text

        try:
            obj = json.loads(text)
        except Exception:
            obj = None

        if isinstance(obj, dict):
            for key in ("audio", "WavPath", "path", "audio_path"):
                value = obj.get(key)
                if value and os.path.isfile(str(value)):
                    return str(value)
            # Key present but file missing — still return path for clearer assert later.
            for key in ("audio", "WavPath", "path", "audio_path"):
                value = obj.get(key)
                if value:
                    logger.warning("audio path from JSON not found on disk: %s", value)
                    return str(value)

        match = _WAV_PATH_RE.search(text)
        if match and os.path.isfile(match.group(1)):
            return match.group(1)

        return text
