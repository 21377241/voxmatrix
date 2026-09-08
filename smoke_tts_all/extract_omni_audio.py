"""Post-process: Omni speech pred JSON {"text","audio"} -> audio path string."""
from __future__ import annotations

import json
import os
import re
from typing import Any


class ExtractOmniAudioPath:
    def __call__(self, pred: Any, **kwargs: Any) -> str:
        text = pred if isinstance(pred, str) else str(pred)
        text = text.strip()
        if os.path.isfile(text):
            return text
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r'(/[^\s"\']+\.wav)', text)
            if match and os.path.isfile(match.group(1)):
                return match.group(1)
            return text
        if isinstance(obj, dict):
            for key in ("audio", "WavPath", "path", "audio_path"):
                value = obj.get(key)
                if value and os.path.isfile(str(value)):
                    return str(value)
        return text
