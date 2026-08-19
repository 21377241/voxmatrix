from copy import deepcopy
from typing import Any

from audio_evals.prompt.base import Prompt


class AudioSegmentPrompt(Prompt):
    def load(self, **kwargs: Any) -> Any:
        prompt = deepcopy(super().load(**kwargs))
        input_obj = kwargs.get("input") or {}
        start = input_obj.get("start_time")
        end = input_obj.get("end_time")
        if start is None or end is None:
            return prompt
        segment = {
            "path": kwargs.get("WavPath") or input_obj.get("audio_path"),
            "start_time": start,
            "end_time": end,
        }
        for message in prompt if isinstance(prompt, list) else []:
            for content in message.get("contents") or []:
                if content.get("type") == "audio":
                    content["value"] = segment
                    return prompt
        return prompt


class ContextualAudioPrompt(Prompt):
    """Use benchmark-provided dialogue history when it is available."""

    def load(self, **kwargs: Any) -> Any:
        input_obj = kwargs.get("input") or {}
        messages = input_obj.get("messages")
        if isinstance(messages, list) and messages:
            return deepcopy(messages)
        return super().load(**kwargs)
