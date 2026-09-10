import json
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
        multi_turn = input_obj.get("multi_turn")
        if isinstance(multi_turn, list) and multi_turn:
            # Qwen3-Omni adapters consume this explicit structure one turn at
            # a time and append the generated assistant response between
            # turns.  Returning it as-is avoids flattening a conversation into
            # an order-ambiguous list.
            return {"multi_turn": deepcopy(multi_turn)}
        messages = input_obj.get("messages")
        if isinstance(messages, list) and messages:
            return deepcopy(messages)
        return super().load(**kwargs)


class ToolSchemaPrompt(ContextualAudioPrompt):
    """Context prompt that carries the complete native tool schemas."""

    def load(self, **kwargs: Any) -> Any:
        rendered = super().load(**kwargs)
        schemas = (kwargs.get("input") or {}).get("tool_schemas")
        if not schemas:
            return rendered
        instruction = (
            "Return JSON only as {\"tool\": string|null, "
            "\"arguments\": object}. Do not claim execution; select only a "
            "tool present in these schemas.\nComplete tool schemas: "
            + json.dumps(schemas, ensure_ascii=False, default=str)
        )
        if isinstance(rendered, dict) and isinstance(rendered.get("multi_turn"), list):
            turns = rendered["multi_turn"]
            if turns and isinstance(turns[0], list):
                turns[0] = deepcopy(turns[0])
                turns[0].insert(0, {"role": "system", "contents": [{"type": "text", "value": instruction}]})
            return rendered
        if isinstance(rendered, list):
            result = deepcopy(rendered)
            system = next((item for item in result if item.get("role") == "system"), None)
            if system is None:
                result.insert(0, {"role": "system", "contents": [{"type": "text", "value": instruction}]})
            else:
                system.setdefault("contents", []).append({"type": "text", "value": instruction})
            return result
        return rendered
