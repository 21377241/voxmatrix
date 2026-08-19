import os
from copy import deepcopy
from typing import Dict, List

from audio_evals.audio_segment import materialize_audio_segment
from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.json_line_subprocess import JsonLineSubprocessMixin
from audio_evals.models.model import OfflineModel


READY_SENTINEL = "__FUN_AUDIO_CHAT_READY__"
AUDIO_TEMPLATE = "<|audio_bos|><|AUDIO|><|audio_eos|>"
DEFAULT_SYSTEM_PROMPT = "You are asked to generate text tokens."


@isolated("audio_evals/lib/fun-audio-chat/main.py")
class FunAudioChat(JsonLineSubprocessMixin, OfflineModel):
    """Text-output adapter for the official Fun-Audio-Chat implementation."""

    def __init__(
        self,
        path: str = "FunAudioLLM/Fun-Audio-Chat-8B",
        max_new_tokens: int = 256,
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "FunAudioLLM/Fun-Audio-Chat-8B" and not os.path.exists(path):
            path = self._download_model(path)
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be positive")
        self.command_args = {
            "path": path,
            "max-new-tokens": int(max_new_tokens),
        }
        self._init_json_line_client(
            model_label="fun-audio-chat",
            ready_sentinel=READY_SENTINEL,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            request_log_interval=request_log_interval,
        )
        super().__init__(is_chat=True, sample_params=sample_params)

    @staticmethod
    def _content_parts(contents, audio_paths: List[str]) -> List[str]:
        if not isinstance(contents, list):
            return [str(contents or "")]
        parts = []
        for item in contents:
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError(f"invalid Fun-Audio-Chat content item: {item!r}")
            content_type = item["type"]
            value = item.get("value")
            if content_type == "audio":
                if isinstance(value, dict):
                    value = materialize_audio_segment(value)
                audio_path = str(value or "")
                if not audio_path:
                    raise ValueError("Fun-Audio-Chat audio content has an empty path")
                audio_paths.append(audio_path)
                parts.append(AUDIO_TEMPLATE)
            elif content_type == "text":
                text = str(value or "")
                if text:
                    parts.append(text)
            else:
                raise ValueError(
                    f"Fun-Audio-Chat text evaluation does not support {content_type!r}"
                )
        return parts

    def _prepare_request(self, prompt) -> Dict:
        messages = deepcopy(prompt)
        if not isinstance(messages, list):
            raise TypeError("Fun-Audio-Chat expects a list of chat messages")
        audio_paths: List[str] = []
        conversation = []
        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("Fun-Audio-Chat message must be an object")
            contents = message.get("contents", message.get("content", ""))
            parts = self._content_parts(contents, audio_paths)
            conversation.append(
                {"role": str(message.get("role") or "user"), "content": "\n".join(parts)}
            )
        if conversation and conversation[0]["role"] == "system":
            existing = conversation[0]["content"].strip()
            if not existing.startswith(DEFAULT_SYSTEM_PROMPT):
                conversation[0]["content"] = (
                    f"{DEFAULT_SYSTEM_PROMPT}\n\n{existing}" if existing else DEFAULT_SYSTEM_PROMPT
                )
        else:
            conversation.insert(
                0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
            )
        return {"conversation": conversation, "audio_paths": audio_paths}

    def _inference(self, prompt: PromptStruct, **kwargs):
        if isinstance(prompt, dict) and "multi_turn" in prompt:
            conversation = []
            responses = []
            for turn in prompt["multi_turn"]:
                messages = turn if isinstance(turn, list) else [turn]
                conversation.extend(deepcopy(messages))
                response = self._request(self._prepare_request(conversation))
                text = str(response.get("text") or "").strip()
                responses.append(text)
                conversation.append(
                    {
                        "role": "assistant",
                        "contents": [{"type": "text", "value": text}],
                    }
                )
            import json

            return json.dumps({"responses": responses}, ensure_ascii=False)
        response = self._request(self._prepare_request(prompt))
        return str(response.get("text") or "").strip()
