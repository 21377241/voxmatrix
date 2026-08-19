import json
import os
from copy import deepcopy
from typing import Dict

from audio_evals.audio_segment import materialize_audio_segment
from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.json_line_subprocess import JsonLineSubprocessMixin
from audio_evals.models.model import OfflineModel


READY_SENTINEL = "__QWEN2_5_OMNI_READY__"
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs and generating text."
)


@isolated("audio_evals/lib/qwen2-5omni/main.py")
class QwenOmni(JsonLineSubprocessMixin, OfflineModel):
    def __init__(
        self,
        path: str = "Qwen/Qwen2.5-Omni-7B",
        speech: bool = False,
        speaker: str = "Chelsie",
        speech_output_dir: str = "",
        max_new_tokens: int = 0,
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "Qwen/Qwen2.5-Omni-7B" and not os.path.exists(path):
            path = self._download_model(path)
        self.command_args = {"path": path, "speaker": speaker}
        if speech:
            self.command_args["speech"] = ""
            if speech_output_dir:
                self.command_args["speech-output-dir"] = speech_output_dir
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            self.command_args["thinker-max-new-tokens"] = int(max_new_tokens)
        self._init_json_line_client(
            model_label="qwen2.5-omni",
            ready_sentinel=READY_SENTINEL,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            request_log_interval=request_log_interval,
        )
        super().__init__(is_chat=True, sample_params=sample_params)

    @staticmethod
    def _parse_content(content: Dict):
        if "type" not in content:
            raise ValueError("Qwen2.5-Omni content is missing type")
        content_type = content["type"]
        value = content.get("value")
        if content_type == "audio" and isinstance(value, dict):
            value = materialize_audio_segment(value)
        return {"type": content_type, content_type: value}

    def _parse_role_content(self, role_content: Dict):
        parsed = deepcopy(role_content)
        contents = parsed.pop("contents", parsed.get("content", ""))
        if isinstance(contents, list):
            parsed["content"] = [self._parse_content(item) for item in contents]
        else:
            parsed["content"] = contents
        return parsed

    @staticmethod
    def _response_text(response):
        text = str(response.get("text") or "")
        return text.strip()

    def _single_inference(self, conversation):
        if not conversation or conversation[0].get("role") != "system":
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
                },
                *conversation,
            ]
        response = self._request(conversation)
        response["text"] = self._response_text(response)
        if len(response) == 1:
            return response["text"]
        return json.dumps(response, ensure_ascii=False)

    def _inference(self, prompt: PromptStruct, **kwargs):
        if isinstance(prompt, dict) and "multi_turn" in prompt:
            conversation = []
            responses = []
            for turn in prompt["multi_turn"]:
                messages = turn if isinstance(turn, list) else [turn]
                conversation.extend(
                    self._parse_role_content(message) for message in messages
                )
                response = self._response_text(self._request(conversation))
                responses.append(response)
                conversation.append({"role": "assistant", "content": response})
            return json.dumps({"responses": responses}, ensure_ascii=False)

        conversation = [self._parse_role_content(item) for item in prompt]
        return self._single_inference(conversation)
