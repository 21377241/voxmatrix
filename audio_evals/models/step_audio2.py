import json
import os
import re
from copy import deepcopy
from typing import Dict, List

from audio_evals.audio_segment import materialize_audio_segment
from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.json_line_subprocess import JsonLineSubprocessMixin
from audio_evals.models.model import OfflineModel


READY_SENTINEL = "__STEP_AUDIO2_READY__"
# Step sometimes prefixes transcripts with a language tag such as "<英语>".
LANGUAGE_TAG_PREFIX = re.compile(r"^(?:<[^<>\s]{1,10}>\s*)+")


@isolated("audio_evals/lib/StepAudio2/main.py")
class StepAudio2(JsonLineSubprocessMixin, OfflineModel):
    """Text or speech-output adapter for the official Step-Audio-2 implementation."""

    def __init__(
        self,
        path: str = "stepfun-ai/Step-Audio-2-mini",
        speech: bool = False,
        max_new_tokens: int = 512,
        prompt_wav: str = "",
        speech_output_dir: str = "",
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "stepfun-ai/Step-Audio-2-mini" and not os.path.exists(path):
            path = self._download_model(path)
        if int(max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be positive")
        self.speech = bool(speech)
        self.command_args = {
            "path": path,
            "max-new-tokens": int(max_new_tokens),
        }
        if self.speech:
            self.command_args["speech"] = ""
            if prompt_wav:
                self.command_args["prompt-wav"] = prompt_wav
            if speech_output_dir:
                self.command_args["speech-output-dir"] = speech_output_dir
        self._init_json_line_client(
            model_label="step-audio-2",
            ready_sentinel=READY_SENTINEL,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            request_log_interval=request_log_interval,
        )
        super().__init__(is_chat=True, sample_params=sample_params)

    @staticmethod
    def _parts(contents) -> List[Dict]:
        if not isinstance(contents, list):
            return [{"type": "text", "text": str(contents or "")}]
        parts = []
        for item in contents:
            if not isinstance(item, dict) or "type" not in item:
                raise ValueError(f"invalid Step-Audio-2 content item: {item!r}")
            value = item.get("value")
            if item["type"] == "audio":
                if isinstance(value, dict):
                    value = materialize_audio_segment(value)
                if not value:
                    raise ValueError("Step-Audio-2 audio content has an empty path")
                parts.append({"type": "audio", "audio": str(value)})
            elif item["type"] == "text":
                parts.append({"type": "text", "text": str(value or "")})
            else:
                raise ValueError(f"Step-Audio-2 does not support {item['type']!r}")
        return parts

    def _prepare_request(self, prompt, generate_kwargs) -> Dict:
        if not isinstance(prompt, list):
            raise TypeError("Step-Audio-2 expects a list of chat messages")
        messages = []
        for message in deepcopy(prompt):
            contents = message.get("contents", message.get("content", ""))
            messages.append(
                {"role": str(message.get("role") or "user"), "content": self._parts(contents)}
            )
        if self.speech:
            messages.append({"role": "assistant", "content": "<tts_start>", "eot": False})
        else:
            messages.append({"role": "assistant", "content": None})
        return {"messages": messages, "generate_kwargs": generate_kwargs}

    @staticmethod
    def _text(response) -> str:
        return LANGUAGE_TAG_PREFIX.sub("", str(response.get("text") or "").strip())

    def _inference(self, prompt: PromptStruct, **kwargs):
        if isinstance(prompt, dict) and "multi_turn" in prompt:
            conversation = []
            responses = []
            audio_responses = []
            for turn in prompt["multi_turn"]:
                conversation.extend(deepcopy(turn if isinstance(turn, list) else [turn]))
                response = self._request(self._prepare_request(conversation, kwargs))
                text = self._text(response)
                responses.append(text)
                audio_responses.append(response.get("audio"))
                conversation.append(
                    {"role": "assistant", "contents": [{"type": "text", "value": text}]}
                )
            output = {"responses": responses}
            if self.speech:
                # One entry per turn: speech-to-speech rubrics reject partial audio.
                output["audio_responses"] = audio_responses
            return json.dumps(output, ensure_ascii=False)
        response = self._request(self._prepare_request(prompt, kwargs))
        if self.speech:
            return json.dumps(
                {"text": self._text(response), "audio": response.get("audio", "")},
                ensure_ascii=False,
            )
        return self._text(response)
