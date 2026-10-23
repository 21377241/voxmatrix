from copy import deepcopy
import json
import logging
import os
from typing import Dict, List

from audio_evals.audio_segment import materialize_audio_segment
from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.json_line_subprocess import JsonLineSubprocessMixin
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

READY_SENTINEL = "__KIMI_AUDIO_READY__"


@isolated("audio_evals/lib/Kimi-Audio/main.py")
class KimiAudioModel(JsonLineSubprocessMixin, OfflineModel):
    def __init__(
        self,
        model_path: str = "moonshotai/Kimi-Audio-7B-Instruct",
        speech: bool = False,
        audio_tokenizer_path: str = "",
        speech_output_dir: str = "",
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if model_path == "moonshotai/Kimi-Audio-7B-Instruct" and not os.path.exists(
            model_path
        ):
            model_path = self._download_model(model_path)

        self.speech = bool(speech)
        self.command_args = {
            "model_path": model_path,
        }
        if audio_tokenizer_path:
            self.command_args["audio_tokenizer_path"] = audio_tokenizer_path
        if self.speech:
            self.command_args["speech"] = ""
            if speech_output_dir:
                self.command_args["speech_output_dir"] = speech_output_dir
        self._init_json_line_client(
            model_label="kimi-audio",
            ready_sentinel=READY_SENTINEL,
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            request_log_interval=request_log_interval,
        )
        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_role_content(self, role_content: Dict) -> List[Dict]:
        contents = role_content.get("contents", role_content.get("content"))
        if not isinstance(contents, list):
            raise TypeError("prompt should be list not string")

        res = []
        for c in contents:
            value = c.get("value")
            if c.get("type") == "audio" and isinstance(value, dict):
                value = materialize_audio_segment(value)
            res.append(
                {
                    "role": role_content["role"],
                    "message_type": c["type"],
                    "content": value,
                }
            )
        return res

    def _messages(self, prompt) -> List[Dict]:
        # Kimi's prompt manager only accepts user/assistant; fold system text into
        # the first user turn so native slots with system prompts still run.
        messages: List[Dict] = []
        system_texts: List[str] = []
        for item in prompt:
            role = str(item.get("role") or "user")
            if role == "system":
                for part in item.get("contents") or []:
                    if part.get("type") == "text" and part.get("value"):
                        system_texts.append(str(part["value"]))
                continue
            parsed = self._parse_role_content(item)
            if system_texts and role == "user":
                messages.append(
                    {
                        "role": "user",
                        "message_type": "text",
                        "content": "\n".join(system_texts),
                    }
                )
                system_texts = []
            messages.extend(parsed)
        if system_texts:
            messages.insert(
                0,
                {
                    "role": "user",
                    "message_type": "text",
                    "content": "\n".join(system_texts),
                },
            )
        return messages

    def _inference(self, prompt: PromptStruct, **kwargs):
        if isinstance(prompt, dict) and "multi_turn" in prompt:
            conversation = []
            responses = []
            audio_responses = []
            for turn in prompt["multi_turn"]:
                conversation.extend(deepcopy(turn if isinstance(turn, list) else [turn]))
                response = self._request({"messages": self._messages(conversation)})
                text = str(response.get("text") or "").strip()
                responses.append(text)
                audio_responses.append(response.get("audio"))
                conversation.append(
                    {"role": "assistant", "contents": [{"type": "text", "value": text}]}
                )
            output = {"responses": responses}
            if self.speech:
                output["audio_responses"] = audio_responses
            return json.dumps(output, ensure_ascii=False)

        response = self._request({"messages": self._messages(prompt)})
        if self.speech:
            return json.dumps(
                {
                    "text": str(response.get("text") or "").strip(),
                    "audio": response.get("audio", ""),
                },
                ensure_ascii=False,
            )
        return str(response.get("text") or "").strip()
