import json
import os
import logging
import select
from copy import deepcopy
from typing import Dict
from audio_evals.audio_segment import materialize_audio_segment
from audio_evals.base import PromptStruct
from audio_evals.models.json_line_subprocess import JsonLineSubprocessMixin
from audio_evals.models.model import OfflineModel
from audio_evals.isolate import isolated


logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/minicpm/main.py")
class MiniCPMo(OfflineModel):
    def __init__(
        self,
        path: str,
        speech: bool = False,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "openbmb/MiniCPM-o-2_6" and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
        }
        if speech:
            self.command_args["speech"] = ""
        super().__init__(is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs) -> str:
        import uuid

        uid = str(uuid.uuid4())
        prefix = f"{uid}->"

        input_o = {"prompt": prompt}
        input_o.update(kwargs)

        while True:
            _, wlist, _ = select.select([], [self.process.stdin], [], 60)
            if wlist:
                self.process.stdin.write(f"{prefix}{json.dumps(input_o)}\n")
                self.process.stdin.flush()
                print("already write in")
                break

        while True:
            reads, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1.0
            )
            for read in reads:
                if read is self.process.stdout:
                    result = self.process.stdout.readline()
                    if result:
                        if result.startswith(prefix):
                            self.process.stdin.write("{}close\n".format(prefix))
                            self.process.stdin.flush()
                            res = json.loads(result[len(prefix) :])
                            if len(res) == 1:
                                return res["text"]
                            return json.dumps(res, ensure_ascii=False)
                        elif result.startswith("Error:"):
                            raise RuntimeError(
                                "mimicpm-o 2.6 failed: {}".format(result)
                            )
                        else:
                            logger.info(result)
                if read is self.process.stderr:
                    error_output = self.process.stderr.readline()
                    if error_output:
                        print(f"stderr: {error_output.strip()}")


@isolated("audio_evals/lib/minicpm/main_o4_5.py")
class MiniCPMo45(JsonLineSubprocessMixin, OfflineModel):
    """MiniCPM-o 4.5 half-duplex chat: audio/text in, text or text+speech out."""

    def __init__(
        self,
        path: str = "openbmb/MiniCPM-o-4_5",
        speech: bool = False,
        max_new_tokens: int = 512,
        ref_audio: str = "",
        speech_output_dir: str = "",
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "openbmb/MiniCPM-o-4_5" and not os.path.exists(path):
            path = self._download_model(path)
        self.speech = bool(speech)
        self.command_args = {"path": path, "max-new-tokens": int(max_new_tokens)}
        if self.speech:
            self.command_args["speech"] = ""
            if ref_audio:
                self.command_args["ref-audio"] = ref_audio
            if speech_output_dir:
                self.command_args["speech-output-dir"] = speech_output_dir
        self._init_json_line_client(
            model_label="minicpm-o-4.5",
            ready_sentinel="__MINICPMO45_READY__",
            startup_timeout=startup_timeout,
            request_timeout=request_timeout,
            request_log_interval=request_log_interval,
        )
        super().__init__(is_chat=True, sample_params=sample_params)

    @staticmethod
    def _normalize(prompt):
        messages = []
        for message in deepcopy(prompt):
            contents = message.get("contents", message.get("content", ""))
            if not isinstance(contents, list):
                contents = [{"type": "text", "value": str(contents or "")}]
            for item in contents:
                if item.get("type") == "audio" and isinstance(item.get("value"), dict):
                    item["value"] = materialize_audio_segment(item["value"])
            messages.append({"role": message.get("role") or "user", "contents": contents})
        return messages

    def _inference(self, prompt: PromptStruct, **kwargs):
        if isinstance(prompt, dict) and "multi_turn" in prompt:
            conversation = []
            responses = []
            audio_responses = []
            for turn in prompt["multi_turn"]:
                conversation.extend(self._normalize(turn if isinstance(turn, list) else [turn]))
                response = self._request(
                    {"prompt": conversation, "generate_kwargs": kwargs}
                )
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
        response = self._request(
            {"prompt": self._normalize(prompt), "generate_kwargs": kwargs}
        )
        if self.speech:
            return json.dumps(
                {"text": response.get("text", ""), "audio": response.get("audio", "")},
                ensure_ascii=False,
            )
        return str(response.get("text") or "").strip()
