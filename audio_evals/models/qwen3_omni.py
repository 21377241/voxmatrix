import json
import logging
import os
import queue
import select
import threading
import time
from collections import deque
from typing import Dict

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel

logger = logging.getLogger(__name__)

READY_SENTINEL = "__QWEN3_OMNI_READY__"


@isolated(
    "audio_evals/lib/qwen3-omni/main.py",
)
class Qwen3Omni(OfflineModel):
    def __init__(
        self,
        path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        speech: bool = False,
        speaker: str = "Ethan",
        speech_output_dir: str = "",
        max_new_tokens: int = 0,
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Dict = None,
        *args,
        **kwargs,
    ):
        if path == "Qwen/Qwen3-Omni-30B-A3B-Instruct" and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "speaker": speaker,
        }
        if speech:
            self.command_args["speech"] = ""
            if speech_output_dir:
                self.command_args["speech-output-dir"] = speech_output_dir
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            self.command_args["thinker-max-new-tokens"] = int(max_new_tokens)

        self._ready = False
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        self._request_log_interval = float(request_log_interval)
        self._output_queue = None
        self._output_threads = []
        self._recent_process_logs = deque(maxlen=100)

        super().__init__(is_chat=True, sample_params=sample_params)

    def _parse_content(self, content: Dict):
        assert "type" in content
        value = content["value"]
        if content["type"] == "audio" and isinstance(value, dict):
            from audio_evals.audio_segment import materialize_audio_segment

            value = materialize_audio_segment(value)
        return {"type": content["type"], content["type"]: value}

    def _parse_role_content(self, role_content: Dict):
        parsed = dict(role_content)
        contents = parsed.pop("contents", parsed.get("content", ""))
        if isinstance(contents, list):
            parsed["content"] = [self._parse_content(item) for item in contents]
        else:
            parsed["content"] = contents
        return parsed

    def _raise_process_error(self, context: str):
        exit_code = self.process.poll()
        detail = f"{context}; qwen3-omni subprocess exited with code {exit_code}"
        if self._recent_process_logs:
            detail = f"{detail}. recent logs:\n" + "\n".join(
                self._recent_process_logs
            )
        raise RuntimeError(detail)

    def _ensure_output_pumps(self):
        if self._output_queue is not None:
            return

        self._output_queue = queue.Queue()

        def pump(stream_name, stream):
            try:
                for line in iter(stream.readline, ""):
                    self._output_queue.put((stream_name, line))
            finally:
                self._output_queue.put((stream_name, None))

        for stream_name, stream in (
            ("stdout", self.process.stdout),
            ("stderr", self.process.stderr),
        ):
            thread = threading.Thread(
                target=pump,
                args=(stream_name, stream),
                daemon=True,
                name=f"qwen3-omni-{stream_name}",
            )
            thread.start()
            self._output_threads.append(thread)

    def _next_output(self, timeout):
        self._ensure_output_pumps()
        try:
            stream_name, line = self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return None
        stripped = line.strip()
        if stripped:
            self._recent_process_logs.append(f"{stream_name}: {stripped}")
        return stream_name, line

    def _wait_until_ready(self, timeout: float = None):
        if self._ready:
            return

        timeout = self._startup_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + timeout
        while True:
            if self.process.poll() is not None:
                self._raise_process_error("qwen3-omni subprocess exited during startup")

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = "\n".join(self._recent_process_logs)
                self.release()
                detail = f"qwen3-omni subprocess was not ready after {timeout:.1f}s"
                if tail:
                    detail = f"{detail}. recent logs:\n{tail}"
                raise TimeoutError(detail)

            item = self._next_output(min(1.0, remaining))
            if item is None:
                continue
            stream_name, line = item
            line = line.strip()
            if not line:
                continue
            if stream_name == "stdout":
                if line.startswith(READY_SENTINEL):
                    logger.info(line[len(READY_SENTINEL) :].strip())
                    self._ready = True
                    return
                if line.startswith("Error:"):
                    raise RuntimeError(f"qwen3-omni startup failed: {line}")
                logger.info(line)
            else:
                logger.info("qwen3-omni stderr: %s", line)

    def _request(self, conversation):
        import uuid

        self._wait_until_ready()
        uid = str(uuid.uuid4())
        prefix = f"{uid}->"
        deadline = time.monotonic() + self._request_timeout

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self._raise_process_error(
                    "qwen3-omni subprocess exited before request write"
                )
            remaining = deadline - time.monotonic()
            _, wlist, _ = select.select(
                [], [self.process.stdin], [], min(1.0, max(0.0, remaining))
            )
            if wlist:
                try:
                    payload = json.dumps(conversation, ensure_ascii=False)
                    self.process.stdin.write(f"{prefix}{payload}\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._raise_process_error(
                        "qwen3-omni subprocess pipe closed during request write"
                    )
                break
        else:
            self.release()
            raise TimeoutError(
                f"qwen3-omni request could not be written after {self._request_timeout:.1f}s"
            )

        request_start = time.monotonic()
        last_progress_log = request_start
        while True:
            if self.process.poll() is not None:
                self._raise_process_error(
                    "qwen3-omni subprocess exited before producing a response"
                )
            now = time.monotonic()
            if now >= deadline:
                self.release()
                raise TimeoutError(
                    f"qwen3-omni request exceeded {self._request_timeout:.1f}s"
                )
            item = self._next_output(min(1.0, deadline - now))
            now = time.monotonic()
            if item is None and now - last_progress_log >= self._request_log_interval:
                logger.info(
                    "waiting for qwen3-omni response for %.1fs",
                    now - request_start,
                )
                last_progress_log = now
            if item is None:
                continue
            stream_name, result = item
            if stream_name == "stdout":
                if result.startswith(prefix):
                    try:
                        self.process.stdin.write(f"{prefix}close\n")
                        self.process.stdin.flush()
                    except (BrokenPipeError, OSError):
                        self._raise_process_error(
                            "qwen3-omni subprocess pipe closed during response ack"
                        )
                    res = json.loads(result[len(prefix) :])
                    logger.debug("return output: %s", res)
                    res["text"] = res["text"].split("assistant")[-1].strip()
                    return res
                if result.startswith("Error:"):
                    raise RuntimeError("qwen3-omni failed: {}".format(result))
                logger.info(result.strip())
            elif result.strip():
                logger.info("qwen3-omni stderr: %s", result.strip())

    def release(self):
        process = getattr(self, "process", None)
        self._ready = False
        if process is None or process.poll() is not None:
            return
        terminate = getattr(self, "_terminate_isolated_process", None)
        if callable(terminate):
            terminate()
            return
        process.terminate()
        try:
            process.wait(timeout=30)
        except Exception:
            process.kill()
            process.wait(timeout=30)

    def _inference(self, prompt: PromptStruct, **kwargs):
        if isinstance(prompt, dict) and "multi_turn" in prompt:
            conversation = []
            responses = []
            audio_responses = []
            for turn in prompt["multi_turn"]:
                messages = turn if isinstance(turn, list) else [turn]
                conversation.extend(
                    self._parse_role_content(message) for message in messages
                )
                result = self._request(conversation)
                response = result["text"]
                responses.append(response)
                audio_responses.append(result.get("audio"))
                conversation.append({"role": "assistant", "content": response})
            output = {"responses": responses}
            if any(audio_responses):
                # Preserve one entry per turn so speech-to-speech rubrics can
                # reject partial audio generation instead of silently scoring
                # only the final waveform.
                output["audio_responses"] = audio_responses
            return json.dumps(output, ensure_ascii=False)

        conversation = [self._parse_role_content(item) for item in prompt]
        result = self._request(conversation)
        if len(result) == 1:
            return result["text"]
        return json.dumps(result, ensure_ascii=False)
