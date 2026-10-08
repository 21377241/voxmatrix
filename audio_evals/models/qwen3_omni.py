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
from audio_evals.models.model import Model, OfflineModel

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
        self._init_concurrency(1)

        super().__init__(is_chat=True, sample_params=sample_params)

    def inference(self, prompt: PromptStruct, **kwargs) -> str:
        # OfflineModel.inference serializes on self.lock; concurrent mode is thread-safe.
        if getattr(self, "max_inflight", 1) > 1:
            return Model.inference(self, prompt, **kwargs)
        return super().inference(prompt, **kwargs)

    def _init_concurrency(self, max_inflight: int):
        self.max_inflight = max(1, int(max_inflight))
        self._concurrency_lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: Dict[str, queue.Queue] = {}
        self._dispatcher = None

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

    @staticmethod
    def _finish_response(res: dict) -> dict:
        text = str(res.get("text") or "").strip()
        if text.startswith("assistant"):
            text = text[len("assistant") :].lstrip()
        res["text"] = text
        return res

    def _ensure_dispatcher(self):
        """Concurrent mode: one thread routes ``<uid>->`` replies to their waiters."""
        with self._concurrency_lock:
            if self._dispatcher is not None:
                return
            self._ensure_output_pumps()

            def dispatch():
                while True:
                    item = self._next_output(1.0)
                    if item is None:
                        if self.process.poll() is not None:
                            break
                        continue
                    stream_name, line = item
                    line = line.strip()
                    if not line:
                        continue
                    anchor = line.find("->")
                    key = line[: anchor + 2] if anchor != -1 else ""
                    with self._concurrency_lock:
                        waiter = self._pending.get(key) if stream_name == "stdout" else None
                    if waiter is not None:
                        waiter.put(line[len(key) :])
                    elif stream_name == "stdout" and line.startswith("Error:"):
                        logger.warning("qwen3-omni unattributed error: %s", line)
                    elif stream_name == "stdout":
                        logger.info(line)
                    else:
                        logger.info("qwen3-omni stderr: %s", line)
                with self._concurrency_lock:
                    waiters = list(self._pending.values())
                for waiter in waiters:
                    waiter.put(None)

            self._dispatcher = threading.Thread(
                target=dispatch, daemon=True, name="qwen3-omni-dispatch"
            )
            self._dispatcher.start()

    def _request_concurrent(self, conversation):
        import uuid

        with self._concurrency_lock:
            self._wait_until_ready()
        self._ensure_dispatcher()
        prefix = f"{uuid.uuid4()}->"
        replies: queue.Queue = queue.Queue()
        with self._concurrency_lock:
            self._pending[prefix] = replies
        try:
            if self.process.poll() is not None:
                self._raise_process_error(
                    "qwen3-omni subprocess exited before request write"
                )
            payload = json.dumps(conversation, ensure_ascii=False)
            with self._write_lock:
                try:
                    self.process.stdin.write(f"{prefix}{payload}\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._raise_process_error(
                        "qwen3-omni subprocess pipe closed during request write"
                    )
            request_start = time.monotonic()
            deadline = request_start + self._request_timeout
            last_progress_log = request_start
            while True:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError(
                        f"qwen3-omni request exceeded {self._request_timeout:.1f}s"
                    )
                try:
                    body = replies.get(timeout=min(1.0, deadline - now))
                except queue.Empty:
                    now = time.monotonic()
                    if now - last_progress_log >= self._request_log_interval:
                        logger.info(
                            "waiting for qwen3-omni response for %.1fs",
                            now - request_start,
                        )
                        last_progress_log = now
                    continue
                if body is None:
                    self._raise_process_error(
                        "qwen3-omni subprocess exited before producing a response"
                    )
                res = json.loads(body)
                if "error" in res:
                    raise RuntimeError("qwen3-omni failed: {}".format(res["error"]))
                return self._finish_response(res)
        finally:
            with self._concurrency_lock:
                self._pending.pop(prefix, None)

    def _request(self, conversation):
        import uuid

        if getattr(self, "max_inflight", 1) > 1:
            return self._request_concurrent(conversation)
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
                    return self._finish_response(res)
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
        force_close = int(kwargs.get("force_close_thinking_tokens") or 0)
        if force_close > 0:
            # Only the vLLM-Omni worker understands this envelope.
            result = self._request(
                {"conversation": conversation, "force_close_thinking_tokens": force_close}
            )
        else:
            result = self._request(conversation)
        if len(result) == 1:
            return result["text"]
        return json.dumps(result, ensure_ascii=False)
