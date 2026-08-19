import json
import logging
import queue
import select
import threading
import time
from collections import deque


class JsonLineSubprocessMixin:
    """Strict request/response protocol for long-lived isolated model workers."""

    def _init_json_line_client(
        self,
        *,
        model_label,
        ready_sentinel,
        startup_timeout,
        request_timeout,
        request_log_interval,
    ):
        self._model_label = str(model_label)
        self._ready_sentinel = str(ready_sentinel)
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        self._request_log_interval = float(request_log_interval)
        self._ready = False
        self._output_queue = None
        self._output_threads = []
        self._recent_process_logs = deque(maxlen=100)
        self._protocol_logger = logging.getLogger(type(self).__module__)

    def _raise_process_error(self, context):
        exit_code = self.process.poll()
        detail = (
            f"{context}; {self._model_label} subprocess exited with code {exit_code}"
        )
        if self._recent_process_logs:
            detail += ". recent logs:\n" + "\n".join(self._recent_process_logs)
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
                name=f"{self._model_label}-{stream_name}",
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

    def _wait_until_ready(self, timeout=None):
        if self._ready:
            return
        timeout = self._startup_timeout if timeout is None else float(timeout)
        deadline = time.monotonic() + timeout
        while True:
            if self.process.poll() is not None:
                self._raise_process_error(
                    f"{self._model_label} subprocess exited during startup"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                tail = "\n".join(self._recent_process_logs)
                self.release()
                detail = (
                    f"{self._model_label} subprocess was not ready after {timeout:.1f}s"
                )
                if tail:
                    detail += f". recent logs:\n{tail}"
                raise TimeoutError(detail)
            item = self._next_output(min(1.0, remaining))
            if item is None:
                continue
            stream_name, line = item
            stripped = line.strip()
            if not stripped:
                continue
            if stream_name == "stdout":
                if stripped.startswith(self._ready_sentinel):
                    self._protocol_logger.info(
                        stripped[len(self._ready_sentinel) :].strip()
                    )
                    self._ready = True
                    return
                if stripped.startswith("Error:"):
                    raise RuntimeError(
                        f"{self._model_label} startup failed: {stripped}"
                    )
                self._protocol_logger.info(stripped)
            else:
                self._protocol_logger.info(
                    "%s stderr: %s", self._model_label, stripped
                )

    def _request(self, payload):
        import uuid

        self._wait_until_ready()
        prefix = f"{uuid.uuid4()}->"
        deadline = time.monotonic() + self._request_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self._raise_process_error(
                    f"{self._model_label} subprocess exited before request write"
                )
            remaining = deadline - time.monotonic()
            _, writable, _ = select.select(
                [], [self.process.stdin], [], min(1.0, max(0.0, remaining))
            )
            if writable:
                try:
                    encoded = json.dumps(payload, ensure_ascii=False)
                    self.process.stdin.write(f"{prefix}{encoded}\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._raise_process_error(
                        f"{self._model_label} pipe closed during request write"
                    )
                break
        else:
            self.release()
            raise TimeoutError(
                f"{self._model_label} request could not be written after "
                f"{self._request_timeout:.1f}s"
            )

        started = time.monotonic()
        last_progress_log = started
        while True:
            if self.process.poll() is not None:
                self._raise_process_error(
                    f"{self._model_label} subprocess exited before a response"
                )
            now = time.monotonic()
            if now >= deadline:
                self.release()
                raise TimeoutError(
                    f"{self._model_label} request exceeded "
                    f"{self._request_timeout:.1f}s"
                )
            item = self._next_output(min(1.0, deadline - now))
            now = time.monotonic()
            if item is None:
                if now - last_progress_log >= self._request_log_interval:
                    self._protocol_logger.info(
                        "waiting for %s response for %.1fs",
                        self._model_label,
                        now - started,
                    )
                    last_progress_log = now
                continue
            stream_name, line = item
            if stream_name == "stderr":
                if line.strip():
                    self._protocol_logger.info(
                        "%s stderr: %s", self._model_label, line.strip()
                    )
                continue
            if line.startswith(prefix):
                try:
                    response = json.loads(line[len(prefix) :])
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{self._model_label} returned invalid protocol JSON: {line!r}"
                    ) from exc
                try:
                    self.process.stdin.write(f"{prefix}close\n")
                    self.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    self._raise_process_error(
                        f"{self._model_label} pipe closed during response ack"
                    )
                if not isinstance(response, dict):
                    raise RuntimeError(
                        f"{self._model_label} response must be an object"
                    )
                if response.get("error"):
                    raise RuntimeError(
                        f"{self._model_label} inference failed: {response['error']}"
                    )
                return response
            if line.startswith("Error:"):
                raise RuntimeError(f"{self._model_label} failed: {line.strip()}")
            if line.strip():
                self._protocol_logger.info(line.strip())

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
