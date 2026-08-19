import json
import os
import threading
import typing


class Recorder:
    """Thread/process-safe append-only JSONL event recorder."""

    _locks: typing.Dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, f_name: str, overwrite: bool = True, fsync: bool = False):
        self.name = os.path.abspath(f_name)
        self.fsync = bool(fsync)
        directory = os.path.dirname(self.name) or "."
        os.makedirs(directory, exist_ok=True)
        with self._locks_guard:
            self._lock = self._locks.setdefault(self.name, threading.RLock())
        with self._lock:
            existed = os.path.exists(self.name)
            with open(self.name, "a+", encoding="utf-8") as handle:
                self._flock(handle)
                try:
                    if overwrite:
                        if existed:
                            print(f"File {self.name} already exists, overwriting it.")
                        handle.seek(0)
                        handle.truncate()
                        handle.flush()
                        if self.fsync:
                            os.fsync(handle.fileno())
                finally:
                    self._funlock(handle)

    @staticmethod
    def _flock(handle) -> None:
        try:
            import fcntl
        except ImportError:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    @staticmethod
    def _funlock(handle) -> None:
        try:
            import fcntl
        except ImportError:
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def add(self, data: typing.Dict[str, typing.Any]):
        line = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with open(self.name, "a", encoding="utf-8") as handle:
                self._flock(handle)
                try:
                    handle.write(line)
                    handle.flush()
                    if self.fsync:
                        os.fsync(handle.fileno())
                finally:
                    self._funlock(handle)
