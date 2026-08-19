"""GPU-isolated model replicas and sample-level data parallel scheduling."""

import importlib
import inspect
import logging
import os
import queue
import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from mesh_eval.core.measurement import consume_reported_runtime_metrics

logger = logging.getLogger(__name__)

GpuToken = Union[int, str]
_GPU_TOKEN_PATTERN = re.compile(r"^(?:\d+|GPU-[A-Za-z0-9._:/-]+|MIG-[A-Za-z0-9._:/-]+)$")
_NO_DEVICE_TOKENS = {"", "-1", "none", "void", "nodevfiles"}


def model_supports_pool(cls_path: str) -> bool:
    """Return whether a model constructor accepts the isolated ``gpu_id`` kwarg."""
    module_name, qualname = cls_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), qualname)
    try:
        # @isolated uses functools.wraps.  Avoid unwrapping because gpu_id is
        # injected by the wrapper rather than the original constructor.
        signature = inspect.signature(cls.__init__, follow_wrapped=False)
    except (TypeError, ValueError):
        return False
    return "gpu_id" in signature.parameters


def _normalize_gpu_tokens(values: Sequence[GpuToken]) -> List[str]:
    tokens: List[str] = []
    for value in values:
        token = str(value).strip()
        if token.lower() in _NO_DEVICE_TOKENS:
            return []
        if not _GPU_TOKEN_PATTERN.fullmatch(token):
            raise ValueError(
                "unsupported GPU token {!r}; expected an integer index, GPU UUID, "
                "or MIG UUID".format(token)
            )
        if token in tokens:
            raise ValueError(f"duplicate GPU token: {token}")
        tokens.append(token)
    return tokens


def get_available_gpu_ids() -> List[str]:
    """Discover visible GPU tokens without escaping scheduler restrictions.

    An explicitly present ``CUDA_VISIBLE_DEVICES`` is authoritative, including
    the empty/-1 forms that intentionally hide every GPU.  UUID and MIG tokens
    are preserved verbatim for the isolated subprocess.  Host-wide discovery is
    attempted only when the variable is absent.
    """
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        raw_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        raw_tokens = [item.strip() for item in raw_value.split(",")]
        tokens = _normalize_gpu_tokens(raw_tokens)
        logger.info("Using CUDA_VISIBLE_DEVICES GPU tokens: %s", tokens)
        return tokens

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Failed to detect GPUs via nvidia-smi: %s", exc)
        return []
    if result.returncode != 0:
        logger.warning(
            "nvidia-smi GPU discovery failed with exit code %s: %s",
            result.returncode,
            result.stderr.strip(),
        )
        return []
    tokens = _normalize_gpu_tokens(
        [line.strip() for line in result.stdout.splitlines() if line.strip()]
    )
    logger.info("Detected GPU tokens via nvidia-smi: %s", tokens)
    return tokens


def _terminate_model(model: Any) -> None:
    """Release a model and, as a safety net, its complete process group."""
    try:
        release = getattr(model, "release", None)
        unload = getattr(model, "unload", None)
        if callable(release):
            release()
        elif callable(unload):
            unload()
    except Exception as exc:
        logger.warning("Model release hook failed: %s", exc)

    process = getattr(model, "process", None)
    if process is None or process.poll() is not None:
        return
    terminate_group = getattr(model, "_terminate_isolated_process", None)
    try:
        if callable(terminate_group):
            terminate_group()
            return
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    except Exception as exc:
        logger.warning("Isolated model process cleanup failed: %s", exc)


class IsolatedModelPool:
    """Own GPU-bound model replicas and dispatch concurrent inference calls.

    ``replicas``/``num_instances`` controls model copies, while
    ``gpus_per_replica`` controls an explicit contiguous GPU group.  Sharing a
    token between replicas is rejected unless ``allow_gpu_sharing`` is enabled.
    Replicas are constructed concurrently after their assignments are frozen.
    """

    def __init__(
        self,
        model_factory: Callable[..., Any],
        model_kwargs: Dict[str, Any],
        gpu_ids: Optional[Sequence[GpuToken]] = None,
        num_instances: Optional[int] = None,
        *,
        replicas: Optional[int] = None,
        gpus_per_replica: int = 1,
        allow_gpu_sharing: bool = False,
        parallel_startup: bool = True,
        startup_workers: Optional[int] = None,
        ready_policy: str = "auto",
    ):
        if replicas is not None and num_instances is not None and replicas != num_instances:
            raise ValueError("replicas and num_instances disagree")
        if replicas is None:
            replicas = num_instances
        if gpu_ids is None:
            gpu_ids = get_available_gpu_ids()
        normalized_gpu_ids = _normalize_gpu_tokens(gpu_ids)
        if not normalized_gpu_ids:
            raise RuntimeError(
                "no visible GPU is available for IsolatedModelPool; check the "
                "scheduler allocation and CUDA_VISIBLE_DEVICES"
            )
        if gpus_per_replica < 1:
            raise ValueError("gpus_per_replica must be >= 1")
        capacity = len(normalized_gpu_ids) // gpus_per_replica
        if capacity < 1:
            raise ValueError(
                f"gpus_per_replica={gpus_per_replica} exceeds the "
                f"{len(normalized_gpu_ids)} visible GPUs"
            )
        if replicas is None:
            replicas = capacity
        if replicas < 1:
            raise ValueError("replicas must be >= 1")
        if ready_policy not in {"auto", "required", "launch"}:
            raise ValueError("ready_policy must be auto, required, or launch")

        self.gpu_ids = normalized_gpu_ids
        self.replicas = int(replicas)
        self.num_instances = int(replicas)  # compatibility attribute
        self.gpus_per_replica = int(gpus_per_replica)
        self.allow_gpu_sharing = bool(allow_gpu_sharing)
        self.ready_policy = ready_policy
        self._pool: queue.Queue = queue.Queue()
        self._models: List[Any] = []
        self._state_lock = threading.RLock()
        self._released = False
        self.replica_metadata: List[Dict[str, Any]] = []

        assignments = self._compute_gpu_assignments(
            self.gpu_ids,
            self.replicas,
            self.gpus_per_replica,
            self.allow_gpu_sharing,
        )
        logger.info(
            "Creating IsolatedModelPool: replicas=%s, gpus_per_replica=%s, "
            "assignments=%s",
            self.replicas,
            self.gpus_per_replica,
            assignments,
        )

        def create(replica_id: int):
            assigned = assignments[replica_id]
            gpu_id = assigned[0] if len(assigned) == 1 else ",".join(assigned)
            kwargs = dict(model_kwargs)
            kwargs["gpu_id"] = gpu_id
            logger.info("Launching model replica %s on GPU(s) %s", replica_id, gpu_id)
            model = None
            try:
                model = model_factory(**kwargs)
                waiter = getattr(model, "wait_until_ready", None)
                if callable(waiter):
                    waiter()
                protocol = getattr(model, "_isolated_ready_protocol", "unknown")
                if self.ready_policy == "required" and protocol != "custom":
                    raise RuntimeError(
                        f"model replica {replica_id} has no strict readiness protocol"
                    )
                return replica_id, assigned, gpu_id, protocol, model
            except BaseException:
                if model is not None:
                    _terminate_model(model)
                raise

        created: Dict[int, tuple] = {}
        errors: List[BaseException] = []
        worker_count = (
            1
            if not parallel_startup
            else min(self.replicas, startup_workers or self.replicas)
        )
        executor = ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="model-replica-start"
        )
        futures = [executor.submit(create, index) for index in range(self.replicas)]
        try:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    created[result[0]] = result
                except BaseException as exc:
                    errors.append(exc)
                    for pending in futures:
                        pending.cancel()
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if errors:
            for result in created.values():
                _terminate_model(result[-1])
            self._released = True
            self.num_instances = 0
            raise errors[0]

        for replica_id in range(self.replicas):
            _, assigned, gpu_id, protocol, model = created[replica_id]
            self._models.append(model)
            self._pool.put(model)
            self.replica_metadata.append(
                {
                    "replica_id": replica_id,
                    "gpu_tokens": list(assigned),
                    "cuda_visible_devices": gpu_id,
                    "ready_protocol": protocol,
                }
            )
        logger.info("IsolatedModelPool initialized with %s ready replicas", self.replicas)

    @staticmethod
    def _compute_gpu_assignments(
        gpu_ids: Sequence[GpuToken],
        num_instances: int,
        gpus_per_replica: int = 1,
        allow_gpu_sharing: bool = False,
    ) -> List[List[str]]:
        tokens = _normalize_gpu_tokens(gpu_ids)
        if not tokens:
            raise ValueError("gpu_ids cannot be empty")
        if num_instances < 1 or gpus_per_replica < 1:
            raise ValueError("num_instances and gpus_per_replica must be >= 1")
        if gpus_per_replica > len(tokens):
            raise ValueError("gpus_per_replica exceeds visible GPU count")
        capacity = len(tokens) // gpus_per_replica
        if num_instances > capacity and not allow_gpu_sharing:
            raise ValueError(
                f"{num_instances} replicas x {gpus_per_replica} GPUs exceeds "
                f"{len(tokens)} visible GPUs; pass allow_gpu_sharing=True only "
                "after validating memory capacity"
            )
        assignments = []
        for replica_id in range(num_instances):
            start = (replica_id * gpus_per_replica) % len(tokens)
            assignments.append(
                [tokens[(start + offset) % len(tokens)] for offset in range(gpus_per_replica)]
            )
        return assignments

    def _acquire(self, timeout: float = None):
        with self._state_lock:
            if self._released:
                raise RuntimeError("model pool has been released")
        model = self._pool.get(timeout=timeout)
        with self._state_lock:
            if self._released:
                raise RuntimeError("model pool was released while acquiring a replica")
        return model

    def _return(self, model):
        with self._state_lock:
            if not self._released:
                self._pool.put(model)

    def inference(self, prompt, **kwargs) -> str:
        model = self._acquire()
        try:
            return model.inference(prompt, **kwargs)
        finally:
            self._return(model)

    def inference_with_runtime(self, prompt, **kwargs):
        """Return adapter metrics before releasing the selected replica."""
        model = self._acquire()
        try:
            measured = getattr(model, "inference_with_runtime", None)
            if callable(measured):
                result = measured(prompt, **kwargs)
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError(
                        "inference_with_runtime must return (output, runtime_metrics)"
                    )
                return result
            output = model.inference(prompt, **kwargs)
            return output, consume_reported_runtime_metrics(model)
        finally:
            self._return(model)

    def release(self):
        """Idempotently release every replica and its isolated process group."""
        with self._state_lock:
            if self._released:
                return
            self._released = True
            models = list(self._models)
            self._models.clear()
            self.num_instances = 0
            while True:
                try:
                    self._pool.get_nowait()
                except queue.Empty:
                    break
        for model in models:
            _terminate_model(model)
        logger.info("IsolatedModelPool released")

    def _cleanup(self):
        self.release()

    def __del__(self):
        try:
            self.release()
        except Exception:
            pass

    def __len__(self):
        return self.num_instances

    @property
    def available_count(self) -> int:
        return self._pool.qsize()
