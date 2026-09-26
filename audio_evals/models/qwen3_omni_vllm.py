"""Qwen3-Omni listen→text via vLLM (Instruct / Thinking). Speech stays on HF."""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Dict, Optional

from audio_evals.base import PromptStruct
from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel
from audio_evals.models.qwen3_omni import Qwen3Omni

logger = logging.getLogger(__name__)


@isolated("audio_evals/lib/qwen3-omni-vllm/main.py")
class Qwen3OmniVLLM(Qwen3Omni):
    """Same chat protocol as Qwen3Omni, thinker-only through vLLM."""

    def __init__(
        self,
        path: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        speech: bool = False,
        speaker: str = "Ethan",
        speech_output_dir: str = "",
        max_new_tokens: int = 0,
        max_model_len: int = 32768,
        max_num_seqs: int = 1,
        gpu_memory_utilization: float = 0.88,
        tensor_parallel_size: int = 0,
        limit_mm_audio: int = 16,
        truncation: str = "false",
        temperature: float = 0.0,
        startup_timeout: float = 1800.0,
        request_timeout: float = 3600.0,
        request_log_interval: float = 30.0,
        sample_params: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        if speech:
            raise ValueError(
                "Qwen3OmniVLLM does not support speech=True; use Qwen3Omni for TTS"
            )
        if path == "Qwen/Qwen3-Omni-30B-A3B-Instruct" and not os.path.exists(path):
            path = self._download_model(path)

        self.command_args = {
            "path": path,
            "max-model-len": int(max_model_len),
            "max-num-seqs": int(max_num_seqs),
            "gpu-memory-utilization": float(gpu_memory_utilization),
            "tensor-parallel-size": int(tensor_parallel_size),
            "limit-mm-audio": int(limit_mm_audio),
            "truncation": str(truncation),
            "temperature": float(temperature),
        }
        if max_new_tokens is not None and int(max_new_tokens) > 0:
            self.command_args["thinker-max-new-tokens"] = int(max_new_tokens)

        self._ready = False
        self._startup_timeout = float(startup_timeout)
        self._request_timeout = float(request_timeout)
        self._request_log_interval = float(request_log_interval)
        self._output_queue = None
        self._output_threads = []
        self._recent_process_logs = deque(maxlen=100)
        OfflineModel.__init__(self, is_chat=True, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs):
        return Qwen3Omni._inference(self, prompt, **kwargs)
