from typing import Any, Dict

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import coerce_bool, first_present, get_nested


class RuntimeEvaluator(Evaluator):
    """Extract per-sample runtime values; MeshAgg computes percentiles and rates."""

    FIELD_ALIASES = {
        "latency_ms": ("runtime.latency_ms", "latency_ms", "latency"),
        "first_token_latency_ms": ("runtime.first_token_latency_ms", "first_token_latency_ms", "ttft_ms"),
        "first_audio_chunk_latency_ms": ("runtime.first_audio_chunk_latency_ms", "first_audio_chunk_latency_ms", "ttfa_ms"),
        "rtf": ("runtime.rtf", "rtf", "RTF"),
        "memory_peak_mb": ("runtime.memory_peak_mb", "memory_peak_mb", "peak_memory_mb"),
        "cpu_usage": ("runtime.cpu_usage", "cpu_usage"),
        "npu_usage": ("runtime.npu_usage", "npu_usage"),
        "power_watts": ("runtime.power_watts", "power_watts", "power"),
    }

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for output, aliases in self.FIELD_ALIASES.items():
            value = first_present(*(get_nested(kwargs, alias) if "." in alias else kwargs.get(alias) for alias in aliases))
            if value is not None:
                try:
                    result[output] = float(value)
                except (TypeError, ValueError):
                    pass
        timeout = first_present(get_nested(kwargs, "runtime.timeout"), kwargs.get("timeout"))
        failure = first_present(get_nested(kwargs, "runtime.failure"), kwargs.get("failure"), kwargs.get("failed"))
        result["timeout"] = int(bool(coerce_bool(timeout))) if timeout is not None else 0
        result["failure"] = int(bool(coerce_bool(failure))) if failure is not None else 0
        for field in ("failure_stage", "failure_type"):
            value = first_present(
                get_nested(kwargs, f"runtime.{field}"), kwargs.get(field)
            )
            if value not in (None, ""):
                result[field] = str(value)
        return result
