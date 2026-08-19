from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from audio_evals.config import expand_environment


DEFAULT_BENCHMARK_MAP = (
    Path(__file__).resolve().parents[1] / "config" / "benchmark_evaluator_map.yaml"
)


@lru_cache(maxsize=8)
def load_benchmark_map(path: str = "") -> Dict[str, Any]:
    mapping_path = Path(path) if path else DEFAULT_BENCHMARK_MAP
    with open(mapping_path, encoding="utf-8") as handle:
        data = expand_environment(yaml.safe_load(handle) or {})
    if not isinstance(data.get("benchmarks"), dict):
        raise ValueError(f"benchmark map has no 'benchmarks' object: {mapping_path}")
    return data


def get_benchmark(name: str, path: str = "") -> Dict[str, Any]:
    data = load_benchmark_map(path)
    benchmarks = data["benchmarks"]
    if name in benchmarks:
        return dict(benchmarks[name])
    aliases = data.get("aliases") or {}
    canonical = aliases.get(name)
    if canonical in benchmarks:
        return dict(benchmarks[canonical])
    lowered = name.lower()
    for benchmark_name, config in benchmarks.items():
        names = [benchmark_name, *(config.get("aliases") or [])]
        if lowered in {str(item).lower() for item in names}:
            return dict(config)
    raise KeyError(f"benchmark is not mapped: {name}")


def benchmark_profile(name: str, path: str = "") -> Dict[str, Any]:
    config = get_benchmark(name, path)
    required = ("task", "capability", "scenario", "condition", "answer_type", "metrics")
    missing = [field for field in required if config.get(field) in (None, "", [])]
    if missing:
        raise ValueError(f"benchmark {name} is missing profile fields: {missing}")
    return {
        "task": config["task"],
        "capability": config["capability"],
        "scenario": config["scenario"],
        "condition": dict(config["condition"]),
        "answer_type": config["answer_type"],
        "metrics": list(config["metrics"]),
        "evaluators": list(config.get("evaluators") or []),
        "prompt": config.get("prompt", ""),
        "language": config.get("language", ""),
        "target_language": config.get("target_language", ""),
        "use_bucket": config.get("use_bucket", "diagnostic_evidence"),
    }


def apply_benchmark_fallback(
    sample: Dict[str, Any], profile: Dict[str, Any], *, benchmark: str
) -> Dict[str, Any]:
    """Fill missing attributes without overwriting sample-level evidence."""
    result = dict(sample)
    if result.get("schema_version") == "2.0":
        metadata = dict(result.get("metadata") or {})
        provenance = dict(result.get("provenance") or {})
        metadata_provenance = dict(provenance.get("metadata") or {})
        if metadata.get("language") in (None, "", []):
            metadata["language"] = profile.get("language", "") or None
            metadata_provenance["language"] = {
                "source": "benchmark_fallback",
                "confidence": 0.25,
                "evidence": benchmark,
            }
        provenance["metadata"] = metadata_provenance
        hints = dict(result.get("evaluation_hints") or {})
        for target, source in (
            ("metrics", "metrics"),
            ("evaluators", "evaluators"),
            ("prompt", "prompt"),
            ("answer_type", "answer_type"),
        ):
            if hints.get(target) in (None, "", []):
                hints[target] = profile.get(source)
        hints["source"] = "benchmark_fallback"
        result["metadata"] = metadata
        result["provenance"] = provenance
        result["evaluation_hints"] = hints
        return result
    provenance = dict(result.get("attribute_provenance") or {})

    def fallback(field: str, value: Any) -> None:
        if result.get(field) not in (None, "", []):
            return
        result[field] = value
        provenance[field] = {
            "source": "benchmark_fallback",
            "benchmark": benchmark,
            "confidence": 0.25,
        }

    for field in ("task", "capability", "scenario", "answer_type"):
        fallback(field, profile.get(field))
    condition = dict(result.get("condition") or {})
    for group, value in (profile.get("condition") or {}).items():
        if condition.get(group) in (None, "", []):
            condition[group] = value
            provenance[f"condition.{group}"] = {
                "source": "benchmark_fallback",
                "benchmark": benchmark,
                "confidence": 0.25,
            }
    if condition:
        result["condition"] = condition
    metadata = dict(result.get("metadata") or {})
    if metadata.get("language") in (None, "", []):
        metadata["language"] = profile.get("language", "")
        provenance["metadata.language"] = {
            "source": "benchmark_fallback",
            "benchmark": benchmark,
            "confidence": 0.25,
        }
    result["metadata"] = metadata
    for field, profile_field in (
        ("metrics", "metrics"),
        ("evaluators", "evaluators"),
        ("prompt_name", "prompt"),
        ("use_bucket", "use_bucket"),
    ):
        fallback(field, profile.get(profile_field))
    result["attribute_provenance"] = provenance
    return result


def benchmark_names(path: str = "", groups: Optional[Iterable[str]] = None) -> List[str]:
    benchmarks = load_benchmark_map(path)["benchmarks"]
    if not groups:
        return list(benchmarks)
    wanted = set(groups)
    return [name for name, config in benchmarks.items() if config.get("group") in wanted]
