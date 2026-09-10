"""Versioned CapabilityProtocol resolution for V2 evaluation samples."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from mesh_eval.core.measurement import (
    MeasurementProtocolError,
    resolve_measurement_protocol,
    validate_measurement_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "mesh_eval/config/capability_protocols.v2.yaml"
METRIC_CATALOG_PATH = REPO_ROOT / "mesh_eval/config/metric_evaluator_catalog.yaml"
TAXONOMY_V2_PATH = REPO_ROOT / "annotation/schema/taxonomy.v2.yaml"

REQUIRED_PROTOCOL_FIELDS = (
    "task",
    "capability",
    "input_schema",
    "output_schema",
    "parser",
    "required_reference",
    "prompt",
    "metrics",
)


class ProtocolResolutionError(ValueError):
    pass


def validate_protocol_output(prediction: Any, resolved: Dict[str, Any]) -> None:
    """Fail closed on the parser/output shape declared by a formal protocol."""
    parser = str(resolved.get("parser") or "")
    if prediction is None:
        raise ProtocolResolutionError("formal protocol output is null")
    if parser in {"transcript_text@1", "answer_text@1"}:
        if not isinstance(prediction, str) or not prediction.strip():
            raise ProtocolResolutionError(f"{parser} requires non-empty text output")
        return
    if parser == "classification@1":
        if isinstance(prediction, (dict, list)) or not str(prediction).strip():
            raise ProtocolResolutionError(
                "classification@1 requires one scalar/non-empty label"
            )
        return
    if parser in {"structured_json@1", "dialogue@1"}:
        value = prediction
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ProtocolResolutionError(
                    f"{parser} received invalid JSON: {exc}"
                ) from exc
        if not isinstance(value, (dict, list)):
            raise ProtocolResolutionError(f"{parser} requires an object or array")
        return
    if parser == "generated_audio@1":
        value = prediction
        if isinstance(value, str) and Path(value).is_file():
            return
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = None
        if not isinstance(value, dict) or not value.get("audio"):
            raise ProtocolResolutionError(
                "generated_audio@1 requires an audio path or {audio: path} object"
            )
        return
    if parser == "identity@0":
        return
    raise ProtocolResolutionError(f"unregistered formal parser: {parser}")


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _get_path(value: Dict[str, Any], path: str):
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


@lru_cache(maxsize=8)
def load_protocol_registry(path: str = "") -> Dict[str, Any]:
    source = Path(path) if path else DEFAULT_PROTOCOL_PATH
    with open(source, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if data.get("schema_version") != "capability-protocol-registry/2.0":
        raise ValueError(f"unsupported protocol registry version: {source}")
    if not isinstance(data.get("protocols"), dict):
        raise ValueError(f"protocol registry has no protocols object: {source}")
    return data


@lru_cache(maxsize=1)
def _metric_catalog() -> Dict[str, Any]:
    with open(METRIC_CATALOG_PATH, encoding="utf-8") as handle:
        return (yaml.safe_load(handle) or {}).get("metrics") or {}


def validate_protocol_registry(path: str = "") -> List[str]:
    data = load_protocol_registry(path)
    protocols = data["protocols"]
    errors = []
    with open(TAXONOMY_V2_PATH, encoding="utf-8") as handle:
        taxonomy = yaml.safe_load(handle) or {}
    expected = {
        f"{task}/{capability}@1"
        for task, task_spec in (taxonomy.get("tasks") or {}).items()
        for capability in task_spec.get("capabilities") or []
    }
    missing_protocols = sorted(expected - set(protocols))
    errors.extend(f"missing protocol: {item}" for item in missing_protocols)
    errors.extend(validate_measurement_registry())
    runtime_profile = str(data.get("runtime_profile") or "")
    try:
        resolve_measurement_protocol(runtime_profile)
    except MeasurementProtocolError as exc:
        errors.append(str(exc))
    for protocol_id, spec in protocols.items():
        if not isinstance(spec, dict):
            errors.append(f"protocol {protocol_id} must be an object")
            continue
        for field in REQUIRED_PROTOCOL_FIELDS:
            if spec.get(field) in (None, "", []):
                errors.append(f"protocol {protocol_id} is missing {field}")
        expected_id = f"{spec.get('task')}/{spec.get('capability')}@1"
        if protocol_id != expected_id:
            errors.append(
                f"protocol ID/task/capability mismatch: {protocol_id} != {expected_id}"
            )
        for metric in spec.get("metrics") or []:
            if metric not in _metric_catalog():
                errors.append(f"protocol {protocol_id} has unknown metric: {metric}")
    return errors


def _language_key(sample: Dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    raw = metadata.get("language") if isinstance(metadata, dict) else ""
    raw = raw or sample.get("language") or ""
    language = str(raw).strip().lower().replace("_", "-")
    if language.startswith(("zh", "cmn", "yue")) or language in {
        "chinese",
        "mandarin",
        "cantonese",
    }:
        return "zh"
    if language.startswith("en") or language == "english":
        return "en"
    return language


def _metric_evaluators(
    metrics: List[str],
    formal: bool,
    sample: Dict[str, Any],
    *,
    explicit_override: bool = False,
) -> Tuple[List[str], List[str]]:
    evaluators = []
    missing = []
    warnings = []
    catalog = _metric_catalog()
    for metric in metrics:
        config = catalog.get(metric)
        if not config:
            missing.append(metric)
            continue
        language_routes = config.get("evaluator_by_language") or {}
        if language_routes:
            language = _language_key(sample)
            declared = language_routes.get(language)
            if not declared:
                message = (
                    f"metric {metric} requires metadata.language mapped in "
                    f"{sorted(language_routes)} or an explicit evaluator override"
                )
                if formal and not explicit_override:
                    raise ProtocolResolutionError(message)
                warnings.append(message)
                continue
        else:
            declared = config.get("evaluator")
        values = declared if isinstance(declared, list) else [declared]
        evaluators.extend(str(item) for item in values if item)
    if formal and missing:
        raise ProtocolResolutionError(
            "unmapped formal metrics: " + ", ".join(sorted(missing))
        )
    return list(dict.fromkeys(evaluators)), list(dict.fromkeys(warnings))


def _validate_input(sample: Dict[str, Any], input_schema: str) -> None:
    input_obj = sample.get("input") or {}
    audios = input_obj.get("audio") or []
    if input_schema.startswith("audio_") and not audios:
        raise ProtocolResolutionError(f"{input_schema} requires input.audio")
    if input_schema.startswith("audio_pair") and len(audios) < 2:
        raise ProtocolResolutionError("audio_pair@1 requires at least two audio items")
    if input_schema == "speech_generation_input@1" and not (
        input_obj.get("text") or audios
    ):
        raise ProtocolResolutionError(
            "speech_generation_input@1 requires input.text or input.audio"
        )


class CapabilityProtocolRegistry:
    def __init__(self, path: str = "", mode: str = "formal"):
        if mode not in {"formal", "diagnostic"}:
            raise ValueError("protocol mode must be formal or diagnostic")
        self.mode = mode
        self.data = load_protocol_registry(path)

    @property
    def formal(self) -> bool:
        return self.mode == "formal"

    def resolve(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        task = str(sample.get("task") or "")
        capability = str(sample.get("capability") or "")
        protocol_id = str(
            sample.get("capability_protocol") or f"{task}/{capability}@1"
        )
        protocol = self.data["protocols"].get(protocol_id)
        runtime_profile = str(self.data.get("runtime_profile") or "")
        try:
            measurement = resolve_measurement_protocol(runtime_profile)
        except MeasurementProtocolError as exc:
            raise ProtocolResolutionError(str(exc)) from exc
        if protocol is None:
            if self.formal:
                raise ProtocolResolutionError(
                    f"formal sample has no mapped protocol: {protocol_id}"
                )
            hints = sample.get("evaluation_hints") or {}
            fallback = {
                "id": protocol_id,
                "task": task,
                "capability": capability,
                "input_schema": "unmapped@0",
                "output_schema": "unmapped@0",
                "parser": "identity@0",
                "required_reference": [],
                "prompt": str(hints.get("prompt") or ""),
                "metrics": list(hints.get("metrics") or []),
                "evaluators": list(hints.get("evaluators") or []),
                "runtime_profile": runtime_profile,
                "runtime_profile_hash": measurement["hash"],
                "resolution_source": "diagnostic_fallback",
                "warnings": [f"unmapped protocol: {protocol_id}"],
            }
            fallback["hash"] = _stable_hash(fallback)
            return fallback

        if protocol.get("task") != task or protocol.get("capability") != capability:
            raise ProtocolResolutionError(
                f"sample task/capability does not match protocol {protocol_id}"
            )
        hints = sample.get("evaluation_hints") or {}
        structural_overrides = {}
        for field in ("input_schema", "output_schema", "parser", "required_reference"):
            if hints.get(field) not in (None, "", []):
                if self.formal:
                    raise ProtocolResolutionError(
                        f"formal protocol does not allow sample-level {field} override"
                    )
                structural_overrides[field] = hints[field]
        effective_input_schema = str(
            structural_overrides.get("input_schema") or protocol["input_schema"]
        )
        effective_required_reference = structural_overrides.get(
            "required_reference", protocol.get("required_reference") or []
        )
        _validate_input(sample, effective_input_schema)
        reference = sample.get("reference") or {}
        missing_reference = [
            field
            for field in effective_required_reference
            if _get_path(reference, field) in (None, "", [])
        ]
        if missing_reference and self.formal:
            raise ProtocolResolutionError(
                f"protocol {protocol_id} requires reference fields: "
                + ", ".join(missing_reference)
            )

        resolved = dict(protocol)
        resolved["id"] = protocol_id
        resolved["runtime_profile"] = runtime_profile
        resolved["runtime_profile_hash"] = measurement["hash"]
        resolved["resolution_source"] = "protocol_registry"
        overrides = {}
        for target, value in structural_overrides.items():
            resolved[target] = value
            overrides[target] = {
                "source": str(hints.get("source") or "sample_evaluation_hint"),
                "value": value,
            }
        for source, target in (
            ("prompt", "prompt"),
            ("metrics", "metrics"),
            ("evaluators", "evaluators"),
            ("answer_type", "answer_type"),
        ):
            if hints.get(source) not in (None, "", []):
                resolved[target] = hints[source]
                overrides[target] = {
                    "source": str(hints.get("source") or "sample_evaluation_hint"),
                    "value": hints[source],
                }
        resolved["metrics"] = [str(item) for item in resolved.get("metrics") or []]
        explicit_evaluators = bool(resolved.get("evaluators"))
        catalog_evaluators, evaluator_warnings = _metric_evaluators(
            resolved["metrics"],
            self.formal,
            sample,
            explicit_override=explicit_evaluators,
        )
        if resolved.get("evaluators"):
            resolved["evaluators"] = [
                str(item) for item in resolved.get("evaluators") or []
            ]
        else:
            resolved["evaluators"] = catalog_evaluators
        if overrides:
            resolved["overrides"] = overrides
        warnings = list(evaluator_warnings)
        if missing_reference:
            warnings.append(
                "missing reference fields: " + ", ".join(missing_reference)
            )
        if warnings:
            resolved["warnings"] = list(dict.fromkeys(warnings))
        resolved["hash"] = _stable_hash(resolved)
        return resolved
