"""Versioned runtime measurement contracts and metric normalization."""

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import yaml


DEFAULT_MEASUREMENT_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "measurement_protocols.v2.yaml"
)
REGISTRY_VERSION = "measurement-protocol-registry/2.0"


class MeasurementProtocolError(ValueError):
    pass


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def load_measurement_registry(path: str = "") -> Dict[str, Any]:
    source = Path(path) if path else DEFAULT_MEASUREMENT_PATH
    with open(source, encoding="utf-8") as handle:
        registry = yaml.safe_load(handle) or {}
    if registry.get("schema_version") != REGISTRY_VERSION:
        raise MeasurementProtocolError(
            f"unsupported measurement protocol registry version: {source}"
        )
    if not isinstance(registry.get("profiles"), dict):
        raise MeasurementProtocolError(
            f"measurement protocol registry has no profiles object: {source}"
        )
    return registry


def validate_measurement_registry(path: str = "") -> List[str]:
    registry = load_measurement_registry(path)
    errors: List[str] = []
    allowed_types = {"duration", "ratio", "gauge", "boolean", "taxonomy"}
    for profile_id, profile in registry["profiles"].items():
        if not isinstance(profile, dict):
            errors.append(f"measurement profile {profile_id} must be an object")
            continue
        for field in ("clock", "latency_boundary", "fields", "reliability"):
            if profile.get(field) in (None, "", [], {}):
                errors.append(f"measurement profile {profile_id} is missing {field}")
        for field_name, spec in (profile.get("fields") or {}).items():
            if not isinstance(spec, dict):
                errors.append(
                    f"measurement field {profile_id}/{field_name} must be an object"
                )
                continue
            if spec.get("type") not in allowed_types:
                errors.append(
                    f"measurement field {profile_id}/{field_name} has unknown type: "
                    f"{spec.get('type')}"
                )
            if not spec.get("source"):
                errors.append(
                    f"measurement field {profile_id}/{field_name} has no source"
                )
    return errors


def resolve_measurement_protocol(
    profile_id: str, path: str = ""
) -> Dict[str, Any]:
    registry = load_measurement_registry(path)
    profile = registry["profiles"].get(profile_id)
    if profile is None:
        raise MeasurementProtocolError(
            f"unmapped measurement protocol: {profile_id}"
        )
    resolved = copy.deepcopy(profile)
    resolved["id"] = profile_id
    resolved["registry_version"] = registry["schema_version"]
    resolved["hash"] = _stable_hash(resolved)
    return resolved


def normalize_runtime_metrics(
    values: Any, profile_id: str = "ondevice_audio@1", *, strict: bool = False
) -> Dict[str, Any]:
    """Keep only contract fields and coerce their declared primitive types."""
    if not isinstance(values, dict):
        if strict and values not in (None, ""):
            raise MeasurementProtocolError("runtime metrics must be an object")
        return {}
    fields = resolve_measurement_protocol(profile_id)["fields"]
    normalized: Dict[str, Any] = {}
    for name, value in values.items():
        if name == "audio_duration_seconds":
            try:
                duration = float(value)
            except (TypeError, ValueError):
                if strict:
                    raise MeasurementProtocolError(
                        "audio_duration_seconds must be numeric"
                    )
                continue
            if duration > 0:
                normalized[name] = duration
            elif strict:
                raise MeasurementProtocolError(
                    "audio_duration_seconds must be positive"
                )
            continue
        spec = fields.get(name)
        if spec is None:
            if strict:
                raise MeasurementProtocolError(f"unknown runtime field: {name}")
            continue
        field_type = spec["type"]
        if field_type in {"duration", "ratio", "gauge"}:
            try:
                number = float(value)
            except (TypeError, ValueError):
                if strict:
                    raise MeasurementProtocolError(
                        f"runtime field {name} must be numeric"
                    )
                continue
            if number < 0:
                if strict:
                    raise MeasurementProtocolError(
                        f"runtime field {name} must be non-negative"
                    )
                continue
            normalized[name] = number
        elif field_type == "boolean":
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"1", "true", "yes"}:
                    value = True
                elif lowered in {"0", "false", "no", ""}:
                    value = False
                elif strict:
                    raise MeasurementProtocolError(
                        f"runtime field {name} must be boolean"
                    )
                else:
                    continue
            normalized[name] = int(bool(value))
        elif value not in (None, ""):
            normalized[name] = str(value)
    return normalized


def consume_reported_runtime_metrics(source: Any) -> Dict[str, Any]:
    """Read optional per-request metrics while an adapter/replica is still owned.

    Adapters should prefer ``consume_runtime_metrics()`` because it makes the
    one-request ownership explicit. ``get_runtime_metrics()`` and a
    ``last_runtime_metrics`` mapping are compatibility fallbacks.
    """
    for method_name in ("consume_runtime_metrics", "get_runtime_metrics"):
        method = getattr(source, method_name, None)
        if callable(method):
            value = method()
            return dict(value) if isinstance(value, dict) else {}
    value = getattr(source, "last_runtime_metrics", None)
    return dict(value) if isinstance(value, dict) else {}
