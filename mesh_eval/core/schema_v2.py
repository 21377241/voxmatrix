"""Canonical V2 sample, V1 adapter, and RuntimeSample/ResultEvent projections."""

import copy
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from mesh_eval.core.protocol import (
    CapabilityProtocolRegistry,
    ProtocolResolutionError,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "annotation/schema/taxonomy.v2.yaml"
CANONICAL_SCHEMA_PATH = REPO_ROOT / "annotation/schema/evaluation_sample.v2.schema.json"
RUNTIME_SCHEMA_PATH = REPO_ROOT / "mesh_eval/schema/runtime_sample.v2.schema.json"
EVENT_SCHEMA_PATH = REPO_ROOT / "mesh_eval/schema/result_event.v2.schema.json"
CANONICAL_VERSION = "2.0"
RUNTIME_VERSION = "runtime-sample/2.0"
EVENT_VERSION = "result-event/2.0"


class V2MigrationError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_taxonomy_v2() -> Dict[str, Any]:
    with open(TAXONOMY_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def is_canonical_v2(record: Dict[str, Any]) -> bool:
    return str(record.get("schema_version") or "") == CANONICAL_VERSION


def is_runtime_v2(record: Dict[str, Any]) -> bool:
    return str(record.get("schema_version") or "") == RUNTIME_VERSION


def _provenance_source(value: Any) -> str:
    source = str(value or "").lower()
    return {
        "native": "dataset_native",
        "rule": "legacy_adapter",
        "acoustic_model": "acoustic_model",
        "llm_draft": "llm_draft",
        "knowledge_verified": "verified_derived",
        "knowledge_revise": "verified_derived",
        "knowledge_flagged": "unknown",
        "human": "human",
    }.get(source, "legacy_adapter")


def _value_present(value: Any) -> bool:
    return value not in (None, "", [])


def _reference_text(reference: Dict[str, Any]) -> str:
    for key in (
        "text",
        "answer",
        "caption",
        "summary",
        "transcript",
        "label",
    ):
        if _value_present(reference.get(key)):
            return str(reference[key])
    return ""


def _audio_descriptors(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    input_obj = record.get("input") or {}
    if isinstance(input_obj.get("audio"), list):
        return [dict(item) for item in input_obj["audio"] if isinstance(item, dict)]
    candidates: List[Tuple[int, Any]] = []
    primary = record.get("WavPath") or input_obj.get("audio_path")
    if primary:
        candidates.append((0, primary))
    for key, value in record.items():
        if key.startswith("WavPath") and key[7:].isdigit() and value:
            candidates.append((int(key[7:]), value))
    second = input_obj.get("audio_path_b")
    if second:
        candidates.append((2, second))
    for index, item in enumerate(input_obj.get("audios") or [], start=1):
        if isinstance(item, dict) and (item.get("uri") or item.get("path")):
            descriptor = dict(item)
            descriptor["uri"] = descriptor.get("uri") or descriptor.pop("path")
            candidates.append((int(item.get("turn") or index), descriptor))
    descriptors = []
    seen = set()
    for _index, value in sorted(candidates, key=lambda item: item[0]):
        descriptor = dict(value) if isinstance(value, dict) else {"uri": str(value)}
        uri = str(descriptor.get("uri") or "")
        if uri and uri not in seen:
            descriptor["uri"] = uri
            descriptors.append(descriptor)
            seen.add(uri)
    return descriptors


def _map_task_capability(task: str, capability: str) -> Tuple[str, str, List[str]]:
    warnings = []
    taxonomy = load_taxonomy_v2().get("tasks") or {}
    if task in taxonomy and capability in taxonomy[task].get("capabilities", []):
        return task, capability, warnings
    if task == "runtime":
        raise V2MigrationError(
            "legacy runtime Task is a measurement family and has no canonical V2 Task"
        )
    t1 = {"asr", "long_form_asr", "speech_translation", "code_switch"}
    t2 = {
        "audio_reasoning",
        "audio_caption",
        "sound_event",
        "acoustic_scene",
        "paralinguistic",
        "paralinguistic_recognition",
        "qa",
        "meeting_summary",
        "meeting_qa",
        "timeline_query",
        "todo_extraction",
    }
    t3 = {
        "speaker_count",
        "speaker_verification",
        "diarization",
        "speaker_attribution",
        "target_speaker",
        "spoof_detection",
    }
    t4 = {
        "instruction_following",
        "tool_call",
        "navigation",
        "calendar",
        "reminder",
        "phone_message",
        "music_control",
        "vehicle_control",
        "translation_instruction",
        "multi_turn_dialogue",
        "clarification",
        "interruption",
    }
    if capability in t1:
        new_task = "speech_recognition_transcription"
    elif capability in t2 or (task == "agent" and capability == "qa"):
        new_task = "audio_speech_understanding"
    elif capability in t3 or task == "speaker":
        new_task = "speaker_attribution"
    elif capability in t4 or task == "agent":
        new_task = "spoken_agentic_interaction"
    elif task == "speech_output":
        new_task, capability = "speech_generation_delivery", "speech_generation"
        warnings.append(
            "legacy speech_output capability retained as an evaluation dimension"
        )
    else:
        raise V2MigrationError(
            f"cannot map legacy task/capability without evidence: {task}/{capability}"
        )
    if capability == "paralinguistic":
        capability = "paralinguistic_recognition"
    return new_task, capability, warnings


def adapt_v1_to_v2(record: Dict[str, Any], ref_col: str = "text") -> Dict[str, Any]:
    """Deterministically adapt V1 without guessing an unsupported scenario."""
    if is_canonical_v2(record):
        return copy.deepcopy(record)
    source = copy.deepcopy(record)
    old_task = str(source.get("task") or source.get("task_name") or "")
    old_capability = str(source.get("capability") or "")
    task, capability, warnings = _map_task_capability(old_task, old_capability)

    label_meta = source.get("label_meta") or {}
    label_sources = label_meta.get("sources") or {}
    metadata = copy.deepcopy(source.get("metadata") or {})
    metadata.pop("metrics", None)
    metadata_provenance: Dict[str, Any] = {}

    def set_metadata(name: str, value: Any, source_key: str, confidence=0.6):
        if not _value_present(metadata.get(name)) and _value_present(value):
            metadata[name] = value
        if _value_present(metadata.get(name)):
            metadata_provenance.setdefault(
                name,
                {
                    "source": _provenance_source(label_sources.get(source_key)),
                    "confidence": confidence,
                    "derivation_version": "v1-to-v2@1",
                },
            )

    condition = source.get("condition") or {}
    if not isinstance(condition, dict):
        condition = {}
        warnings.append("legacy condition was not an object")
    device = condition.get("device")
    if device:
        set_metadata("device_type", device, "condition.device")
    spatial = condition.get("spatial")
    if spatial in {"near_field", "far_field"}:
        set_metadata("recording_setup", spatial, "condition.spatial")
    elif spatial:
        warnings.append(f"legacy spatial value retained only in legacy: {spatial}")
    acoustic = condition.get("acoustic")
    noise_map = {
        "clean": "clean",
        "traffic_noise": "traffic",
        "crowd_noise": "crowd",
        "music_noise": "music",
        "wind_noise": "wind",
        "reverb": "reverberation",
        "overlap_speech": "speech",
    }
    if acoustic in noise_map:
        set_metadata("noise_type", noise_map[acoustic], "condition.acoustic")
    elif acoustic:
        warnings.append(f"ambiguous legacy acoustic value retained: {acoustic}")
    speaker = condition.get("speaker")
    if speaker == "single_speaker":
        set_metadata("speaker_count", 1, "condition.speaker")
    elif speaker:
        warnings.append(f"legacy speaker value retained only in legacy: {speaker}")
    if acoustic == "overlap_speech":
        set_metadata("overlap_level", "mild", "condition.acoustic", confidence=0.4)

    old_scenario = source.get("scenario")
    if isinstance(old_scenario, dict):
        scenario = {
            "primary": str(old_scenario.get("primary") or "unknown"),
            "secondary": old_scenario.get("secondary"),
        }
    elif old_scenario == "phone":
        scenario = {"primary": "unknown", "secondary": None}
        set_metadata("device_type", metadata.get("device_type") or "phone", "scenario")
        warnings.append(
            "legacy scenario=phone migrated to metadata.device_type; scenario left unknown"
        )
    elif old_scenario in {"home", "vehicle", "meeting", "public", "outdoor"}:
        scenario = {"primary": old_scenario, "secondary": None}
    else:
        scenario = {"primary": "unknown", "secondary": None}
        warnings.append("scenario has no supported evidence and remains unknown")

    defaults = {
        "device_type": "unknown",
        "recording_setup": "unknown",
        "noise_type": "unknown",
        "snr_bucket": "unknown",
        "speaker_count": None,
        "overlap_level": "unknown",
    }
    for field, default in defaults.items():
        if field not in metadata:
            metadata[field] = default
            metadata_provenance[field] = {
                "source": "unknown",
                "confidence": 0.0,
                "derivation_version": "v1-to-v2@1",
            }

    reference = source.get("reference_obj", source.get("reference", {}))
    if not isinstance(reference, dict):
        reference = {"text": str(reference)}
    if not reference and _value_present(source.get(ref_col)):
        reference = {"text": source[ref_col]}
    input_obj = copy.deepcopy(source.get("input") or {})
    for legacy_key in ("audio_path", "audio_path_b", "audios"):
        input_obj.pop(legacy_key, None)
    audios = _audio_descriptors(source)
    if audios:
        input_obj["audio"] = audios
    for key in ("question", "instruction", "choices"):
        if _value_present(source.get(key)) and key not in input_obj:
            input_obj[key] = copy.deepcopy(source[key])

    split = str(source.get("split") or "test").lower()
    split = {
        "validation": "dev",
        "valid": "dev",
        "eval": "test",
        "evaluation": "test",
        "all": "test",
        "mini": "test",
        "smoke": "test",
    }.get(split, split)
    use_bucket = str(source.get("use_bucket") or "diagnostic_evidence")
    unknown_core = [
        field
        for field, value in metadata.items()
        if field in defaults and value in (None, "unknown", "")
    ]
    if use_bucket == "formal_subscores" and (
        scenario["primary"] == "unknown" or unknown_core
    ):
        use_bucket = "coverage_debt"
        warnings.append(
            "formal_subscores downgraded to coverage_debt because V2 attributes are unknown"
        )

    hints = {
        "prompt": str(source.get("prompt_name") or source.get("prompt") or ""),
        "metrics": list(source.get("metrics") or []),
        "evaluators": list(source.get("evaluators") or []),
        "answer_type": str(source.get("answer_type") or ""),
        "source": "legacy_adapter",
    }
    hints = {key: value for key, value in hints.items() if _value_present(value)}
    sample_id = str(
        source.get("sample_id")
        or source.get("id")
        or f"{source.get('dataset_id') or source.get('dataset') or 'sample'}-unknown"
    )
    label_provenance = {
        key: {
            "source": _provenance_source(value),
            "confidence": float(label_meta.get("confidence", 1.0)),
            "derivation_version": "v1-to-v2@1",
        }
        for key, value in label_sources.items()
        if key.startswith("reference.") or key in reference
    }
    result = {
        "schema_version": CANONICAL_VERSION,
        "sample_id": sample_id,
        "dataset": str(source.get("dataset_id") or source.get("dataset") or "unknown"),
        "task": task,
        "capability": capability,
        "capability_protocol": f"{task}/{capability}@1",
        "scenario": scenario,
        "metadata": metadata,
        "input": input_obj,
        "reference": reference,
        "provenance": {
            "scenario": {
                "source": _provenance_source(label_sources.get("scenario")),
                "confidence": 0.5 if scenario["primary"] != "unknown" else 0.0,
                "derivation_version": "v1-to-v2@1",
            },
            "metadata": metadata_provenance,
            "labels": label_provenance,
        },
        "use_bucket": use_bucket,
        "split": split,
        "legacy": {
            "schema_version": str(source.get("schema_version") or "1.0"),
            "task": old_task,
            "capability": old_capability,
            "scenario": old_scenario,
            "condition": condition,
            "requested_use_bucket": source.get("use_bucket"),
        },
        "conversion_warnings": list(dict.fromkeys(warnings)),
    }
    if hints:
        result["evaluation_hints"] = hints
    if old_task == "speech_output":
        result["legacy"]["speech_output_dimension"] = old_capability
    return result


def validate_canonical_sample(
    sample: Dict[str, Any], *, formal: bool = True, check_audio_exists=False, cwd=""
) -> List[str]:
    errors = []
    required = (
        "schema_version",
        "sample_id",
        "dataset",
        "task",
        "capability",
        "capability_protocol",
        "scenario",
        "metadata",
        "input",
        "reference",
        "provenance",
        "use_bucket",
        "split",
    )
    for field in required:
        if not _value_present(sample.get(field)):
            errors.append(f"missing required V2 field: {field}")
    if sample.get("schema_version") != CANONICAL_VERSION:
        errors.append(f"unsupported canonical schema_version: {sample.get('schema_version')}")
    taxonomy = load_taxonomy_v2()
    task = str(sample.get("task") or "")
    capability = str(sample.get("capability") or "")
    task_spec = (taxonomy.get("tasks") or {}).get(task)
    if task_spec is None:
        errors.append(f"unknown V2 task: {task}")
    elif capability not in (task_spec.get("capabilities") or []):
        errors.append(f"capability {capability} does not belong to V2 task {task}")
    scenario = sample.get("scenario") or {}
    if not isinstance(scenario, dict):
        errors.append("V2 scenario must be an object")
    else:
        primary = scenario.get("primary")
        allowed_primary = (taxonomy.get("scenario") or {}).get("primary") or []
        if primary not in allowed_primary:
            errors.append(f"unknown V2 scenario.primary: {primary}")
        secondary = scenario.get("secondary")
        allowed_secondary = (
            ((taxonomy.get("scenario") or {}).get("secondary") or {}).get(primary, [])
        )
        if secondary not in (None, "") and secondary not in allowed_secondary:
            errors.append(
                f"scenario.secondary {secondary} does not belong to {primary}"
            )
    metadata = sample.get("metadata") or {}
    if not isinstance(metadata, dict):
        errors.append("V2 metadata must be an object")
        metadata = {}
    metadata_spec = taxonomy.get("metadata") or {}
    enums = metadata_spec.get("enums") or {}
    for field, allowed in enums.items():
        value = metadata.get(field)
        values = value if isinstance(value, list) else [value]
        for item in values:
            if item not in (None, "") and item not in allowed:
                errors.append(f"unknown metadata.{field}: {item}")
    speaker_count = metadata.get("speaker_count")
    if speaker_count is not None and (
        not isinstance(speaker_count, int) or isinstance(speaker_count, bool) or speaker_count < 0
    ):
        errors.append("metadata.speaker_count must be a non-negative integer or null")
    if formal and sample.get("use_bucket") == "formal_subscores":
        for field in metadata_spec.get("core_fields") or []:
            if metadata.get(field) in (None, "", "unknown"):
                errors.append(f"formal V2 sample has unknown core metadata: {field}")
        if isinstance(scenario, dict) and scenario.get("primary") == "unknown":
            errors.append("formal V2 sample has unknown scenario.primary")
    audios = (sample.get("input") or {}).get("audio") or []
    if check_audio_exists:
        for item in audios:
            uri = item.get("uri") if isinstance(item, dict) else ""
            full_path = uri if os.path.isabs(str(uri)) else os.path.join(cwd, str(uri))
            if not uri or not os.path.isfile(full_path):
                errors.append(f"audio path not found: {uri}")
    try:
        CapabilityProtocolRegistry(
            mode="formal" if formal else "diagnostic"
        ).resolve(sample)
    except ProtocolResolutionError as exc:
        errors.append(str(exc))
    return errors


def canonicalize_sample(
    record: Dict[str, Any], ref_col: str = "text", *, validate=True
) -> Dict[str, Any]:
    sample = copy.deepcopy(record) if is_canonical_v2(record) else adapt_v1_to_v2(record, ref_col)
    if validate:
        errors = validate_canonical_sample(sample, formal=False)
        if errors:
            raise V2MigrationError("; ".join(errors))
    return sample


def to_runtime_sample(
    sample: Dict[str, Any],
    *,
    ref_col: str = "text",
    protocol_mode: str = "formal",
    run_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    canonical = canonicalize_sample(sample, ref_col, validate=True)
    resolved = CapabilityProtocolRegistry(mode=protocol_mode).resolve(canonical)
    reference = copy.deepcopy(canonical.get("reference") or {})
    input_obj = copy.deepcopy(canonical.get("input") or {})
    context = copy.deepcopy(run_context or {})
    context.setdefault("measurement_protocol", resolved.get("runtime_profile", ""))
    context.setdefault(
        "measurement_protocol_hash", resolved.get("runtime_profile_hash", "")
    )
    context.setdefault("request_mode", "unspecified")
    runtime = {
        "schema_version": RUNTIME_VERSION,
        "canonical_schema_version": CANONICAL_VERSION,
        "canonical": canonical,
        "resolved_protocol": resolved,
        "run_context": context,
        "sample_id": canonical["sample_id"],
        "dataset_id": canonical["dataset"],
        "dataset": canonical["dataset"],
        "task": canonical["task"],
        "capability": canonical["capability"],
        "capability_protocol": canonical["capability_protocol"],
        "scenario": copy.deepcopy(canonical["scenario"]),
        "metadata": copy.deepcopy(canonical["metadata"]),
        "provenance": copy.deepcopy(canonical["provenance"]),
        "input": input_obj,
        "reference_obj": reference,
        "use_bucket": canonical["use_bucket"],
        "split": canonical["split"],
        "metrics": list(resolved.get("metrics") or []),
        "evaluators": list(resolved.get("evaluators") or []),
        "prompt_name": str(resolved.get("prompt") or ""),
        "answer_type": str(resolved.get("answer_type") or resolved.get("output_schema") or ""),
        "parser": str(resolved.get("parser") or ""),
        "protocol_hash": resolved["hash"],
        "measurement_protocol": resolved.get("runtime_profile", ""),
        "measurement_protocol_hash": resolved.get("runtime_profile_hash", ""),
    }
    runtime[ref_col] = _reference_text(reference)
    primary = canonical["scenario"].get("primary")
    secondary = canonical["scenario"].get("secondary")
    runtime["scenario__primary"] = primary
    if secondary:
        runtime["scenario__secondary"] = secondary
    for field in (load_taxonomy_v2().get("metadata") or {}).get("slice_fields") or []:
        value = canonical["metadata"].get(field)
        if _value_present(value):
            runtime[f"metadata__{field}"] = (
                "|".join(str(item) for item in value)
                if isinstance(value, list)
                else value
            )
    runtime["language"] = str(canonical["metadata"].get("language") or "")
    audios = input_obj.get("audio") or []
    for index, audio in enumerate(audios):
        if not isinstance(audio, dict) or not audio.get("uri"):
            continue
        key = "WavPath" if index == 0 else f"WavPath{index + 1}"
        runtime[key] = str(audio["uri"])
    for field in ("question", "instruction", "choices", "turns"):
        if field in input_obj:
            runtime[field] = copy.deepcopy(input_obj[field])
    return runtime


def result_event_v2(
    event_type: str,
    event_id: Any,
    runtime_sample: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    dimensions = {
        key: copy.deepcopy(runtime_sample[key])
        for key in (
            "sample_id",
            "dataset_id",
            "task",
            "capability",
            "scenario",
            "metadata",
            "use_bucket",
            "split",
        )
        if key in runtime_sample
    }
    return {
        "schema_version": EVENT_VERSION,
        "type": event_type,
        "id": event_id,
        "sample_id": str(runtime_sample.get("sample_id") or event_id),
        "dimensions": dimensions,
        "resolved_protocol": copy.deepcopy(runtime_sample.get("resolved_protocol") or {}),
        "run_context": copy.deepcopy(runtime_sample.get("run_context") or {}),
        "data": copy.deepcopy(data),
    }


def validate_json_schema(record: Dict[str, Any], kind: str = "canonical") -> List[str]:
    paths = {
        "canonical": CANONICAL_SCHEMA_PATH,
        "runtime": RUNTIME_SCHEMA_PATH,
        "event": EVENT_SCHEMA_PATH,
    }
    if kind not in paths:
        raise ValueError(f"unknown V2 schema kind: {kind}")
    try:
        from jsonschema import Draft7Validator
    except ImportError:
        return []
    with open(paths[kind], encoding="utf-8") as handle:
        validator = Draft7Validator(json.load(handle))
    return [
        f"{'.'.join(str(part) for part in error.path) or '$'}: {error.message}"
        for error in sorted(
            validator.iter_errors(record),
            key=lambda item: tuple(str(part) for part in item.path),
        )
    ]
