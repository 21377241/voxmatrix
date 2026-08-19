import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


CONDITION_GROUPS = ("acoustic", "spatial", "speaker", "device", "interaction")
REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "annotation" / "schema" / "taxonomy.yaml"
METRIC_CATALOG_PATH = REPO_ROOT / "mesh_eval" / "config" / "metric_evaluator_catalog.yaml"


@lru_cache(maxsize=1)
def load_taxonomy() -> Dict[str, Any]:
    with open(TAXONOMY_PATH, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@lru_cache(maxsize=1)
def metric_catalog_names() -> frozenset[str]:
    with open(METRIC_CATALOG_PATH, encoding="utf-8") as handle:
        catalog = yaml.safe_load(handle) or {}
    return frozenset(str(name) for name in (catalog.get("metrics") or {}))


def as_list(value: Any) -> List[Any]:
    if value in (None, "", []):
        return []
    if isinstance(value, list):
        return value
    return [value]


def strip_prefix(value: Any) -> str:
    text = str(value).strip()
    return text.split(":", 1)[1] if ":" in text else text


def normalize_metric_name(value: Any) -> str:
    name = strip_prefix(value).lower()
    aliases = load_taxonomy().get("metric_aliases") or {}
    return str(aliases.get(name, name))


def normalize_metrics(value: Any) -> List[str]:
    raw_values: List[Any] = []
    for item in as_list(value):
        if isinstance(item, str):
            raw_values.extend(part for part in re.split(r"[,|]", item) if part.strip())
        else:
            raw_values.append(item)
    return list(
        dict.fromkeys(
            name for name in (normalize_metric_name(item) for item in raw_values) if name
        )
    )


def normalize_split(value: Any) -> str:
    split = strip_prefix(value).lower().strip()
    aliases = load_taxonomy().get("split_aliases") or {}
    return str(aliases.get(split, split))


def first_present(*values: Any) -> Optional[Any]:
    for value in values:
        if value not in (None, "", []):
            return value
    return None


def first_tag(tags: Dict[str, Any], key: str) -> str:
    values = as_list(tags.get(key))
    return strip_prefix(values[0]) if values else ""


def compact(value: Any) -> Any:
    if isinstance(value, list):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return "|".join(f"{k}={v}" for k, v in sorted(value.items()))
    return value


def get_path(doc: Dict[str, Any], path: str) -> Any:
    value: Any = doc
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def infer_answer_type(record: Dict[str, Any]) -> str:
    reference = record.get("reference") or {}
    target = record.get("target") or {}
    if first_present(record.get("answer_type"), target.get("answer_type")):
        return str(first_present(record.get("answer_type"), target.get("answer_type")))
    if first_present(
        reference.get("json"),
        reference.get("expected_output"),
        reference.get("segments"),
        reference.get("utterances"),
        target.get("reference_json"),
    ):
        return "structured_json"
    if first_present(reference.get("audio_uri"), target.get("reference_audio_uri")):
        return "audio"
    if first_present(
        reference.get("label"),
        reference.get("answer"),
        reference.get("intent"),
        reference.get("same_speaker"),
        reference.get("should_execute"),
    ) is not None:
        return "classification"
    if first_present(reference.get("turns"), target.get("turns")):
        return "dialogue"
    return "text"


def infer_reference_text(record: Dict[str, Any]) -> str:
    reference = record.get("reference") or {}
    target = record.get("target") or {}
    value = first_present(
        reference.get("text"),
        reference.get("answer"),
        reference.get("label"),
        reference.get("caption"),
        reference.get("summary"),
        reference.get("transcript"),
        reference.get("same_speaker"),
        reference.get("should_execute"),
        target.get("reference_text"),
        target.get("answer"),
    )
    if value is None:
        acceptable = first_present(reference.get("acceptable_answers"), target.get("acceptable_answers"))
        values = as_list(acceptable)
        value = values[0] if values else ""
    return "" if value is None else str(value)


def infer_audio_path(record: Dict[str, Any]) -> str:
    input_obj = record.get("input") or {}
    audios = input_obj.get("audios") or []
    value = first_present(
        record.get("WavPath"),
        record.get("WavPath1"),
        input_obj.get("audio_path"),
        audios[0].get("uri") if audios and isinstance(audios[0], dict) else None,
    )
    return "" if value is None else str(value)


def infer_second_audio_path(record: Dict[str, Any]) -> str:
    input_obj = record.get("input") or {}
    audios = input_obj.get("audios") or []
    value = first_present(
        record.get("WavPath2"),
        input_obj.get("audio_path_b"),
        audios[1].get("uri") if len(audios) > 1 and isinstance(audios[1], dict) else None,
    )
    return "" if value is None else str(value)


def normalize_condition(condition: Any) -> Dict[str, Any]:
    if not isinstance(condition, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in condition.items():
        if value not in (None, "", []):
            normalized[str(key)] = compact(value)
    return normalized


def normalize_record(
    record: Dict[str, Any],
    ref_col: str,
    default_task: str = "",
    default_dataset_id: str = "",
    run_context: Dict[str, Any] = None,
) -> Dict[str, Any]:
    schema_version = str(record.get("schema_version") or "")
    if schema_version == "2.0":
        from mesh_eval.core.schema_v2 import to_runtime_sample

        mode = (
            "formal"
            if record.get("use_bucket") == "formal_subscores"
            else "diagnostic"
        )
        return to_runtime_sample(
            record,
            ref_col=ref_col,
            protocol_mode=mode,
            run_context=run_context,
        )
    if schema_version == "runtime-sample/2.0":
        return dict(record)
    doc = dict(record)
    tags = doc.get("tags") or {}
    metadata = doc.get("metadata") or {}
    source = doc.get("source") or {}

    dataset_id = first_present(
        doc.get("dataset_id"),
        doc.get("dataset"),
        source.get("benchmark_id") if isinstance(source, dict) else None,
        default_dataset_id,
    )
    if dataset_id:
        doc["dataset_id"] = str(dataset_id)
        doc.setdefault("dataset", str(dataset_id))

    task = first_present(doc.get("task"), first_tag(tags, "task_tags"), default_task)
    capability = first_present(doc.get("capability"), first_tag(tags, "capability_tags"))
    scenario = first_present(doc.get("scenario"), first_tag(tags, "scenario_tags"))
    if task:
        doc["task"] = strip_prefix(task)
    if capability:
        doc["capability"] = strip_prefix(capability)
    if scenario:
        doc["scenario"] = strip_prefix(scenario)

    condition = normalize_condition(doc.get("condition") or {})
    for group in CONDITION_GROUPS:
        value = first_present(condition.get(group), metadata.get(group))
        if value not in (None, "", []):
            condition[group] = compact(value)
            doc[f"condition__{group}"] = compact(value)
    if condition:
        doc["condition"] = condition

    metrics = normalize_metrics(doc.get("metrics") or metadata.get("metrics"))
    if not metrics:
        metrics = default_metrics(doc.get("task", ""), doc.get("capability", ""))
    doc["metrics"] = metrics

    doc["answer_type"] = infer_answer_type(doc)
    language = first_present(doc.get("language"), metadata.get("language"), doc.get("lang"), "")
    doc["language"] = "" if language is None else str(language)
    doc["use_bucket"] = str(first_present(doc.get("use_bucket"), "diagnostic_evidence"))
    raw_split = first_present(
        doc.get("split"),
        source.get("original_split") if isinstance(source, dict) else None,
        "",
    )
    normalized_split = normalize_split(raw_split)
    if raw_split and normalized_split != strip_prefix(raw_split).lower().strip():
        doc.setdefault("source_split", str(raw_split))
    doc["split"] = normalized_split

    wav_path = infer_audio_path(doc)
    if wav_path and "WavPath" not in doc:
        doc["WavPath"] = wav_path
    second_wav_path = infer_second_audio_path(doc)
    if second_wav_path and "WavPath2" not in doc:
        doc["WavPath2"] = second_wav_path
    if ref_col not in doc:
        doc[ref_col] = infer_reference_text(doc)

    if "sample_id" not in doc:
        original_id = source.get("original_id") if isinstance(source, dict) else None
        doc["sample_id"] = str(first_present(original_id, f"{doc.get('dataset_id', 'sample')}-unknown"))

    # UltraEval passes doc as **kwargs into EvalTask._eval(..., reference, **doc).
    # Keep the structured reference available without colliding with that argument.
    if isinstance(doc.get("reference"), dict):
        doc["reference_obj"] = doc.pop("reference")

    return doc


def default_metrics(task: str, capability: str) -> List[str]:
    del task
    defaults = load_taxonomy().get("default_metrics") or {}
    return normalize_metrics(defaults.get(strip_prefix(capability), []))


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or value == []


def _audio_paths(doc: Dict[str, Any]) -> List[str]:
    paths = []
    for key, value in doc.items():
        if (key == "WavPath" or (key.startswith("WavPath") and key[7:].isdigit())) and value:
            paths.append(str(value))
    input_obj = doc.get("input") or {}
    for key in ("audio_path", "audio_path_b"):
        if input_obj.get(key):
            paths.append(str(input_obj[key]))
    for item in input_obj.get("audios") or []:
        if isinstance(item, dict) and item.get("uri"):
            paths.append(str(item["uri"]))
    return list(dict.fromkeys(paths))


def validate_record(
    doc: Dict[str, Any],
    ref_col: str,
    check_audio_exists: bool = False,
    cwd: str = "",
) -> List[str]:
    schema_version = str(doc.get("schema_version") or "")
    if schema_version in {"2.0", "runtime-sample/2.0"}:
        from mesh_eval.core.schema_v2 import validate_canonical_sample

        sample = doc.get("canonical") if schema_version.startswith("runtime-") else doc
        formal = sample.get("use_bucket") == "formal_subscores"
        return validate_canonical_sample(
            sample,
            formal=formal,
            check_audio_exists=check_audio_exists,
            cwd=cwd,
        )
    errors = []
    taxonomy = load_taxonomy()
    for key in (
        "sample_id",
        "dataset_id",
        "task",
        "capability",
        "scenario",
        "condition",
        "answer_type",
        "split",
    ):
        if _is_missing(doc.get(key)):
            errors.append(f"missing required field: {key}")
    metrics = normalize_metrics(doc.get("metrics"))
    if not metrics:
        errors.append("missing required field: metrics")
    unknown_metrics = [metric for metric in metrics if metric not in metric_catalog_names()]
    for metric in unknown_metrics:
        errors.append(f"unknown metric: {metric}")

    task = str(doc.get("task") or "")
    capability = str(doc.get("capability") or "")
    if task and task not in (taxonomy.get("tasks") or []):
        errors.append(f"unknown task: {task}")
    valid_capabilities = (taxonomy.get("capabilities") or {}).get(task, [])
    if capability and capability not in valid_capabilities:
        errors.append(f"capability {capability} does not belong to task {task}")
    scenario = str(doc.get("scenario") or "")
    if scenario and scenario not in (taxonomy.get("scenarios") or []):
        errors.append(f"unknown scenario: {scenario}")
    split = str(doc.get("split") or "")
    if split and split not in (taxonomy.get("splits") or []):
        errors.append(f"unknown split: {split}")
    if doc.get("use_bucket") and doc["use_bucket"] not in (taxonomy.get("use_buckets") or []):
        errors.append(f"unknown use_bucket: {doc['use_bucket']}")

    condition = doc.get("condition") or {}
    if condition and not isinstance(condition, dict):
        errors.append("condition must be an object")
        condition = {}
    condition_taxonomy = taxonomy.get("conditions") or {}
    for group, value in condition.items():
        if group not in condition_taxonomy:
            errors.append(f"unknown condition group: {group}")
            continue
        values = str(value).split("|") if not isinstance(value, list) else value
        for item in values:
            if str(item) not in condition_taxonomy[group]:
                errors.append(f"unknown condition {group}: {item}")
    required_conditions = (taxonomy.get("required_conditions") or {}).get(
        capability,
        (taxonomy.get("required_conditions") or {}).get("default", []),
    )
    if doc.get("use_bucket") == "formal_subscores":
        for group in required_conditions:
            if _is_missing(condition.get(group)):
                errors.append(f"missing required condition for {capability}: {group}")

    reference = doc.get("reference_obj") or doc.get("reference") or {}
    if not isinstance(reference, dict):
        reference = {}
    for field in (taxonomy.get("required_reference") or {}).get(capability, []):
        if _is_missing(reference.get(field)):
            errors.append(f"capability {capability} requires reference.{field}")
    if doc.get(ref_col) in (None, "") and doc.get("answer_type") not in {
        "audio",
        "structured_json",
        "action",
        "dialogue",
    }:
        errors.append(f"missing reference column: {ref_col}")

    audio_paths = _audio_paths(doc)
    if not audio_paths:
        errors.append("missing audio input")
    if check_audio_exists:
        for wav_path in audio_paths:
            full_path = wav_path if os.path.isabs(wav_path) else os.path.join(cwd, wav_path)
            if not os.path.isfile(full_path):
                errors.append(f"audio path not found: {wav_path}")
    return errors


def flatten_fields(doc: Dict[str, Any], fields: Iterable[str]) -> Dict[str, Any]:
    result = {}
    for field in fields:
        value = get_path(doc, field) if "." in field else doc.get(field)
        if value not in (None, "", []):
            result[field.replace(".", "__")] = compact(value)
    return result
