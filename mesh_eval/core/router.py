import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from audio_evals.evaluator.base import Evaluator
from mesh_eval.core.schema import as_list, compact, flatten_fields, strip_prefix
from mesh_eval.core.protocol import validate_protocol_output


def _first_present(doc: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for key in keys:
        value = doc.get(key)
        if value not in (None, "", []):
            return value
    return None


def _tag_values(doc: Dict[str, Any], key: str) -> List[str]:
    tags = doc.get("tags") or {}
    values = doc.get(key) or tags.get(key)
    return [str(item) for item in as_list(values)]


def _route_candidates(doc: Dict[str, Any]) -> List[str]:
    raw_candidates = []
    direct_task = _first_present(doc, ["task", "task_name", "eval_task"])
    if direct_task:
        raw_candidates.append(str(direct_task))
    raw_candidates.extend(_tag_values(doc, "task_tags"))
    raw_candidates.extend(_tag_values(doc, "capability_tags"))

    task = strip_prefix(direct_task) if direct_task else ""
    capability = strip_prefix(doc.get("capability", ""))
    answer_type = strip_prefix(doc.get("answer_type", ""))
    language = strip_prefix(doc.get("language", ""))

    candidates: List[str] = []
    for parts in (
        (task, capability, answer_type, language),
        (task, capability, answer_type),
        (task, capability, language),
        (task, capability),
        (task, language),
        (capability, answer_type, language),
        (capability, answer_type),
        (capability, language),
        (capability,),
        (task,),
    ):
        clean = [part for part in parts if part]
        if clean:
            candidates.append(":".join(clean))
            if len(clean) == 2:
                candidates.append("/".join(clean))

    for item in raw_candidates:
        candidates.append(item)
        stripped = strip_prefix(item)
        candidates.append(stripped)
        if ":" not in item:
            candidates.append(f"task:{item}")

    return list(dict.fromkeys(candidates))


def _resolve_route(doc: Dict[str, Any], routes: Dict[str, str], default: str) -> str:
    for candidate in _route_candidates(doc):
        if candidate in routes:
            return routes[candidate]
    if default:
        return default
    raise KeyError(f"No route for candidates: {_route_candidates(doc)}")


def _as_names(value: Any) -> List[str]:
    return [str(item) for item in as_list(value)]


def _declared_names(value: Any) -> List[str]:
    if isinstance(value, dict):
        names: List[str] = []
        for item in value.values():
            names.extend(_declared_names(item))
        return names
    return _as_names(value)


def _conditional_route(value: Any, doc: Dict[str, Any]) -> Any:
    if not isinstance(value, dict):
        return value
    candidates = [
        *_route_candidates(doc),
        strip_prefix(doc.get("language", "")),
        strip_prefix(doc.get("answer_type", "")),
        "default",
    ]
    for candidate in candidates:
        if candidate and candidate in value:
            return value[candidate]
    return []


def _ensure_global_registry_paths(paths: Optional[List[str]]) -> None:
    if not paths:
        return
    from audio_evals.registry import registry

    existing = {Path(path).resolve() for path in registry._registry_paths}
    additions = [Path(path) for path in paths if Path(path).resolve() not in existing]
    if not additions:
        return
    registry.add_registry_paths(additions)
    for resource in ("_model", "_dataset", "_evaluator", "_prompt", "_eval_task", "_agg", "_process"):
        registry.__dict__.pop(resource, None)


class RouterPrompt:
    def __init__(
        self,
        routes: Dict[str, str],
        default_prompt: str = "",
        registry_paths: Optional[List[str]] = None,
        formal: bool = False,
    ):
        _ensure_global_registry_paths(registry_paths)
        from audio_evals.registry import registry

        self.routes = routes
        self.default_prompt = default_prompt
        self.formal = formal
        prompt_names = set()
        for value in routes.values():
            prompt_names.update(_as_names(value))
        if default_prompt:
            prompt_names.add(default_prompt)
        self.prompts = {name: registry.get_prompt(name) for name in prompt_names}
        self._prompt_lock = threading.Lock()

    def load(self, **doc):
        resolved = doc.get("resolved_protocol") or {}
        prompt_name = doc.get("prompt_name") or resolved.get("prompt")
        if not prompt_name:
            formal = self.formal or (
                doc.get("schema_version") == "runtime-sample/2.0"
                and doc.get("use_bucket") == "formal_subscores"
            )
            prompt_name = _resolve_route(
                doc, self.routes, "" if formal else self.default_prompt
            )
        if isinstance(prompt_name, list):
            prompt_name = prompt_name[0]
        return self._get_prompt(str(prompt_name)).load(**doc)

    def _get_prompt(self, name: str):
        prompt = self.prompts.get(name)
        if prompt is not None:
            return prompt
        with self._prompt_lock:
            prompt = self.prompts.get(name)
            if prompt is None:
                from audio_evals.registry import registry

                prompt = registry.get_prompt(name)
                if prompt is None:
                    raise KeyError(f"Prompt is not registered: {name}")
                self.prompts[name] = prompt
        return prompt


class RouterEvaluator(Evaluator):
    DEFAULT_INJECT_FIELDS = [
        "sample_id",
        "benchmark_id",
        "source_dataset_id",
        "dataset_id",
        "dataset",
        "task",
        "capability",
        "answer_type",
        "language",
        "scenario",
        "scenario.primary",
        "scenario.secondary",
        "scenario__primary",
        "scenario__secondary",
        "use_bucket",
        "split",
        "metrics",
        "evaluators",
        "condition.acoustic",
        "condition.spatial",
        "condition.speaker",
        "condition.device",
        "condition.interaction",
        "condition__acoustic",
        "condition__spatial",
        "condition__speaker",
        "condition__device",
        "condition__interaction",
        "tags.source_tags",
        "tags.scenario_tags",
        "tags.task_tags",
        "tags.capability_tags",
        "tags.model_part_tags",
        "tags.data_property_tags",
        "metadata.language",
        "metadata.domain",
        "metadata.difficulty",
        "metadata.device_type",
        "metadata.device_type_subtype",
        "metadata.recording_setup",
        "metadata.recording_setup_subtype",
        "metadata.noise_type",
        "metadata.noise_type_subtype",
        "metadata.snr_bucket",
        "metadata.snr_bucket_subtype",
        "metadata.speaker_count",
        "metadata.speaker_count_bucket",
        "metadata.overlap_level",
        "metadata.overlap_level_subtype",
        "metadata.distance",
        "metadata.distance_subtype",
        "metadata.motion_state",
        "metadata.motion_state_subtype",
        "metadata.authorization_state",
        "metadata.authorization_state_subtype",
        "metadata.window_state",
        "metadata.window_state_subtype",
        "metadata.hvac_state",
        "metadata.hvac_state_subtype",
        "metadata.media_state",
        "metadata.media_state_subtype",
        "metadata.scenario_subtype",
        "subset_manifest_record_id",
        "subset_manifest_schema_version",
        "subset_manifest_sha256",
        "source_benchmark_id",
        "subset_id",
        "source_protocol_id",
        "source_metric_names",
        "mapping_status",
        "resource_status",
        "annotation_confidence",
        "protocol_hash",
        "measurement_protocol",
        "measurement_protocol_hash",
        "edge.should_respond",
        "edge.authorized",
        "edge.speaker_role",
        "edge.target_speaker",
    ]

    def __init__(
        self,
        routes: Dict[str, Any],
        metric_routes: Optional[Dict[str, Any]] = None,
        default_evaluator: str = "",
        inject_fields: Optional[List[str]] = None,
        registry_paths: Optional[List[str]] = None,
        formal: bool = False,
    ):
        _ensure_global_registry_paths(registry_paths)
        self.routes = routes
        self.metric_routes = metric_routes or {}
        self.default_evaluator = default_evaluator
        self.inject_fields = inject_fields or self.DEFAULT_INJECT_FIELDS
        self.formal = formal
        evaluator_names = set()
        for value in routes.values():
            evaluator_names.update(_declared_names(value))
        for value in self.metric_routes.values():
            evaluator_names.update(_declared_names(value))
        if default_evaluator:
            evaluator_names.add(default_evaluator)
        self._known_evaluator_names = evaluator_names
        self.evaluators: Dict[str, Evaluator] = {}
        self._evaluator_lock = threading.Lock()

    def _eval(self, pred, label, **kwargs):
        raise NotImplementedError("RouterEvaluator dispatches through __call__.")

    def __call__(self, pred, ref, **doc):
        formal = self.formal or (
            doc.get("schema_version") == "runtime-sample/2.0"
            and doc.get("use_bucket") == "formal_subscores"
        )
        resolved = doc.get("resolved_protocol") or {}
        if formal:
            if not resolved:
                raise ValueError("formal V2 evaluation requires resolved_protocol")
            validate_protocol_output(pred, resolved)
            missing = self._unmapped_metric_names(doc)
            if missing:
                raise ValueError(
                    "formal metrics have no evaluator mapping: " + ", ".join(missing)
                )
        evaluator_names = self._resolve_evaluator_names(doc)
        if formal and not evaluator_names:
            raise ValueError("formal protocol resolved no evaluators")
        score: Dict[str, Any] = {"pred": pred, "ref": ref}
        for evaluator_name in evaluator_names:
            part = self._get_evaluator(evaluator_name)(pred, ref, **doc)
            for key, value in part.items():
                if key in {"pred", "ref"}:
                    score[key] = value
                elif key not in score:
                    score[key] = value
                elif score[key] != value:
                    score[f"{evaluator_name}__{key}"] = value

        score["mesh_task"] = doc.get("task", "")
        score["mesh_capability"] = doc.get("capability", "")
        score["mesh_route"] = "|".join(_route_candidates(doc))
        score["mesh_evaluator"] = "|".join(evaluator_names)
        score["metric_names"] = compact(doc.get("metrics", []))
        score["mesh_unmapped_metrics"] = "|".join(self._unmapped_metric_names(doc))
        score.update(flatten_fields(doc, self.inject_fields))
        for key, value in doc.items():
            if key.startswith(("metadata__", "scenario__")) and value not in (
                None,
                "",
                [],
            ):
                score[key] = compact(value)
        if doc.get("schema_version"):
            score["sample_schema_version"] = doc["schema_version"]
        if doc.get("provenance"):
            score["_slice_provenance"] = doc["provenance"]
        return score

    def _get_evaluator(self, name: str) -> Evaluator:
        evaluator = self.evaluators.get(name)
        if evaluator is not None:
            return evaluator
        with self._evaluator_lock:
            evaluator = self.evaluators.get(name)
            if evaluator is None:
                from audio_evals.registry import registry

                evaluator = registry.get_evaluator(name)
                if evaluator is None:
                    raise KeyError(f"Evaluator is not registered: {name}")
                self.evaluators[name] = evaluator
        return evaluator

    def _resolve_evaluator_names(self, doc: Dict[str, Any]) -> List[str]:
        resolved = doc.get("resolved_protocol") or {}
        if resolved.get("evaluators"):
            return list(dict.fromkeys(_as_names(resolved["evaluators"])))
        direct = _first_present(doc, ["evaluators", "evaluator"])
        if direct:
            return list(dict.fromkeys(_as_names(direct)))
        metric_names = [strip_prefix(item) for item in as_list(doc.get("metrics"))]
        evaluator_names: List[str] = []
        for metric_name in metric_names:
            route = self.metric_routes.get(metric_name)
            if route:
                evaluator_names.extend(_as_names(_conditional_route(route, doc)))
        if not evaluator_names:
            evaluator_names.extend(_as_names(_resolve_route(doc, self.routes, self.default_evaluator)))
        if not evaluator_names and self.default_evaluator:
            evaluator_names.append(self.default_evaluator)
        return list(dict.fromkeys(evaluator_names))

    def _unmapped_metric_names(self, doc: Dict[str, Any]) -> List[str]:
        resolved = doc.get("resolved_protocol") or {}
        if resolved:
            resolved_metrics = {str(item) for item in resolved.get("metrics") or []}
            declared_metrics = {
                strip_prefix(item) for item in as_list(doc.get("metrics"))
            }
            if declared_metrics.issubset(resolved_metrics) and resolved.get("evaluators"):
                return []
        missing = []
        for metric in [strip_prefix(item) for item in as_list(doc.get("metrics"))]:
            route = self.metric_routes.get(metric)
            if not route or not _as_names(_conditional_route(route, doc)):
                missing.append(metric)
        return missing
