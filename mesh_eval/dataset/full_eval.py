import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from audio_evals.config import get_environment
from audio_evals.dataset.dataset import Dataset
from mesh_eval.core.schema import validate_record
from mesh_eval.core.schema_v2 import (
    adapt_v1_to_v2,
    is_canonical_v2,
    to_runtime_sample,
)


DEFAULT_FULL_EVAL_ROOT = Path(
    get_environment("VOXMATRIX_FULL_EVAL_ROOT", "data/full_eval")
).expanduser()

ZH_DATASETS = {
    "aishell_1",
    "aishell_4",
    "aishell_5",
    "alimeeting",
    "common_voice",
    "magicdata_ramc",
    "misp",
}

# These manifests are generated from the benchmark-native annotations rather
# than from the older HF registry.  Keeping the ids here lets the same
# FullEvalDataset adapter handle both the historical ASR manifests and the
# newly connected heterogeneous understanding benchmarks.
ZH_EXTERNAL_DATASETS = {
    "vocalbench_zh",
}

EXTERNAL_DATASETS = {
    "mmsu",
    "vocalbench",
    "vocalbench_zh",
    "uro_bench",
}

ESC50_LABELS = [
    "airplane",
    "breathing",
    "brushing_teeth",
    "can_opening",
    "car_horn",
    "cat",
    "chainsaw",
    "chirping_birds",
    "church_bells",
    "clapping",
    "clock_alarm",
    "clock_tick",
    "coughing",
    "cow",
    "crackling_fire",
    "crickets",
    "crow",
    "crying_baby",
    "dog",
    "door_wood_creaks",
    "door_wood_knock",
    "drinking_sipping",
    "engine",
    "fireworks",
    "footsteps",
    "frog",
    "glass_breaking",
    "hand_saw",
    "helicopter",
    "hen",
    "insects",
    "keyboard_typing",
    "laughing",
    "mouse_click",
    "pig",
    "pouring_water",
    "rain",
    "rooster",
    "sea_waves",
    "sheep",
    "siren",
    "sneezing",
    "snoring",
    "thunderstorm",
    "toilet_flush",
    "train",
    "vacuum_cleaner",
    "washing_machine",
    "water_drops",
    "wind",
]


class FullEvalDataset(Dataset):
    def __init__(
        self,
        dataset: str,
        benchmark_id: str,
        view: str = "native",
        root: str = "",
        default_task: str = "mesh-unified-speech",
        ref_col: str = "text",
        strict: bool = True,
        check_audio_exists: bool = True,
    ):
        super().__init__(default_task=default_task, ref_col=ref_col)
        configured_root = os.environ.get("MESH_FULL_EVAL_ROOT") or root
        self.root = Path(configured_root) if configured_root else DEFAULT_FULL_EVAL_ROOT
        self.dataset_id = dataset
        self.benchmark_id = benchmark_id
        self.view = view
        self.strict = strict
        self.check_audio_exists = check_audio_exists
        self._caption_references_by_path: Optional[Dict[str, List[str]]] = None

    @property
    def manifest_path(self) -> Path:
        return self.root / self.dataset_id / "final" / "manifest.jsonl"

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(offset=0, limit=limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        if self.view in {"alimeeting_diarization", "alimeeting_attribution"}:
            rows = self._load_alimeeting_groups(limit, offset=offset)
        else:
            rows = self._load_stream(limit, offset=offset)
        return rows

    def count(self, limit=0) -> int:
        if self.view in {"alimeeting_diarization", "alimeeting_attribution"}:
            count = 0
            for _line_no, _group in self._iter_alimeeting_asr_groups():
                count += 1
                if limit > 0 and count >= limit:
                    break
            return count

        count = 0
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                if self._transform(json.loads(line)) is None:
                    continue
                count += 1
                if limit > 0 and count >= limit:
                    break
        return count

    def _load_stream(self, limit: int, offset: int = 0) -> List[Dict[str, Any]]:
        records = []
        seen = 0
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                transformed = self._transform(json.loads(line))
                if transformed is None:
                    continue
                if seen < offset:
                    seen += 1
                    continue
                records.append(self._normalize(transformed, line_no))
                if limit > 0 and len(records) >= limit:
                    break
        return records

    def _load_alimeeting_groups(
        self, limit: int, offset: int = 0
    ) -> List[Dict[str, Any]]:
        records = []
        seen = 0
        for line_no, group in self._iter_alimeeting_asr_groups():
            if seen < offset:
                seen += 1
                continue
            transformed = self._transform_alimeeting_group(group)
            records.append(self._normalize(transformed, line_no))
            if limit > 0 and len(records) >= limit:
                break
        return records

    def _iter_alimeeting_asr_groups(self) -> Iterator[tuple[int, List[Dict[str, Any]]]]:
        current_audio = ""
        current_group: List[Dict[str, Any]] = []
        current_line = 0
        with self.manifest_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "text" not in (row.get("reference") or {}):
                    continue
                audio_path = self._primary_audio_uri(row)
                if current_group and audio_path != current_audio:
                    yield current_line, current_group
                    current_group = []
                if not current_group:
                    current_audio = audio_path
                    current_line = line_no
                current_group.append(row)
        if current_group:
            yield current_line, current_group

    def _transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = deepcopy(record)
        if not self._view_matches(row):
            return None
        if self.dataset_id == "alimeeting":
            if "text" not in (row.get("reference") or {}):
                return None
        if self.view == "covost_zh_en":
            if self._covost_source_language(row) != "zh":
                return None
            self._set_translation_protocol(row, "zh", "en")
        elif self.view == "covost_en_zh":
            if self._covost_source_language(row) != "en":
                return None
            self._set_translation_protocol(row, "en", "zh")
        elif self.view == "tool_call_proxy":
            self._set_tool_call_proxy(row)
        elif self.view == "target_speaker_proxy":
            self._set_target_speaker_proxy(row)
        else:
            self._set_native_protocol(row)
        return row

    def _view_matches(self, row: Dict[str, Any]) -> bool:
        """Apply lightweight, deterministic views to heterogeneous manifests.

        A view is intentionally expressed as a string (for example
        ``capability_qa`` or ``non_speech``) so it can be selected directly in
        the registry YAML without introducing a second benchmark-specific
        dataset class.  The default ``native`` view keeps every row.
        """
        view = str(self.view or "native").strip().lower()
        if view in {"", "native", "all"}:
            return True
        task = str(row.get("task") or "").strip().lower()
        capability = str(row.get("capability") or "").strip().lower()
        if view.startswith("capability_"):
            return capability == view[len("capability_") :]
        if view.startswith("task_"):
            return task == view[len("task_") :]
        if view == "non_speech":
            return task != "speech_output" and capability not in {
                "emotion_control",
                "style_following",
            }
        if view == "speech_understanding":
            return task == "speech_understanding"
        if view == "agent":
            return task == "agent"
        return True

    def _set_native_protocol(self, row: Dict[str, Any]) -> None:
        task = row.get("task")
        capability = row.get("capability")
        language = self._language(row)
        row["language"] = language
        if capability == "asr":
            self._bind_protocol(row, "speech_recognition_transcription", "asr")
            input_obj = row.get("input") or {}
            is_segment = (
                input_obj.get("start_time") is not None
                and input_obj.get("end_time") is not None
            )
            prefix = "mesh-segment-asr" if is_segment else "qwen3-omni-asr"
            row["prompt_name"] = f"{prefix}-{language}"
            row["metrics"] = ["cer" if language == "zh" else "wer"]
            row["evaluators"] = ["cer" if language == "zh" else "wer"]
        elif capability == "audio_caption":
            self._bind_protocol(
                row, "audio_speech_understanding", "audio_caption"
            )
            reference = row.setdefault("reference", {})
            audio_path = self._primary_audio_uri(row)
            references = self._caption_reference_map().get(audio_path)
            if references:
                reference["captions"] = references
            row["prompt_name"] = "qwen3-omni-audio-caption"
            row["evaluators"] = ["coco"]
            row["metrics"] = ["cider", "spice"]
            row["answer_type"] = "text"
        elif capability == "sound_event" and self.dataset_id == "esc50":
            self._bind_protocol(row, "audio_speech_understanding", "sound_event")
            row["prompt_name"] = "mesh-classification"
            row["evaluators"] = ["mesh-classification"]
            row["answer_type"] = "classification"
            row["question"] = (
                "Return exactly one ESC-50 label from this list: "
                + ", ".join(ESC50_LABELS)
            )
        elif (
            task == "agent"
            and capability == "instruction_following"
            and self.dataset_id not in EXTERNAL_DATASETS
        ):
            self._bind_protocol(
                row, "spoken_agentic_interaction", "instruction_following"
            )
            reference = row.setdefault("reference", {})
            reference.setdefault("slots", {})
            row["prompt_name"] = "mesh-intent-slots"
            row["evaluators"] = ["mesh-intent", "mesh-slot-f1"]
            row["answer_type"] = "structured_json"
        elif (
            task == "agent"
            and capability == "qa"
            and self.dataset_id not in EXTERNAL_DATASETS
        ):
            self._bind_protocol(row, "audio_speech_understanding", "qa")
            row["prompt_name"] = "mesh-open-qa"
            row["evaluators"] = ["em", "mesh-text-f1"]
            row["answer_type"] = "text"
        elif capability == "speaker_verification":
            self._bind_protocol(
                row, "speaker_attribution", "speaker_verification"
            )
            row["prompt_name"] = "mesh-speaker-verification"
            row["evaluators"] = ["mesh-speaker-verification"]
            row["metrics"] = ["eer", "auc"]
            row["answer_type"] = "classification"
        elif capability == "translation_instruction":
            self._bind_protocol(row, "spoken_agentic_interaction", "translation_instruction")
            reference = row.setdefault("reference", {})
            if "text" not in reference and reference.get("answer") is not None:
                reference["text"] = reference["answer"]
            row["prompt_name"] = "mesh-external-instruction"
            row["evaluators"] = ["mesh-text-f1"]
            row["metrics"] = ["text_f1"]
            row["answer_type"] = "text"
        elif capability in {"qa", "audio_reasoning", "meeting_qa"}:
            # Open-ended spoken QA and reasoning samples occur in MMSU,
            # VocalBench, VocalBench-zh and URO-Bench.  They share the same
            # audio-question contract even when the legacy manifest uses
            # different task/capability names.
            protocol_capability = "qa" if capability in {"qa", "meeting_qa"} else "audio_reasoning"
            protocol_task = "audio_speech_understanding"
            self._bind_protocol(row, protocol_task, protocol_capability)
            row["prompt_name"] = "mesh-external-qa"
            row["evaluators"] = ["em", "mesh-text-f1"]
            row["metrics"] = ["exact_match", "text_f1"]
            row["answer_type"] = "text"
        elif capability == "speech_translation":
            reference = row.setdefault("reference", {})
            if "text" not in reference and reference.get("answer") is not None:
                reference["text"] = reference["answer"]
            self._bind_protocol(
                row, "speech_recognition_transcription", "speech_translation"
            )
            row["prompt_name"] = "mesh-external-qa"
            row["evaluators"] = ["mesh-classification"]
            row["metrics"] = ["accuracy"]
            row["answer_type"] = "classification"
        elif capability in {
            "paralinguistic",
            "paralinguistic_recognition",
            "acoustic_scene",
            "sound_event",
        }:
            if capability == "paralinguistic":
                capability = "paralinguistic_recognition"
            self._bind_protocol(row, "audio_speech_understanding", capability)
            row["prompt_name"] = "mesh-external-classification"
            row["evaluators"] = ["mesh-classification"]
            row["metrics"] = ["accuracy"]
            row["answer_type"] = "classification"
        elif capability == "speaker_count":
            self._bind_protocol(row, "speaker_attribution", "speaker_count")
            row["prompt_name"] = "mesh-external-classification"
            row["evaluators"] = ["mesh-speaker-count"]
            row["metrics"] = ["speaker_count_acc"]
            row["answer_type"] = "classification"
        elif capability == "spoof_detection":
            # MMSU contains a small real/fake diagnostic slice.  It is not an
            # ASVspoof LA/DF trial set, so use accuracy here and retain the
            # diagnostic bucket instead of pretending to compute EER.
            self._bind_protocol(row, "speaker_attribution", "spoof_detection")
            row["prompt_name"] = "mesh-external-classification"
            row["evaluators"] = ["mesh-classification"]
            row["metrics"] = ["accuracy"]
            row["answer_type"] = "classification"
        elif capability in {"instruction_following", "tool_call"}:
            reference = row.setdefault("reference", {})
            if "text" not in reference and reference.get("answer") is not None:
                reference["text"] = reference["answer"]
            self._bind_protocol(
                row, "spoken_agentic_interaction", "instruction_following"
            )
            row["prompt_name"] = "mesh-external-instruction"
            row["evaluators"] = ["mesh-text-f1"]
            row["metrics"] = ["text_f1"]
            row["answer_type"] = "text"
        elif capability in {
            "multi_turn_dialogue",
            "clarification",
            "interruption",
        }:
            reference = row.setdefault("reference", {})
            if "turns" not in reference:
                reference["turns"] = [
                    {"role": "assistant", "content": reference.get("answer", "")}
                ]
            protocol_capability = capability
            self._bind_protocol(
                row, "spoken_agentic_interaction", protocol_capability
            )
            row["prompt_name"] = "mesh-external-qa"
            row["evaluators"] = ["mesh-text-f1"]
            row["metrics"] = ["text_f1"]
            row["answer_type"] = "text"
        elif capability == "meeting_summary":
            reference = row.setdefault("reference", {})
            if "summary" not in reference and reference.get("answer") is not None:
                reference["summary"] = reference["answer"]
            self._bind_protocol(
                row, "audio_speech_understanding", "meeting_summary"
            )
            row["prompt_name"] = "mesh-meeting-summary"
            row["evaluators"] = ["mesh-rouge-l"]
            row["metrics"] = ["rouge_l"]
            row["answer_type"] = "text"
        elif task == "speech_output":
            self._bind_protocol(
                row, "speech_generation_delivery", "speech_generation"
            )
            row["prompt_name"] = "qwen3-omni-tts-en"
            row["evaluators"] = ["seed-tts-eval-asr-wer-en"]
            row["answer_type"] = "audio"

    def _set_translation_protocol(
        self, row: Dict[str, Any], source_language: str, target_language: str
    ) -> None:
        self._bind_protocol(
            row, "speech_recognition_transcription", "speech_translation"
        )
        row["language"] = source_language
        row["source_language"] = source_language
        row["target_language"] = target_language
        row["translation_direction"] = f"{source_language}2{target_language}"
        row["prompt_name"] = f"qwen3-omni-s2tt-{source_language}2{target_language}"
        row["metrics"] = ["bleu" if target_language == "en" else "bleu_zh", "chrf"]
        row["evaluators"] = [
            "bleu" if target_language == "en" else "bleu-zh",
            "mesh-chrf",
        ]
        row["answer_type"] = "text"

    def _set_tool_call_proxy(self, row: Dict[str, Any]) -> None:
        reference = row.setdefault("reference", {})
        intent = reference.get("intent")
        slots = reference.get("slots") or {}
        reference["expected_output"] = {"tool": intent, "arguments": slots}
        self._bind_protocol(row, "spoken_agentic_interaction", "tool_call")
        row["metrics"] = [
            "tool_acc",
            "parameter_acc",
            "parameter_f1",
            "task_success",
        ]
        row["evaluators"] = ["mesh-tool-call"]
        row["prompt_name"] = "mesh-tool-call"
        row["answer_type"] = "structured_json"
        row["use_bucket"] = "diagnostic_evidence"
        row["sample_id"] = f"{row.get('sample_id', self.dataset_id)}__tool_call_proxy"
        row["proxy_protocol"] = "intent_slots_as_tool_call"
        row["language"] = "en"

    def _set_target_speaker_proxy(self, row: Dict[str, Any]) -> None:
        reference = row.setdefault("reference", {})
        reference["should_execute"] = bool(reference.get("same_speaker"))
        condition = dict(row.get("condition") or {})
        condition["speaker"] = (
            "authorized_user" if reference["should_execute"] else "unauthorized_user"
        )
        row["condition"] = condition
        self._bind_protocol(row, "speaker_attribution", "target_speaker")
        row["metrics"] = ["target_speaker_acc", "mis_execution"]
        row["evaluators"] = ["mesh-target-speaker"]
        row["prompt_name"] = "mesh-target-speaker"
        row["answer_type"] = "classification"
        row["use_bucket"] = "diagnostic_evidence"
        row["sample_id"] = f"{row.get('sample_id', self.dataset_id)}__target_speaker_proxy"
        row["proxy_protocol"] = "same_speaker_as_should_execute"
        row["language"] = "en"

    def _transform_alimeeting_group(
        self, group: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        row = deepcopy(group[0])
        input_obj = row.get("input") or {}
        grouped_input = {
            key: input_obj[key]
            for key in ("audio_path", "audio_filepath")
            if input_obj.get(key)
        }
        if isinstance(input_obj.get("audio"), list):
            grouped_input["audio"] = deepcopy(input_obj["audio"])
        primary_audio_uri = self._primary_audio_uri(row)
        if primary_audio_uri and not (
            grouped_input.get("audio") or grouped_input.get("audio_path")
        ):
            grouped_input["audio_path"] = primary_audio_uri
        row["input"] = grouped_input
        audio_stem = Path(primary_audio_uri or "meeting").stem
        utterances = []
        segments = []
        seen_segments = set()
        for index, item in enumerate(group):
            item_input = item.get("input") or {}
            reference = item.get("reference") or {}
            start = item_input.get("start_time")
            end = item_input.get("end_time")
            speaker = item_input.get("speaker_id")
            segment_key = (start, end, speaker)
            if start is not None and end is not None and speaker and segment_key not in seen_segments:
                segments.append({"start": start, "end": end, "speaker": speaker})
                seen_segments.add(segment_key)
            utterances.append(
                {
                    "id": str(index),
                    "source_sample_id": item.get("sample_id") or f"{audio_stem}_{index:05d}",
                    "start": start,
                    "end": end,
                    "speaker": speaker,
                    "text": reference.get("text", ""),
                }
            )
        row["task"] = "speaker_attribution"
        row["language"] = "zh"
        row["answer_type"] = "structured_json"
        row["use_bucket"] = "diagnostic_evidence"
        row["reference_reconstruction"] = "grouped_from_asr_timestamps"
        if self.view == "alimeeting_diarization":
            self._bind_protocol(row, "speaker_attribution", "diarization")
            row["metrics"] = ["der"]
            row["evaluators"] = ["mesh-diarization"]
            row["prompt_name"] = "mesh-diarization"
            row["reference"] = {"segments": segments}
            row["sample_id"] = f"alimeeting_diar_{audio_stem}"
        else:
            self._bind_protocol(
                row, "speaker_attribution", "speaker_attribution"
            )
            row["metrics"] = ["cpcer", "attribution_acc"]
            row["evaluators"] = ["mesh-speaker-attribution"]
            row["prompt_name"] = "mesh-speaker-attribution"
            row["reference"] = {"utterances": utterances}
            row["sample_id"] = f"alimeeting_attr_{audio_stem}"
        return row

    def _normalize(self, record: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        source = deepcopy(record)
        view_annotations = {
            field: source.get(field)
            for field in (
                "proxy_protocol",
                "reference_reconstruction",
                "offline_proxy",
            )
            if source.get(field) not in (None, "", [])
        }
        route_hints = {
            target: source.pop(source_name)
            for source_name, target in (
                ("prompt_name", "prompt"),
                ("metrics", "metrics"),
                ("evaluators", "evaluators"),
                ("answer_type", "answer_type"),
            )
            if source.get(source_name) not in (None, "", [])
        }
        route_hints["source"] = "full_eval_view"
        if is_canonical_v2(source):
            canonical = source
            metadata = dict(canonical.get("metadata") or {})
            for field in (
                "language",
                "source_language",
                "target_language",
                "translation_direction",
            ):
                if canonical.get(field) not in (None, "", []):
                    metadata[field] = canonical.pop(field)
            canonical["metadata"] = metadata
            input_obj = dict(canonical.get("input") or {})
            if canonical.get("question") not in (None, "", []):
                input_obj["question"] = canonical.pop("question")
            canonical["input"] = input_obj
            canonical["capability_protocol"] = (
                f"{canonical['task']}/{canonical['capability']}@1"
            )
            hints = dict(canonical.get("evaluation_hints") or {})
            hints.update(route_hints)
            canonical["evaluation_hints"] = hints
            allowed = {
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
                "evaluation_hints",
                "legacy",
                "conversion_warnings",
                "asset_schema_version",
            }
            extras = {
                key: canonical.pop(key)
                for key in list(canonical)
                if key not in allowed
            }
            if extras:
                canonical.setdefault("legacy", {})["source_fields"] = extras
        else:
            source.update(
                {
                    "metrics": route_hints.get("metrics", []),
                    "evaluators": route_hints.get("evaluators", []),
                    "prompt_name": route_hints.get("prompt", ""),
                    "answer_type": route_hints.get("answer_type", ""),
                }
            )
            canonical = adapt_v1_to_v2(source, ref_col=self.ref_col)
            canonical["evaluation_hints"] = {
                **dict(canonical.get("evaluation_hints") or {}),
                **route_hints,
            }
        provenance = dict(canonical.get("provenance") or {})
        provenance["view"] = {
            "source": "verified_derived",
            "confidence": 1.0,
            "derivation_version": "full-eval-view@2",
            "evidence": self.view,
        }
        canonical["provenance"] = provenance
        legacy = dict(canonical.get("legacy") or {})
        legacy.update(
            {
                "full_eval_view": self.view,
                "source_manifest": str(self.manifest_path),
            }
        )
        canonical["legacy"] = legacy
        protocol_mode = (
            "formal"
            if canonical.get("use_bucket") == "formal_subscores"
            else "diagnostic"
        )
        doc = to_runtime_sample(
            canonical,
            ref_col=self.ref_col,
            protocol_mode=protocol_mode,
            run_context={
                "benchmark_id": self.benchmark_id,
                "source_dataset_id": self.dataset_id,
                "full_eval_view": self.view,
                "source_manifest": str(self.manifest_path),
            },
        )
        doc["benchmark_id"] = self.benchmark_id
        doc["source_dataset_id"] = self.dataset_id
        doc["full_eval_view"] = self.view
        doc["source_manifest"] = str(self.manifest_path)
        doc.update(view_annotations)
        errors = validate_record(
            doc,
            ref_col=self.ref_col,
            check_audio_exists=self.check_audio_exists,
        )
        if errors:
            doc["schema_errors"] = errors
            if self.strict:
                raise ValueError(
                    f"{self.manifest_path}:{line_no} view={self.view} "
                    f"failed schema validation: {errors}"
                )
        return doc

    @staticmethod
    def _bind_protocol(row: Dict[str, Any], task: str, capability: str) -> None:
        row["task"] = task
        row["capability"] = capability
        row["capability_protocol"] = f"{task}/{capability}@1"

    def _language(self, row: Dict[str, Any]) -> str:
        if self.dataset_id in ZH_DATASETS:
            return "zh"
        if self.dataset_id == "covost2":
            return self._covost_source_language(row)
        if self.dataset_id in ZH_EXTERNAL_DATASETS:
            return "zh"
        # URO-Bench and MMSU contain mixed languages.  Prefer explicit
        # metadata when present and fall back to a conservative text check.
        question = str((row.get("input") or {}).get("question") or "")
        if any("\u4e00" <= char <= "\u9fff" for char in question):
            return "zh"
        return "en"

    def _caption_reference_map(self) -> Dict[str, List[str]]:
        if self._caption_references_by_path is not None:
            return self._caption_references_by_path

        references: Dict[str, List[str]] = {}
        raw_index = self.root / self.dataset_id / "raw_index.jsonl"
        if self.dataset_id == "clotho" and raw_index.is_file():
            with raw_index.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    captions = [
                        str(item)
                        for item in row.get("captions") or []
                        if item not in (None, "")
                    ]
                    if not captions:
                        continue
                    for key in (row.get("audio_path"), row.get("audio_filepath")):
                        if key:
                            references[str(key)] = captions
        self._caption_references_by_path = references
        return references

    @staticmethod
    def _audio_uris(row: Dict[str, Any]) -> List[str]:
        """Read audio locations from canonical V2 and legacy FullEval rows."""
        input_obj = row.get("input") or {}
        if not isinstance(input_obj, dict):
            input_obj = {}
        uris: List[str] = []

        def append(value: Any) -> None:
            if isinstance(value, dict):
                value = value.get("uri") or value.get("path")
            if value in (None, ""):
                return
            uri = str(value)
            if uri not in uris:
                uris.append(uri)

        audio_items = input_obj.get("audio") or []
        if not isinstance(audio_items, list):
            audio_items = [audio_items]
        for item in audio_items:
            append(item)
        for key in (
            "audio_path",
            "audio_filepath",
            "audio_path_a",
            "audio_path_b",
        ):
            append(input_obj.get(key))
        for item in input_obj.get("audios") or []:
            append(item)
        append(row.get("WavPath"))
        numbered = sorted(
            (
                int(key[7:]),
                value,
            )
            for key, value in row.items()
            if key.startswith("WavPath") and key[7:].isdigit() and value
        )
        for _index, value in numbered:
            append(value)
        return uris

    @classmethod
    def _primary_audio_uri(cls, row: Dict[str, Any]) -> str:
        uris = cls._audio_uris(row)
        return uris[0] if uris else ""

    @classmethod
    def _covost_source_language(cls, row: Dict[str, Any]) -> str:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        for key in ("source_language", "language"):
            for container in (row, metadata):
                language = str(container.get(key) or "").lower()
                if language.startswith("zh"):
                    return "zh"
                if language.startswith("en"):
                    return "en"
        path = cls._primary_audio_uri(row).replace("\\", "/").lower()
        return "zh" if "/zh-cn/" in path else "en"
