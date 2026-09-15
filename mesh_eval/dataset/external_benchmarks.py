import ast
import csv
import hashlib
import json
import os
import re
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from audio_evals.config import get_environment
from mesh_eval.dataset.full_eval import FullEvalDataset


DEFAULT_AUDIO_AGENT_ROOT = Path(
    get_environment(
        "VOXMATRIX_AUDIO_AGENT_ROOT",
        "/mnt/afs/eval_data/benchmarks/AudioAgentBench",
    )
).expanduser()
DEFAULT_AUDIO_AGENT_CLARIFICATION_REVIEW = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "audioagentbench_clarification_review.v1.json"
)
DEFAULT_IHBENCH_ROOT = Path(
    get_environment(
        "VOXMATRIX_IHBENCH_ROOT",
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/IHBench",
    )
).expanduser()
DEFAULT_IHBENCH_CACHE = Path(
    get_environment(
        "VOXMATRIX_IHBENCH_CACHE",
        "/tmp/voxmatrix_ihbench_audio",
    )
).expanduser()
DEFAULT_VOCALBENCH_ROOT = Path(
    get_environment(
        "VOXMATRIX_VOCALBENCH_ROOT",
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench",
    )
).expanduser()
DEFAULT_VOCALBENCH_ZH_ROOT = Path(
    get_environment(
        "VOXMATRIX_VOCALBENCH_ZH_ROOT",
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh",
    )
).expanduser()
DEFAULT_STEPEVAL_ROOT = Path(
    get_environment(
        "VOXMATRIX_STEPEVAL_ROOT",
        "/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall",
    )
).expanduser()
DEFAULT_MTALK_ROOT = Path(
    get_environment(
        "VOXMATRIX_MTALK_ROOT", "/mnt/afs/eval_data/benchmarks/MTalk-Bench"
    )
).expanduser()
DEFAULT_EAR_ROOT = Path(
    get_environment("VOXMATRIX_EAR_ROOT", "/mnt/afs/eval_data/benchmarks/EAR")
).expanduser()
DEFAULT_VOICEBENCH_ROOT = Path(
    get_environment(
        "VOXMATRIX_VOICEBENCH_ROOT",
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench",
    )
).expanduser()
DEFAULT_FSC_ROOT = Path(
    get_environment(
        "VOXMATRIX_FSC_ROOT",
        "/mnt/afs/eval_data/04_dialogue_slu/fluent_speech_commands_dataset",
    )
).expanduser()
DEFAULT_SLURP_ROOT = Path(
    get_environment(
        "VOXMATRIX_SLURP_ROOT",
        "/mnt/afs/oss_data/datasets/04_dialogue_slu/SLURP_repo",
    )
).expanduser()


def _message(role: str, *, text: str = "", audio: str = "") -> Dict[str, Any]:
    contents = []
    if audio:
        contents.append({"type": "audio", "value": audio})
    if text:
        contents.append({"type": "text", "value": text})
    return {"role": role, "contents": contents}


class AudioAgentBenchDataset(FullEvalDataset):
    """Adapt AudioAgentBench turn metadata to the Mesh runtime contract."""

    def __init__(
        self,
        benchmark_id: str,
        view: str = "dialogue",
        root: str = "",
        strict: bool = True,
        check_audio_exists: bool = True,
        max_history_turns: Optional[int] = None,
        clarification_review_path: str = "",
    ):
        super().__init__(
            dataset="audio_agent_bench",
            benchmark_id=benchmark_id,
            view=view,
            root=root or str(DEFAULT_AUDIO_AGENT_ROOT.parent),
            strict=strict,
            check_audio_exists=check_audio_exists,
        )
        self.root = Path(root).expanduser() if root else DEFAULT_AUDIO_AGENT_ROOT
        self.clarification_review_path = (
            Path(clarification_review_path).expanduser()
            if clarification_review_path
            else DEFAULT_AUDIO_AGENT_CLARIFICATION_REVIEW
        )
        self.max_history_turns = (
            None if max_history_turns is None else int(max_history_turns)
        )
        if self.max_history_turns is not None and self.max_history_turns < 0:
            raise ValueError("max_history_turns must be non-negative")

    @property
    def manifest_path(self) -> Path:
        return self.root

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(offset=0, limit=limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        selected = []
        seen = 0
        for line_no, record in self._iter_records():
            if seen < offset:
                seen += 1
                continue
            selected.append(self._normalize(record, line_no))
            if limit and len(selected) >= limit:
                break
        return selected

    def count(self, limit=0) -> int:
        count = 0
        for _line_no, _record in self._iter_records():
            count += 1
            if limit and count >= limit:
                break
        return count

    def _clarification_cases(self) -> Dict[Tuple[str, int], Dict[str, Any]]:
        source = self.clarification_review_path
        if not source.is_file():
            raise FileNotFoundError(
                "AudioAgentBench clarification review is required for the "
                f"{self.view!r} view but was not found: {source}"
            )
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
            raise ValueError(
                f"unsupported AudioAgentBench clarification review schema: {source}"
            )
        raw_cases = payload.get("cases")
        review_id = str(payload.get("review_id") or "").strip()
        if not review_id:
            raise ValueError(
                f"AudioAgentBench clarification review has no review_id: {source}"
            )
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(
                f"AudioAgentBench clarification review has no cases: {source}"
            )

        cases: Dict[Tuple[str, int], Dict[str, Any]] = {}
        for index, raw_item in enumerate(raw_cases):
            if not isinstance(raw_item, Mapping):
                raise ValueError(
                    f"AudioAgentBench clarification review case {index} is not an object"
                )
            benchmark = str(raw_item.get("benchmark") or "").strip()
            target_turn_id = raw_item.get("target_turn_id")
            should_clarify = raw_item.get("should_clarify")
            if not benchmark or isinstance(target_turn_id, bool) or not isinstance(
                target_turn_id, int
            ):
                raise ValueError(
                    "AudioAgentBench clarification review case "
                    f"{index} has an invalid benchmark/target_turn_id"
                )
            if not isinstance(should_clarify, bool):
                raise ValueError(
                    "AudioAgentBench clarification review case "
                    f"{index} must contain a boolean should_clarify"
                )
            item = dict(raw_item)
            item["review_id"] = review_id
            for digest_field in ("input_text_sha256", "golden_text_sha256"):
                digest = item.get(digest_field)
                if not isinstance(digest, str) or re.fullmatch(
                    r"[0-9a-f]{64}", digest
                ) is None:
                    raise ValueError(
                        "AudioAgentBench clarification review case "
                        f"{index} has an invalid {digest_field}"
                    )
            key = (benchmark, target_turn_id)
            if key in cases:
                raise ValueError(
                    f"duplicate AudioAgentBench clarification review case: {key}"
                )
            cases[key] = item

        labels = {item["should_clarify"] for item in cases.values()}
        if self.view == "clarification" and labels != {False, True}:
            raise ValueError(
                "AudioAgentBench clarification evaluation requires reviewed "
                "positive and negative cases"
            )
        return cases

    @staticmethod
    def _validate_clarification_case(
        benchmark_name: str,
        item: Mapping[str, Any],
        clarification_case: Mapping[str, Any],
    ) -> None:
        for field in ("input_text", "golden_text"):
            actual = str(item.get(field) or "")
            actual_digest = hashlib.sha256(actual.encode("utf-8")).hexdigest()
            expected_digest = clarification_case[f"{field}_sha256"]
            if actual_digest != expected_digest:
                raise ValueError(
                    "AudioAgentBench clarification review does not match native "
                    f"metadata for {benchmark_name} turn {item.get('turn_id')} "
                    f"({field} digest mismatch)"
                )

    def _validate_clarification_source(
        self, cases: Mapping[Tuple[str, int], Mapping[str, Any]]
    ) -> None:
        """Validate every reviewed turn before yielding even a limited slice."""
        by_benchmark: Dict[str, Dict[int, Mapping[str, Any]]] = {}
        for (benchmark, turn_id), case in cases.items():
            by_benchmark.setdefault(benchmark, {})[turn_id] = case
        native_ambiguous_cases = set()
        for benchmark, benchmark_cases in by_benchmark.items():
            metadata_path = self.root / benchmark / "metadata.jsonl"
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    "AudioAgentBench clarification review references missing "
                    f"native metadata: {metadata_path}"
                )
            matched: Dict[int, int] = {turn_id: 0 for turn_id in benchmark_cases}
            with metadata_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    item = json.loads(raw_line)
                    turn_id = int(item.get("turn_id") or 0)
                    categories = item.get("categories") or []
                    if isinstance(categories, str):
                        categories = [
                            value.strip() for value in categories.split(",")
                        ]
                    if "ambiguous_entity" in categories:
                        native_ambiguous_cases.add((benchmark, turn_id))
                    case = benchmark_cases.get(turn_id)
                    if case is None:
                        continue
                    self._validate_clarification_case(benchmark, item, case)
                    matched[turn_id] += 1
            missing = [turn_id for turn_id, count in matched.items() if not count]
            if missing:
                raise ValueError(
                    "AudioAgentBench clarification review references absent turns "
                    f"in {benchmark}: {missing}"
                )
        reviewed_cases = set(cases)
        if native_ambiguous_cases != reviewed_cases:
            raise ValueError(
                "AudioAgentBench clarification review does not cover exactly the "
                "native ambiguous_entity turns; unreviewed="
                f"{sorted(native_ambiguous_cases - reviewed_cases)}, "
                f"non_native={sorted(reviewed_cases - native_ambiguous_cases)}"
            )

    def _iter_records(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"AudioAgentBench root not found: {self.root}")
        clarification_cases = (
            self._clarification_cases()
            if self.view in {"all", "dialogue", "clarification"}
            else {}
        )
        if clarification_cases:
            self._validate_clarification_source(clarification_cases)
        line_no = 0
        for metadata_path in sorted(self.root.glob("*-bench/metadata.jsonl")):
            benchmark_name = metadata_path.parent.name
            histories: Dict[str, List[Dict[str, Any]]] = {}
            with metadata_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    line_no += 1
                    item = json.loads(raw_line)
                    speaker = str(item.get("speaker") or "unknown")
                    history = histories.setdefault(speaker, [])
                    current_audio = metadata_path.parent / str(item["file_name"])
                    tool_call = self._parse_tool_call(item.get("required_function_call"))
                    clarification_case = clarification_cases.get(
                        (benchmark_name, int(item.get("turn_id") or 0))
                    )
                    is_clarification = clarification_case is not None
                    if clarification_case is not None:
                        self._validate_clarification_case(
                            benchmark_name, item, clarification_case
                        )
                    include = {
                        "all": True,
                        "dialogue": tool_call is None and not is_clarification,
                        "tool_call": tool_call is not None,
                        "clarification": is_clarification,
                    }.get(self.view)
                    if include is None:
                        raise ValueError(f"unsupported AudioAgentBench view: {self.view}")
                    if include:
                        yield line_no, self._record(
                            benchmark_name,
                            item,
                            str(current_audio),
                            history,
                            tool_call,
                            is_clarification,
                            clarification_case,
                        )
                    history.extend(
                        [
                            _message("user", audio=str(current_audio)),
                            _message(
                                "assistant", text=str(item.get("golden_text") or "")
                            ),
                        ]
                    )

    @staticmethod
    def _parse_tool_call(value: Any) -> Any:
        if value in (None, ""):
            return None
        parsed = json.loads(value) if isinstance(value, str) else deepcopy(value)

        def normalize(item: Any) -> Dict[str, Any]:
            if not isinstance(item, dict):
                raise ValueError(
                    f"AudioAgentBench tool call item is not an object: {item!r}"
                )
            return {
                "tool": item.get("name") or item.get("tool"),
                "arguments": item.get("args") or item.get("arguments") or {},
            }

        if isinstance(parsed, list):
            return [normalize(item) for item in parsed]
        return normalize(parsed)

    def _record(
        self,
        benchmark_name: str,
        item: Dict[str, Any],
        current_audio: str,
        history: List[Dict[str, Any]],
        tool_call: Any,
        is_clarification: bool,
        clarification_case: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if tool_call is not None and self.view != "clarification":
            capability = "tool_call"
            if isinstance(tool_call, list):
                instruction = (
                    'Return a JSON array only, with each item using {"tool": "...", '
                    '"arguments": {...}} in execution order. Do not include a '
                    "natural-language response."
                )
            else:
                instruction = (
                    'Return JSON only as {"tool": "...", "arguments": {...}}. '
                    "Do not include a natural-language response."
                )
            reference = {
                "expected_output": tool_call,
                "answer": str(item.get("golden_text") or ""),
            }
            metrics = ["tool_acc", "parameter_acc", "parameter_f1", "call_exact_match"]
            evaluators = ["mesh-tool-call"]
            prompt_name = "mesh-tool-call"
            answer_type = "structured_json"
        else:
            capability = "clarification" if is_clarification else "multi_turn_dialogue"
            if is_clarification:
                instruction = (
                    'Return JSON only as {"should_clarify": boolean, "response": string}. '
                    "Ask one targeted question only when required information is missing or ambiguous."
                )
            else:
                instruction = "Respond to the latest spoken turn using the conversation context."
            answer = str(item.get("golden_text") or "")
            if is_clarification:
                should_clarify = bool(
                    (clarification_case or {}).get("should_clarify")
                )
                reference = {
                    "should_clarify": should_clarify,
                    "answer": answer,
                    "turns": [{"role": "assistant", "content": answer}],
                    "rubric": (clarification_case or {}).get("subcategory"),
                    "review_id": (clarification_case or {}).get("review_id"),
                }
                metrics = [
                    "clarification_accuracy",
                    "clarification_precision",
                    "clarification_recall",
                    "clarification_f1",
                ]
                evaluators = ["mesh-clarification"]
                prompt_name = "mesh-clarification-decision"
                answer_type = "structured_json"
            else:
                reference = {
                    "answer": answer,
                    "turns": [{"role": "assistant", "content": answer}],
                }
                metrics = ["llm_judge"]
                evaluators = ["mesh-agent-rubric"]
                prompt_name = "mesh-contextual-audio"
                answer_type = "text"

        messages = [
            _message(
                "system",
                text=(
                    "You are a spoken task assistant. Use the supplied dialogue context "
                    "and follow the final output instruction exactly."
                ),
            )
        ]
        history_messages = history
        if self.max_history_turns is not None:
            history_messages = history[-2 * self.max_history_turns :]
        messages.extend(history_messages)
        messages.append(_message("user", audio=current_audio, text=instruction))
        audio_uris = [
            content["value"]
            for message in messages
            for content in message.get("contents") or []
            if content.get("type") == "audio"
        ]
        tool_schemas_path = self.root / benchmark_name / "benchmark" / "tool_schemas.json"
        try:
            tool_schemas = json.loads(tool_schemas_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            tool_schemas = []
        row = {
            "dataset": "audio_agent_bench",
            "task": "spoken_agentic_interaction",
            "capability": capability,
            "scenario": "home",
            "condition": {
                "acoustic": "clean",
                "spatial": "near_field",
                "speaker": "single_speaker",
                "device": "smart_speaker",
                "interaction": (
                    "clarification" if capability == "clarification" else "multi_turn"
                ),
            },
            "metrics": metrics,
            "evaluators": evaluators,
            "prompt_name": prompt_name,
            "answer_type": answer_type,
            "use_bucket": "diagnostic_evidence",
            "split": "test",
            "language": "en",
            "input": {
                "audio_path": current_audio,
                "audio": [{"uri": path} for path in audio_uris],
                "question": instruction,
                "messages": messages,
                "tool_schemas": tool_schemas,
            },
            "reference": reference,
            "sample_id": (
                f"audioagent_{benchmark_name}_{item.get('speaker')}_"
                f"{int(item.get('turn_id') or 0):03d}_{capability}"
            ),
            "label_meta": {
                "sources": {
                    "audio_path": "native",
                    "answer": "native",
                    "expected_output": "native",
                    "task": "knowledge_verified",
                    "capability": "knowledge_verified",
                    "should_clarify": "human",
                    "rubric": "human",
                    "review_id": "human",
                },
                "raw_benchmark": benchmark_name,
                "categories": item.get("categories"),
                "function_call_response": item.get("function_call_response"),
                "clarification_case": clarification_case,
            },
            "offline_proxy": (
                "contextual_turn_without_live_tool_runtime"
                if capability != "clarification"
                else "reviewed_clarification_decision_without_live_tool_runtime"
            ),
        }
        self._bind_protocol(row, "spoken_agentic_interaction", capability)
        return row


class _T4NativeDataset(FullEvalDataset):
    """Small common adapter for benchmark-native T4 files.

    Native files live outside the repository and differ substantially in
    layout.  Each subclass creates a legacy-shaped row and lets
    ``FullEvalDataset._normalize`` perform the same V2 protocol validation as
    the rest of the mesh.
    """

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def _normalize(self, record: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        preserved = dict(record)
        # ``adapt_v1_to_v2`` historically ignored legacy top-level language
        # fields.  Native T4 adapters do carry an explicit language, so move
        # it into the canonical metadata before normalization.
        metadata = dict(preserved.get("metadata") or {})
        for field in ("language", "source_language", "target_language"):
            if preserved.get(field) not in (None, "", []):
                metadata.setdefault(field, preserved[field])
        if metadata:
            preserved["metadata"] = metadata
        preserved["_preserve_protocol"] = True
        return super()._normalize(preserved, line_no)

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(0, limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        # Ask adapters for only the prefix needed to satisfy this slice.  A
        # few native parquet adapters materialize embedded audio; loading the
        # complete benchmark for ``load(10)`` would otherwise write hundreds
        # of temporary WAVs and make smoke checks needlessly expensive.
        needed = offset + limit if limit else 0
        rows = self._rows(needed)
        selected = rows[offset : offset + limit if limit else None]
        return [self._normalize(row, offset + index + 1) for index, row in enumerate(selected)]

    def count(self, limit=0) -> int:
        count = len(self._rows(0))
        return min(count, limit) if limit else count


def _audio_descriptor_input(paths: List[str]) -> List[Dict[str, str]]:
    return [{"uri": str(path)} for path in paths if path]


def _nonempty(value: Any, default: Any = None) -> Any:
    """Coerce Arrow/NumPy containers without using ambiguous truth tests."""
    if value is None:
        return default
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None:
        return default
    if isinstance(value, str) and not value:
        return default
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return default
    # pandas.NA and NaN are common in parquet columns.  Avoid ``value in`` or
    # a bare truth test, both of which can raise an ambiguous-array error.
    try:
        if bool(value) is False:
            return default
    except (TypeError, ValueError):
        pass
    if str(value) in {"<NA>", "NaT", "nan", "NaN"}:
        return default
    return value


class FluentSpeechCommandsT4Dataset(_T4NativeDataset):
    """Read FSC's native test CSV directly for the two T4 diagnostic views."""

    def __init__(
        self,
        benchmark_id: str,
        view: str = "instruction_following",
        root: str = "",
        strict: bool = True,
        check_audio_exists: bool = True,
    ):
        super().__init__(
            dataset="fluent_speech_commands",
            benchmark_id=benchmark_id,
            view=view,
            root=root or "data",
            strict=strict,
            check_audio_exists=check_audio_exists,
        )
        self.root = Path(root).expanduser() if root else DEFAULT_FSC_ROOT
        self.view = str(view)

    @property
    def manifest_path(self) -> Path:
        return self.root / "data" / "test_data.csv"

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"FSC test CSV not found: {self.manifest_path}")
        with self.manifest_path.open(encoding="utf-8") as handle:
            source_rows = list(csv.DictReader(handle))
        intents = sorted(
            {f"{item['action']}_{item['object']}" for item in source_rows}
        )
        slots = {
            intent: ["action", "object", "location"] for intent in intents
        }
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(source_rows):
            audio = str(self.root / str(item["path"]))
            intent = f"{item['action']}_{item['object']}"
            slot_values = {
                "action": item["action"],
                "object": item["object"],
                "location": item["location"],
            }
            if self.view == "tool_call":
                reference = {
                    "expected_output": {
                        "tool": "control_device",
                        "arguments": slot_values,
                    }
                }
                metrics = ["tool_acc", "parameter_acc", "parameter_f1", "call_exact_match"]
                evaluators = ["mesh-tool-call"]
                prompt_name = "mesh-tool-call"
                prompt_input = {
                    "audio": _audio_descriptor_input([audio]),
                    "audio_path": audio,
                    "tool_schemas": [
                        {
                            "name": "control_device",
                            "description": "Control the requested device or setting.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "action": {"type": "string", "enum": sorted({row["action"] for row in source_rows})},
                                    "object": {"type": "string", "enum": sorted({row["object"] for row in source_rows})},
                                    "location": {"type": "string", "enum": sorted({row["location"] for row in source_rows})},
                                },
                                "required": ["action", "object", "location"],
                            },
                        }
                    ],
                }
                capability = "tool_call"
            elif self.view == "instruction_following":
                reference = {"intent": intent, "slots": slot_values}
                metrics = ["intent_acc", "slot_f1"]
                evaluators = ["mesh-intent", "mesh-slot-f1"]
                prompt_name = "mesh-intent-slots"
                prompt_input = {
                    "audio": _audio_descriptor_input([audio]),
                    "audio_path": audio,
                    "allowed_intents": intents,
                    "slot_schema": slots,
                    "question": (
                        'Return JSON only as {"intent": string, "slots": object}. '
                        "Use one supplied intent and preserve native slot values."
                    ),
                }
                capability = "instruction_following"
            else:
                raise ValueError(f"unsupported FSC T4 view: {self.view}")
            rows.append(
                {
                    "dataset": "fluent_speech_commands",
                    "task": "spoken_agentic_interaction",
                    "capability": capability,
                    "scenario": "home",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "smart_speaker", "interaction": "single_turn"},
                    "metrics": metrics,
                    "evaluators": evaluators,
                    "prompt_name": prompt_name,
                    "answer_type": "structured_json",
                    "output_schema": "intent_slots@1" if capability == "instruction_following" else "tool_call@1",
                    "parser_name": "structured_json@1",
                    "required_reference": ["intent"] if capability == "instruction_following" else ["expected_output"],
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": "en",
                    "input": prompt_input,
                    "reference": reference,
                    "sample_id": f"fsc_{index}_{capability}",
                    "label_meta": {"sources": {"audio_path": "native", "reference": "native"}, "transcription": item.get("transcription")},
                    "offline_proxy": "native_slu_ontology_without_live_tool_runtime",
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows


class SLURPT4Dataset(_T4NativeDataset):
    """Read SLURP's native test JSONL directly for T4 diagnostic views."""

    def __init__(
        self,
        benchmark_id: str,
        view: str = "instruction_following",
        root: str = "",
        strict: bool = True,
        check_audio_exists: bool = True,
    ):
        super().__init__(
            dataset="slurp",
            benchmark_id=benchmark_id,
            view=view,
            root=root or "data",
            strict=strict,
            check_audio_exists=check_audio_exists,
        )
        self.root = Path(root).expanduser() if root else DEFAULT_SLURP_ROOT
        self.view = str(view)

    @property
    def manifest_path(self) -> Path:
        return self.root / "dataset" / "slurp" / "test.jsonl"

    @staticmethod
    def _slots(item: Mapping[str, Any]) -> Dict[str, Any]:
        tokens = item.get("tokens") or []
        surfaces = [str(token.get("surface") or "") for token in tokens]
        grouped: Dict[str, List[str]] = {}
        for entity in item.get("entities") or []:
            values = [
                surfaces[int(position)]
                for position in entity.get("span") or []
                if 0 <= int(position) < len(surfaces)
            ]
            grouped.setdefault(str(entity.get("type")), []).append(" ".join(values))
        # Keep scalars for the common case, but preserve repeated native
        # entities as arrays. A plain dict assignment dropped entities in 85
        # rows of the SLURP test set.
        return {
            key: values[0] if len(values) == 1 else values
            for key, values in grouped.items()
        }

    def _audio_path(self, recording: Mapping[str, Any]) -> str:
        name = str(recording.get("file") or "")
        candidates = [
            self.root / name,
            self.root / "audio" / "slurp_real" / name,
            self.root / "audio" / name,
        ]
        for path in candidates:
            if path.is_file():
                return str(path)
        return str(candidates[1])

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"SLURP test JSONL not found: {self.manifest_path}")
        source_rows = [
            json.loads(line)
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        intents = sorted({str(item.get("intent")) for item in source_rows if item.get("intent")})
        slot_schema: Dict[str, set] = {intent: set() for intent in intents}
        for item in source_rows:
            slot_schema.setdefault(str(item.get("intent")), set()).update(self._slots(item))
        native_slots = {key: sorted(value) for key, value in slot_schema.items()}
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(source_rows):
            recordings = item.get("recordings") or []
            if not recordings:
                continue
            recording = recordings[-1] if int(item.get("slurp_id") or 0) % 2 and len(recordings) > 1 else recordings[0]
            audio = self._audio_path(recording)
            intent = str(item.get("intent") or "")
            scenario = str(item.get("scenario") or "")
            slots = self._slots(item)
            if self.view == "tool_call":
                reference = {
                    "expected_output": {
                        "tool": "execute_slurp_intent",
                        "arguments": {"intent": intent, "scenario": scenario, "slots": slots},
                    }
                }
                metrics = ["tool_acc", "parameter_acc", "parameter_f1", "call_exact_match"]
                evaluators = ["mesh-tool-call"]
                prompt_name = "mesh-tool-call"
                prompt_input = {
                    "audio": _audio_descriptor_input([audio]),
                    "audio_path": audio,
                    "tool_schemas": [
                        {
                            "name": "execute_slurp_intent",
                            "description": "Execute one native SLURP intent.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "intent": {"type": "string", "enum": intents},
                                    "scenario": {"type": "string", "enum": sorted({str(row.get("scenario")) for row in source_rows})},
                                    "slots": {"type": "object"},
                                },
                                "required": ["intent", "scenario", "slots"],
                            },
                        }
                    ],
                }
                capability = "tool_call"
            elif self.view == "instruction_following":
                reference = {"intent": intent, "slots": slots}
                metrics = ["intent_acc", "slot_f1"]
                evaluators = ["mesh-intent", "mesh-slot-f1"]
                prompt_name = "mesh-intent-slots"
                prompt_input = {
                    "audio": _audio_descriptor_input([audio]),
                    "audio_path": audio,
                    "allowed_intents": intents,
                    "slot_schema": native_slots,
                    "question": (
                        'Return JSON only as {"intent": string, "slots": object}. '
                        "Use one supplied intent and native slot names."
                    ),
                }
                capability = "instruction_following"
            else:
                raise ValueError(f"unsupported SLURP T4 view: {self.view}")
            rows.append(
                {
                    "dataset": "slurp",
                    "task": "spoken_agentic_interaction",
                    "capability": capability,
                    "scenario": "home",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "single_turn"},
                    "metrics": metrics,
                    "evaluators": evaluators,
                    "prompt_name": prompt_name,
                    "answer_type": "structured_json",
                    "output_schema": "intent_slots@1" if capability == "instruction_following" else "tool_call@1",
                    "parser_name": "structured_json@1",
                    "required_reference": ["intent"] if capability == "instruction_following" else ["expected_output"],
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": "en",
                    "input": prompt_input,
                    "reference": reference,
                    "sample_id": f"slurp_{item.get('slurp_id', index)}_{capability}",
                    "label_meta": {"sources": {"audio_path": "native", "reference": "native"}, "sentence": item.get("sentence"), "recording": dict(recording)},
                    "offline_proxy": "native_slu_ontology_without_live_tool_runtime",
                }
            )
            if limit and len(rows) >= limit:
                break
        return rows


class VocalBenchDataset(_T4NativeDataset):
    """VocalBench/VocalBench-zh instruction and multi-turn adapter."""

    def __init__(
        self,
        benchmark_id: str,
        view: str = "instruction_following",
        root: str = "",
        language: str = "en",
        strict: bool = True,
        check_audio_exists: bool = True,
    ):
        dataset = "vocalbench_zh" if str(language).lower().startswith("zh") else "vocalbench"
        super().__init__(
            dataset=dataset,
            benchmark_id=benchmark_id,
            view=view,
            root=root or "data",
            strict=strict,
            check_audio_exists=check_audio_exists,
        )
        self.root = Path(root).expanduser() if root else (
            DEFAULT_VOCALBENCH_ZH_ROOT if dataset.endswith("_zh") else DEFAULT_VOCALBENCH_ROOT
        )
        self.view = str(view)
        self.language = "zh" if dataset.endswith("_zh") else "en"

    @property
    def manifest_path(self) -> Path:
        return self.root / "json" / (
            "instruction_following.json" if self.view in {"instruction_following", "ifeval"} else "multi_round.json"
        )

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"VocalBench manifest not found: {self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        rows: List[Dict[str, Any]] = []
        instruction = self.view in {"instruction_following", "ifeval"}
        for index, item in enumerate(payload):
            audio_path = self.root / "audio" / str(item.get("Audio") or item.get("audio"))
            if instruction:
                # VocalBench's first 200 instruction examples include speech
                # style/emotion/speed requirements; text-only output cannot
                # receive a complete official score for these rows.
                requires_audio = index < 200
                reference = {
                    "question": item.get("Question"),
                    "rubric": {
                        "instruction": item.get("Question"),
                        "category": item.get("Category"),
                        "subcategory": item.get("Sub-category"),
                    },
                    "instruction_ids": _nonempty(item.get("instruction_id_list"), _nonempty(item.get("instruction_ids"), [])),
                    "kwargs": _nonempty(item.get("kwargs"), []),
                    "requires_audio_output": requires_audio,
                }
                row = {
                    "dataset": self.dataset_id,
                    "task": "spoken_agentic_interaction",
                    "capability": "instruction_following",
                    "scenario": "phone",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "single_turn"},
                    "metrics": ["constraint_satisfaction"],
                    "evaluators": ["mesh-agent-rubric"],
                    "prompt_name": "mesh-agent-rubric",
                    "answer_type": "audio" if requires_audio else "text",
                    "output_schema": "audio@1" if requires_audio else "answer@1",
                    "parser_name": "generated_audio@1" if requires_audio else "answer_text@1",
                    "required_reference": ["rubric"],
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": self.language,
                    "input": {"audio": _audio_descriptor_input([str(audio_path)]), "audio_path": str(audio_path), "question": "Follow every spoken instruction exactly and return only the requested result."},
                    "reference": reference,
                    "sample_id": f"{self.dataset_id}_instruction_{item.get('Qid', index)}",
                    "label_meta": {"sources": {"audio_path": "native", "reference": "native"}, "question": item.get("Question"), "requires_audio_output": requires_audio},
                }
            else:
                context = item.get("Context") or []
                messages = [{"role": "system", "contents": [{"type": "text", "value": "Use the complete conversation context and answer the latest spoken turn."}]}]
                for turn in context:
                    role = "assistant" if str(turn.get("from") or "").lower() == "assistant" else "user"
                    messages.append(_message(role, text=str(turn.get("value") or "")))
                messages.append(_message("user", audio=str(audio_path)))
                answer = str(item.get("Answer") or "")
                row = {
                    "dataset": self.dataset_id,
                    "task": "spoken_agentic_interaction",
                    "capability": "multi_turn_dialogue",
                    "scenario": "phone",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "multi_turn"},
                    "metrics": ["llm_judge"],
                    "evaluators": ["mesh-agent-rubric"],
                    "prompt_name": "mesh-contextual-audio",
                    "answer_type": "text",
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": self.language,
                    "input": {"audio": _audio_descriptor_input([str(audio_path)]), "audio_path": str(audio_path), "messages": messages},
                    "reference": {"answer": answer, "turns": [{"role": "assistant", "content": answer}], "rubric": item.get("Question")},
                    "sample_id": f"{self.dataset_id}_multi_{item.get('Qid', index)}",
                    "label_meta": {"sources": {"audio_path": "native", "answer": "native"}, "context_turns": len(context)},
                }
            rows.append(row)
            if limit and len(rows) >= limit:
                break
        return rows


class StepEvalAudioToolCallDataset(_T4NativeDataset):
    """Adapter for StepEval's positive/negative trigger protocol."""

    TOOL_ALIASES = {"web_search": "web_search", "timbre": "timbre"}

    def __init__(self, benchmark_id: str, root: str = "", strict: bool = True, check_audio_exists: bool = True):
        super().__init__(dataset="stepeval_audio_toolcall", benchmark_id=benchmark_id, view="all", root=root or "data", strict=strict, check_audio_exists=check_audio_exists)
        self.root = Path(root).expanduser() if root else DEFAULT_STEPEVAL_ROOT

    @property
    def manifest_path(self) -> Path:
        return self.root

    @staticmethod
    def _gold_call(value: Any) -> Any:
        text = str(value or "")
        match = re.search(r"<tool_call>\s*function\s*\n\s*([^\n]+)\s*\n\s*(\{[\s\S]*?\})\s*</tool_call>", text, re.I)
        if not match:
            return None
        try:
            args = json.loads(match.group(2))
        except json.JSONDecodeError:
            return None
        return {"name": match.group(1).strip(), "arguments": args}

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for path in sorted(self.root.glob("*.jsonl")):
            stem = path.stem
            family = stem.rsplit("_", 1)[0]
            polarity = "positive" if stem.endswith("_positive") else "negative"
            target_tool = self.TOOL_ALIASES.get(family, family)
            payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for session_index, session in enumerate(payload):
                if not isinstance(session, list) or len(session) < 2:
                    continue
                final = session[-1]
                gold = self._gold_call(final.get("text") or final.get("value"))
                messages = [{"role": "system", "contents": [{"type": "text", "value": "Use a supplied tool only when the latest request actually requires it. Return JSON only as {\"tool\": string|null, \"arguments\": object}."}]}]
                audio_paths: List[str] = []
                for turn in session[:-1]:
                    source = str(turn.get("from") or "").lower()
                    role = "user" if source in {"human", "user"} else "assistant"
                    value = turn.get("value")
                    if isinstance(value, Mapping) and value.get("type") == "wav":
                        audio = self.root / "Step-Audio_Toolcall" / str(value.get("value"))
                        audio_paths.append(str(audio))
                        messages.append(_message(role, audio=str(audio)))
                    elif role == "user":
                        messages.append(_message(role, text=str(turn.get("text") or "")))
                    else:
                        messages.append(_message(role, text=str(turn.get("text") or "")))
                if not audio_paths:
                    continue
                schemas = self._schemas()
                expected = {
                    "target_tool": target_tool,
                    "polarity": polarity,
                    "gold_call": gold,
                    # Keep the canonical protocol field populated as well as
                    # the StepEval-specific trigger metadata.
                    "expected_output": gold,
                }
                rows.append({
                    "dataset": "stepeval_audio_toolcall",
                    "task": "spoken_agentic_interaction",
                    "capability": "tool_call",
                    "scenario": "home",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "multi_turn"},
                    "metrics": ["trigger_correct", "trigger_precision", "trigger_recall", "tool_type_accuracy", "parameter_judge_accuracy"],
                    "evaluators": ["mesh-tool-trigger"],
                    "prompt_name": "mesh-tool-call",
                    "answer_type": "structured_json",
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": "zh",
                    "input": {"audio": _audio_descriptor_input(audio_paths), "messages": messages, "tool_schemas": schemas, "audio_path": audio_paths[-1]},
                    "reference": expected,
                    "sample_id": f"stepeval_{stem}_{session_index}",
                    "label_meta": {"sources": {"audio_path": "native", "gold_call": "native"}, "source_file": path.name, "target_family": family},
                })
                if limit and len(rows) >= limit:
                    return rows
        return rows

    def _schemas(self) -> List[Dict[str, Any]]:
        # The released benchmark has no standalone schema file.  Expose the
        # names used by the official evaluator and native gold calls.
        return [
            {"name": "get_weather", "description": "获取给定地区的天气情况。location 必须是英文地名。", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
            {"name": "get_date_time", "description": "获取中国北京时间。", "parameters": {"type": "object", "properties": {}, "required": []}},
            {"name": "timbre_rag", "description": "按中文人物音色描述改变音色；不用于情感、语速或音量调整。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
            {"name": "search", "description": "网页搜索工具。", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        ]


class MTalkBenchDataset(_T4NativeDataset):
    """MTalk speech-to-speech rubric adapter; never invents an ELO score."""

    def __init__(self, benchmark_id: str, root: str = "", cache_dir: str = "", strict: bool = True, check_audio_exists: bool = True):
        super().__init__(dataset="mtalk_bench", benchmark_id=benchmark_id, view="multi_turn", root=root or "data", strict=strict, check_audio_exists=check_audio_exists)
        self.root = Path(root).expanduser() if root else DEFAULT_MTALK_ROOT
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else Path("/tmp/voxmatrix_mtalk_audio")

    @property
    def manifest_path(self) -> Path:
        return self.root / "data" / "test-00000-of-00001.parquet"

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        import pandas as pd

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"MTalk parquet not found: {self.manifest_path}")
        dataframe = pd.read_parquet(self.manifest_path)
        rows: List[Dict[str, Any]] = []
        # ``number`` repeats across task types; grouping by number alone
        # incorrectly merges unrelated conversations.
        for (task_type, number), group in dataframe.groupby(["type", "number"], sort=True):
            group = group.sort_values("turn")
            system_message = {"role": "system", "contents": [{"type": "text", "value": "Maintain the dialogue state and answer each spoken turn. Generate speech for every turn."}]}
            messages = [system_message]
            audio_paths: List[str] = []
            transcriptions: List[str] = []
            for index, item in enumerate(group.to_dict("records")):
                payload = item.get("audio")
                raw = payload.get("bytes") if isinstance(payload, Mapping) else payload
                if isinstance(raw, str):
                    try:
                        raw = ast.literal_eval(raw)
                    except Exception:
                        raw = None
                if not isinstance(raw, (bytes, bytearray)):
                    continue
                destination = self.cache_dir / f"{task_type}-{number}-{index}.wav"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file() or destination.stat().st_size != len(raw):
                    destination.write_bytes(bytes(raw))
                audio_paths.append(str(destination))
                transcriptions.append(str(item.get("transcription") or ""))
                messages.append(_message("user", audio=str(destination)))
            if not audio_paths:
                continue
            first = group.iloc[0]
            rows.append({
                "dataset": "mtalk_bench",
                "task": "spoken_agentic_interaction",
                "capability": "multi_turn_dialogue",
                "scenario": "home",
                "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "multi_turn"},
                "metrics": ["llm_judge"],
                "evaluators": ["mesh-agent-rubric"],
                "prompt_name": "mesh-contextual-audio",
                "answer_type": "audio",
                "use_bucket": "diagnostic_evidence",
                "split": "test",
                "language": "en",
                "input": {"audio": _audio_descriptor_input(audio_paths), "messages": messages, "multi_turn": [[deepcopy(system_message), _message("user", audio=path)] if index == 0 else [_message("user", audio=path)] for index, path in enumerate(audio_paths)], "audio_path": audio_paths[-1], "requires_audio_output": True},
                "reference": {"turns": [{"role": "user", "content": text} for text in transcriptions], "rubric": first.get("rubric_prompt_specific") or first.get("rubric_prompt_general"), "arena_prompt": first.get("arena_prompt"), "requires_audio_output": True, "required_audio_outputs": len(audio_paths)},
                "sample_id": f"mtalk_{task_type}_{number}",
                "label_meta": {"sources": {"audio_path": "native", "rubric": "native"}, "task_type": str(task_type), "conversation_number": str(number), "turn_count": len(audio_paths)},
            })
            if limit and len(rows) >= limit:
                break
        return rows


class EARWDYLDataset(_T4NativeDataset):
    """EAR WDYL keyword-masking clarification decision adapter."""

    def __init__(self, benchmark_id: str, root: str = "", strict: bool = True, check_audio_exists: bool = True):
        super().__init__(dataset="ear_wdyl", benchmark_id=benchmark_id, view="clarification", root=root or "data", strict=strict, check_audio_exists=check_audio_exists)
        self.root = Path(root).expanduser() if root else DEFAULT_EAR_ROOT

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifests" / "ear_wdyl.jsonl"

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for index, line in enumerate(self.manifest_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            item = json.loads(line)
            audio = str(item.get("audio") or "")
            should = not bool(item.get("answerable"))
            rows.append({
                "dataset": "ear_wdyl",
                "task": "spoken_agentic_interaction",
                "capability": "clarification",
                "scenario": "phone",
                "condition": {"acoustic": "light_noise", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "clarification"},
                "metrics": ["clarification_accuracy", "clarification_precision", "clarification_recall", "clarification_f1"],
                "evaluators": ["mesh-clarification"],
                "prompt_name": "mesh-clarification-decision",
                "answer_type": "structured_json",
                "use_bucket": "diagnostic_evidence",
                "split": "test",
                "language": "en",
                "input": {"audio": _audio_descriptor_input([audio]), "audio_path": audio, "question": "Return an explicit clarification decision JSON."},
                "reference": {"should_clarify": should, "answerable": bool(item.get("answerable")), "masked_word": item.get("masked_word")},
                "sample_id": str(item.get("id") or f"ear_wdyl_{index}"),
                "label_meta": {"sources": {"audio_path": "native", "should_clarify": "verified_derived"}, "condition": item.get("condition"), "masked_word": item.get("masked_word")},
            })
            if limit and len(rows) >= limit:
                break
        return rows


class VoiceBenchT4Dataset(_T4NativeDataset):
    """VoiceBench IFEval and MTBench views used by T4."""

    def __init__(self, benchmark_id: str, view: str = "ifeval", root: str = "", cache_dir: str = "", strict: bool = True, check_audio_exists: bool = True):
        super().__init__(dataset="voicebench", benchmark_id=benchmark_id, view=view, root=root or "data", strict=strict, check_audio_exists=check_audio_exists)
        self.root = Path(root).expanduser() if root else DEFAULT_VOICEBENCH_ROOT
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else Path("/tmp/voxmatrix_voicebench_audio")

    @property
    def manifest_path(self) -> Path:
        return self.root / ("ifeval" if self.view == "ifeval" else "mtbench") / "test-00000-of-00001.parquet"

    @staticmethod
    def _materialize(value: Any, destination: Path) -> str:
        raw = value.get("bytes") if isinstance(value, Mapping) else value
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw)
            except Exception:
                raw = None
        if not isinstance(raw, (bytes, bytearray)):
            raise ValueError("VoiceBench row has no embedded audio bytes")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.is_file() or destination.stat().st_size != len(raw):
            destination.write_bytes(bytes(raw))
        return str(destination)

    def _rows(self, limit: int = 0) -> List[Dict[str, Any]]:
        import pandas as pd

        dataframe = pd.read_parquet(self.manifest_path)
        rows: List[Dict[str, Any]] = []
        for index, item in enumerate(dataframe.to_dict("records")):
            if self.view == "ifeval":
                key = str(item.get("key") or index)
                audio = self._materialize(item.get("audio"), self.cache_dir / f"ifeval-{key}.wav")
                ids = _nonempty(item.get("instruction_id_list"), [])
                kwargs = _nonempty(item.get("kwargs"), [])
                rows.append({
                    "dataset": "voicebench",
                    "task": "spoken_agentic_interaction",
                    "capability": "instruction_following",
                    "scenario": "phone",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "single_turn"},
                    "metrics": ["constraint_satisfaction"],
                    "evaluators": ["mesh-ifeval"],
                    "prompt_name": "mesh-agent-rubric",
                    "answer_type": "text",
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": "en",
                    "input": {"audio": _audio_descriptor_input([audio]), "audio_path": audio, "question": "Follow every explicit instruction in the audio."},
                    "reference": {"prompt": item.get("prompt"), "instruction_id_list": ids, "kwargs": kwargs, "rubric": {"instruction_id_list": ids, "kwargs": kwargs}},
                    "output_schema": "answer@1",
                    "parser_name": "answer_text@1",
                    "required_reference": ["rubric"],
                    "sample_id": f"voicebench_ifeval_{key}",
                    "label_meta": {"sources": {"audio_path": "native", "ifeval_reference": "native"}, "prompt": item.get("prompt"), "instruction_ids": ids, "kwargs": kwargs},
                })
            else:
                qid = str(item.get("question_id") or index)
                a1 = self._materialize(item.get("audio1"), self.cache_dir / f"mtbench-{qid}-1.wav")
                a2 = self._materialize(item.get("audio2"), self.cache_dir / f"mtbench-{qid}-2.wav")
                turns = _nonempty(item.get("turns"), [])
                turns = [str(value) for value in turns]
                reference_answers = _nonempty(item.get("reference"), [])
                if not isinstance(reference_answers, (list, tuple)):
                    reference_answers = [reference_answers]
                reference_answers = [str(value) for value in reference_answers]
                system = _message(
                    "system",
                    text="Use all preceding dialogue context and answer the latest spoken turn.",
                )
                rows.append({
                    "dataset": "voicebench",
                    "task": "spoken_agentic_interaction",
                    "capability": "multi_turn_dialogue",
                    "scenario": "phone",
                    "condition": {"acoustic": "clean", "spatial": "near_field", "speaker": "single_speaker", "device": "phone_mic", "interaction": "multi_turn"},
                    "metrics": ["llm_judge"],
                    "evaluators": ["mesh-agent-rubric"],
                    "prompt_name": "mesh-contextual-audio",
                    "answer_type": "text",
                    "use_bucket": "diagnostic_evidence",
                    "split": "test",
                    "language": "en",
                    "input": {"audio": _audio_descriptor_input([a1, a2]), "audio_path": a2, "multi_turn": [[system, _message("user", audio=a1)], [_message("user", audio=a2)]]},
                    "reference": {"turns": turns, "reference_answers": reference_answers, "rubric": {"protocol": "VoiceBench MTBench", "category": item.get("category"), "reference_answers": reference_answers}, "requires_audio_output": False},
                    "sample_id": f"voicebench_mtbench_{qid}",
                    "label_meta": {"sources": {"audio_path": "native", "rubric": "native"}, "turns": turns, "category": item.get("category")},
                })
            if limit and len(rows) >= limit:
                break
        return rows


# Short aliases are kept for registry/config compatibility.
VocalBenchT4Dataset = VocalBenchDataset
VoiceBenchDataset = VoiceBenchT4Dataset


class IHBenchDataset(FullEvalDataset):
    """Offline recovery proxy for IHBench's real-time interruption protocol."""

    def __init__(
        self,
        benchmark_id: str,
        root: str = "",
        cache_dir: str = "",
        strict: bool = True,
        check_audio_exists: bool = True,
    ):
        super().__init__(
            dataset="ihbench",
            benchmark_id=benchmark_id,
            view="offline_recovery",
            root=root or str(DEFAULT_IHBENCH_ROOT.parent),
            strict=strict,
            check_audio_exists=check_audio_exists,
        )
        self.root = Path(root).expanduser() if root else DEFAULT_IHBENCH_ROOT
        self.cache_dir = (
            Path(cache_dir).expanduser() if cache_dir else DEFAULT_IHBENCH_CACHE
        )

    @property
    def manifest_path(self) -> Path:
        return self.root / "conversations.parquet"

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(offset=0, limit=limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        if offset < 0 or limit < 0:
            raise ValueError("offset and limit must be non-negative")
        baseline_path = self.root / "baseline.parquet"
        conversations_path = self.root / "conversations.parquet"
        if not baseline_path.is_file() or not conversations_path.is_file():
            raise FileNotFoundError(f"IHBench parquet files not found under {self.root}")
        import pandas as pd

        baseline = pd.read_parquet(baseline_path)
        conversations = pd.read_parquet(conversations_path).set_index("conversation_id")
        end = offset + limit if limit else None
        rows = []
        for source_index, item in baseline.iloc[offset:end].iterrows():
            conversation_id = str(item["conversation_id"])
            turn = int(item["turn"])
            conversation = conversations.loc[conversation_id]
            audio_value = conversation.get(f"user_turn_{turn}_audio")
            audio_path = self._materialize_audio(conversation_id, turn, audio_value)
            rows.append(
                self._normalize(
                    self._record(conversation_id, turn, item, conversation, audio_path),
                    int(source_index) + 1,
                )
            )
        return rows

    def count(self, limit=0) -> int:
        import pandas as pd

        count = len(pd.read_parquet(self.root / "baseline.parquet", columns=["turn"]))
        return min(count, limit) if limit else count

    def _materialize_audio(
        self, conversation_id: str, turn: int, audio_value: Any
    ) -> str:
        if not isinstance(audio_value, dict):
            raise ValueError(
                f"IHBench {conversation_id} turn {turn} has no embedded audio object"
            )
        payload = audio_value.get("bytes")
        if not isinstance(payload, (bytes, bytearray)) or not payload:
            raise ValueError(
                f"IHBench {conversation_id} turn {turn} has no embedded audio bytes"
            )
        relative = Path(str(audio_value.get("path") or f"{conversation_id}/{turn}.wav"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe IHBench audio path: {relative}")
        destination = (self.cache_dir / relative).resolve()
        cache_root = self.cache_dir.resolve()
        if cache_root not in destination.parents:
            raise ValueError(f"IHBench audio path escapes cache: {relative}")
        if destination.is_file() and destination.stat().st_size == len(payload):
            return str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, destination)
        return str(destination)

    def _record(
        self,
        conversation_id: str,
        turn: int,
        baseline: Any,
        conversation: Any,
        audio_path: str,
    ) -> Dict[str, Any]:
        interrupted = str(conversation.get(f"assistant_turn_{turn}_transcript") or "")
        expected = str(baseline.get("response") or "")
        interruption_type = str(
            conversation.get(f"user_turn_{turn}_interruption_type") or "unknown"
        )
        task_rubric = str(conversation.get(f"turn_{turn}_tf_rubric") or "")
        raw_rubrics = conversation.get(f"turn_{turn}_rq_rubrics")
        if hasattr(raw_rubrics, "tolist"):
            raw_rubrics = raw_rubrics.tolist()
        if isinstance(raw_rubrics, str):
            try:
                raw_rubrics = json.loads(raw_rubrics)
            except json.JSONDecodeError:
                raw_rubrics = [raw_rubrics] if raw_rubrics.strip() else []
        rubrics = list(raw_rubrics) if isinstance(raw_rubrics, (list, tuple)) else []
        instruction = (
            "Respond naturally to the interruption. Continue or restart as appropriate; "
            "do not mention that you are being evaluated."
        )
        messages = [
            _message(
                "system",
                text=str(conversation.get("system_message") or "")
                or "You are a helpful spoken assistant.",
            )
        ]
        audio_paths: List[str] = []
        dialogue_context: List[Dict[str, str]] = []
        # Keep the complete conversation state through the interrupted turn.
        # User transcripts remain metadata; only their audio enters the model.
        for history_turn in range(1, turn + 1):
            assistant_text = conversation.get(f"assistant_turn_{history_turn}_transcript")
            if assistant_text:
                messages.append(_message("assistant", text=str(assistant_text)))
                dialogue_context.append(
                    {"role": "assistant", "content": str(assistant_text)}
                )
            user_value = conversation.get(f"user_turn_{history_turn}_audio")
            if isinstance(user_value, dict):
                history_audio = self._materialize_audio(
                    conversation_id, history_turn, user_value
                )
                audio_paths.append(history_audio)
                messages.append(_message("user", audio=history_audio))
                dialogue_context.append(
                    {
                        "role": "user",
                        "content": str(
                            conversation.get(f"user_turn_{history_turn}_transcript")
                            or ""
                        ),
                    }
                )
        messages.append(_message("user", text=instruction))
        row = {
            "_preserve_protocol": True,
            "dataset": "ihbench",
            "task": "spoken_agentic_interaction",
            "capability": "interruption",
            "scenario": "phone",
            "condition": {
                "acoustic": "clean",
                "spatial": "near_field",
                "speaker": "single_speaker",
                "device": "phone_mic",
                "interaction": "interruption",
            },
            "metrics": ["tf_win_score", "rq_pass"],
            "evaluators": ["mesh-ihbench"],
            "prompt_name": "mesh-contextual-audio",
            "answer_type": "text",
            "use_bucket": "diagnostic_evidence",
            "split": "test",
            "language": "en",
            "input": {
                "audio_path": audio_path,
                "audio": _audio_descriptor_input(audio_paths),
                "question": instruction,
                "messages": messages,
            },
            "reference": {
                "baseline": expected,
                # ``answer`` is a compatibility alias consumed by generic
                # text/judge runners; the official IHBench fields remain the
                # explicit TF/RQ rubrics below.
                "answer": expected,
                "turns": [{"role": "assistant", "content": expected}],
                "task_fulfillment_rubric": task_rubric,
                "response_quality_rubrics": rubrics,
                "conversation_context": dialogue_context,
                "system_message": str(conversation.get("system_message") or ""),
                "goal": str(conversation.get("goal") or ""),
                "knowledge_base": conversation.get("knowledge_base"),
            },
            "sample_id": f"ihbench_{conversation_id}_turn_{turn:02d}",
            "label_meta": {
                "sources": {
                    "audio_path": "native",
                    "answer": "llm_draft",
                    "task": "knowledge_verified",
                    "capability": "knowledge_verified",
                },
                "domain": conversation.get("domain"),
                "interruption_type": interruption_type,
                "interrupting_user_message_index": int(
                    baseline.get("interrupting_user_message_index")
                ),
            },
            "offline_proxy": "native_tf_rq_rubric_without_streaming_timing",
        }
        self._bind_protocol(row, "spoken_agentic_interaction", "interruption")
        return row
