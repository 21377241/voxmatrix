import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

from audio_evals.config import get_environment
from mesh_eval.dataset.full_eval import FullEvalDataset


DEFAULT_AUDIO_AGENT_ROOT = Path(
    get_environment(
        "VOXMATRIX_AUDIO_AGENT_ROOT",
        "data/AudioAgentBench",
    )
).expanduser()
DEFAULT_IHBENCH_ROOT = Path(
    get_environment(
        "VOXMATRIX_IHBENCH_ROOT",
        "data/IHBench",
    )
).expanduser()
DEFAULT_IHBENCH_CACHE = Path(
    get_environment(
        "VOXMATRIX_IHBENCH_CACHE",
        "data/ihbench_audio",
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
        max_history_turns: int = 8,
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
        self.max_history_turns = int(max_history_turns)
        if self.max_history_turns < 0:
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

    def _clarification_cases(self) -> set[Tuple[str, int]]:
        source = self.root / "clarification_review.json"
        if not source.is_file():
            return set()
        with source.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return {
            (str(item["benchmark"]), int(item["target_turn_id"]))
            for item in payload.get("cases") or []
            if item.get("include")
        }

    def _iter_records(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"AudioAgentBench root not found: {self.root}")
        clarification_cases = self._clarification_cases()
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
                    is_clarification = (
                        benchmark_name,
                        int(item.get("turn_id") or 0),
                    ) in clarification_cases
                    include = {
                        "all": True,
                        "dialogue": tool_call is None,
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
            metrics = ["tool_acc", "parameter_acc", "parameter_f1", "task_success"]
            evaluators = ["mesh-tool-call"]
            answer_type = "structured_json"
        else:
            capability = "clarification" if is_clarification else "multi_turn_dialogue"
            instruction = "Respond to the latest spoken turn using the conversation context."
            answer = str(item.get("golden_text") or "")
            reference = {
                "answer": answer,
                "turns": [{"role": "assistant", "content": answer}],
            }
            metrics = ["text_f1"]
            evaluators = ["mesh-text-f1"]
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
        if self.max_history_turns:
            messages.extend(history[-2 * self.max_history_turns :])
        messages.append(_message("user", audio=current_audio, text=instruction))
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
            "prompt_name": "mesh-contextual-audio",
            "answer_type": answer_type,
            "use_bucket": "diagnostic_evidence",
            "split": "test",
            "language": "en",
            "input": {
                "audio_path": current_audio,
                "question": instruction,
                "messages": messages,
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
                },
                "raw_benchmark": benchmark_name,
                "categories": item.get("categories"),
                "function_call_response": item.get("function_call_response"),
            },
            "offline_proxy": "contextual_turn_without_live_tool_runtime",
        }
        self._bind_protocol(row, "spoken_agentic_interaction", capability)
        return row


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
        rubrics = raw_rubrics.tolist() if hasattr(raw_rubrics, "tolist") else []
        instruction = (
            "Respond naturally to the interruption. Continue or restart as appropriate; "
            "do not mention that you are being evaluated."
        )
        messages = [
            _message(
                "system",
                text=str(conversation.get("system_message") or "")
                or "You are a helpful spoken assistant.",
            ),
            _message("assistant", text=interrupted),
            _message("user", audio=audio_path, text=instruction),
        ]
        row = {
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
            "metrics": ["text_f1"],
            "evaluators": ["mesh-text-f1"],
            "prompt_name": "mesh-contextual-audio",
            "answer_type": "text",
            "use_bucket": "diagnostic_evidence",
            "split": "test",
            "language": "en",
            "input": {
                "audio_path": audio_path,
                "question": instruction,
                "messages": messages,
            },
            "reference": {
                "answer": expected,
                "turns": [{"role": "assistant", "content": expected}],
                "task_fulfillment_rubric": task_rubric,
                "response_quality_rubrics": rubrics,
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
            "offline_proxy": "baseline_text_similarity_without_streaming_stop_latency",
        }
        self._bind_protocol(row, "spoken_agentic_interaction", "interruption")
        return row
