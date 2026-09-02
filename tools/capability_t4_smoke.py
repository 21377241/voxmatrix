#!/usr/bin/env python3
"""Reproducible T4 (spoken-agent) sampling, inference and audit helpers.

The task specification asks for ten samples for every ``(capability,
benchmark)`` pair.  The native benchmark files are intentionally kept outside
this repository, so this script only writes a small, auditable manifest and
JSONL predictions under a caller supplied work directory.  It never edits the
native files or the formal subset manifest.

Typical use (on the shared AFS):

    python tools/capability_t4_smoke.py prepare --work-root /path/to/run
    python tools/capability_t4_smoke.py infer --work-root /path/to/run \
        --gpu-count 2
    python tools/capability_t4_smoke.py evaluate --work-root /path/to/run

``infer`` is normally launched through ``job`` (the model is too large for a
login node).  It uses the same Qwen3-Omni checkpoint and adapter contract as
the add-capbility branch, with a strict acknowledgement-safe worker.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


MANIFEST_PATH = Path(
    "/mnt/afs/users/wangyl/benchmark_annotation_audit/native_v2_pipeline/outputs/"
    "benchmark_subset_manifest.jsonl"
)
QWEN_CHECKPOINT = Path("/mnt/afs/models/Qwen3-Omni-30B-A3B-Instruct")
QWEN_ENV = Path("/mnt/afs/users/wangyl/VoxMatrix/envs/qwen3-omni")

ROOTS = {
    "fsc": Path("/mnt/afs/eval_data/04_dialogue_slu/fluent_speech_commands_dataset"),
    "slurp": Path("/mnt/afs/oss_data/datasets/04_dialogue_slu/SLURP_repo"),
    "vocalbench": Path("/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench"),
    "vocalbench_zh": Path(
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/VocalBench-zh"
    ),
    "voicebench": Path(
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/AudioBench/VoiceBench"
    ),
    "audioagent": Path("/mnt/afs/eval_data/benchmarks/AudioAgentBench"),
    "mtalk": Path("/mnt/afs/eval_data/benchmarks/MTalk-Bench"),
    "stepeval": Path("/mnt/afs/eval_data/benchmarks/StepEval-Audio-Toolcall"),
    "ear": Path("/mnt/afs/eval_data/benchmarks/EAR"),
    "ihbench": Path(
        "/mnt/afs/eval_data/99_auxiliary_or_out_of_scope/IHBench"
    ),
}

TARGETS = {
    "instruction_following": [
        "fluent_speech_commands",
        "slurp",
        "vocalbench",
        "vocalbench_zh",
        "voicebench",
    ],
    "tool_call": [
        "audioagentbench_suite",
        "fluent_speech_commands",
        "slurp",
        "stepeval_audio_toolcall",
    ],
    "multi_turn_dialogue": [
        "audioagentbench_suite",
        "mtalk_bench",
        "vocalbench",
        "vocalbench_zh",
        "voicebench",
    ],
    "clarification": ["audioagentbench_suite", "ear_wdyl"],
    "interruption": ["ihbench"],
}


def _json_default(value: Any) -> Any:
    """Keep numpy/Arrow metadata JSON-native in manifests and audit logs."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes_hex": bytes(value).hex()}
    return str(value)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_json_dump(dict(row)) + "\n")
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _short_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _even_indices(size: int, count: int = 10) -> list[int]:
    if size <= 0:
        return []
    if size <= count:
        return list(range(size))
    if count == 1:
        return [0]
    # Include both ends and make the selection independent of hash/random state.
    return sorted({int(round(i * (size - 1) / (count - 1))) for i in range(count)})


def _stratified(rows: Sequence[dict[str, Any]], key, count: int = 10) -> list[dict[str, Any]]:
    """Select a deterministic, near-even sample across strata."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    if len(groups) <= 1:
        return [rows[i] for i in _even_indices(len(rows), count)]
    names = sorted(groups)
    selected: list[dict[str, Any]] = []
    # One per stratum first, then fill by proportional round-robin.
    cursors = {name: 0 for name in names}
    for name in names:
        if groups[name]:
            selected.append(groups[name][0])
            cursors[name] = 1
    while len(selected) < count:
        candidates = [name for name in names if cursors[name] < len(groups[name])]
        if not candidates:
            break
        name = max(
            candidates,
            key=lambda n: (len(groups[n]) / max(1, cursors[n]), -names.index(n)),
        )
        selected.append(groups[name][cursors[name]])
        cursors[name] += 1
    # Preserve native order for easy manual lookup, while deduplicating rows.
    wanted = {id(row) for row in selected}
    return [row for row in rows if id(row) in wanted][:count]


def _audio_descriptor(path: str) -> dict[str, Any]:
    p = Path(path)
    result = {"path": str(p), "exists": p.is_file()}
    if p.is_file():
        result["bytes"] = p.stat().st_size
        try:
            import soundfile as sf

            info = sf.info(str(p))
            result.update(
                {
                    "samplerate": int(info.samplerate),
                    "channels": int(info.channels),
                    "frames": int(info.frames),
                    "duration_seconds": round(info.frames / info.samplerate, 4),
                }
            )
        except Exception:
            pass
    return result


def _find_audio(root: Path, name: str) -> Path:
    candidate = root / name
    if candidate.is_file():
        return candidate
    matches = list(root.rglob(Path(name).name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"audio not found under {root}: {name}")


def _content(audio: str, text: str = "") -> list[dict[str, str]]:
    result = [{"type": "audio", "value": audio}]
    if text:
        result.append({"type": "text", "value": text})
    return result


def _message(role: str, audio: str = "", text: str = "") -> dict[str, Any]:
    contents: list[dict[str, str]] = []
    if audio:
        contents.append({"type": "audio", "value": audio})
    if text:
        contents.append({"type": "text", "value": text})
    return {"role": role, "contents": contents}


def _base_sample(
    *,
    capability: str,
    benchmark: str,
    subset: str,
    native_id: str,
    audio_paths: Sequence[str],
    prompt: list[dict[str, Any]],
    reference: Any,
    expected: Any,
    metric: str,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sample_id = f"t4_{capability}_{benchmark}_{_short_id(capability, benchmark, subset, native_id)}"
    return {
        "sample_id": sample_id,
        "capability": capability,
        "benchmark": benchmark,
        "subset": subset,
        "native_id": str(native_id),
        "audio_paths": [str(x) for x in audio_paths],
        "audio": [_audio_descriptor(str(x)) for x in audio_paths],
        "prompt": prompt,
        "reference": reference,
        "expected": expected,
        "metric": metric,
        "meta": dict(meta or {}),
    }


def _load_fsc(capability: str) -> list[dict[str, Any]]:
    path = ROOTS["fsc"] / "data/test_data.csv"
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    schema = {
        "name": "control_device",
        "description": "Control a device or setting requested by the speaker.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "object": {"type": "string"},
                "location": {"type": "string"},
            },
            "required": ["action", "object", "location"],
        },
    }
    if capability == "instruction_following":
        system = (
            "You are a spoken assistant. Listen to the audio and identify the requested "
            "action. Return JSON only as {\"intent\": string, \"slots\": object}; "
            "do not add explanation. This is a diagnostic SLU proxy for instruction following."
        )
        metric = "intent_accuracy+slot_f1 (diagnostic proxy)"
    else:
        system = (
            "You are a tool-calling assistant. Listen to the audio and call the supplied "
            "function. Return JSON only as {\"tool\": string, \"arguments\": object}; "
            "never invent a tool.\nTool schema:\n" + _json_dump(schema)
        )
        metric = "tool_exact_match (diagnostic proxy)"
    for row in (rows[i] for i in _even_indices(len(rows), 10)):
        audio = ROOTS["fsc"] / row["path"]
        intent = f"{row['action']}_{row['object']}"
        slots = {
            "action": row["action"],
            "object": row["object"],
            "location": row["location"],
        }
        if capability == "instruction_following":
            expected = {"intent": intent, "slots": slots}
        else:
            expected = {"tool": "control_device", "arguments": slots}
        prompt = [_message("system", text=system), _message("user", audio=str(audio))]
        result.append(
            _base_sample(
                capability=capability,
                benchmark="fluent_speech_commands",
                subset="test",
                native_id=row["path"],
                audio_paths=[str(audio)],
                prompt=prompt,
                reference={"transcription": row["transcription"], "intent": intent, "slots": slots},
                expected=expected,
                metric=metric,
                meta={"native": row},
            )
        )
    return result


def _slurp_slots(row: Mapping[str, Any]) -> dict[str, Any]:
    tokens = row.get("tokens") or []
    entities = row.get("entities") or []
    surfaces = [str(item.get("surface", "")) for item in tokens]
    slots: dict[str, str] = {}
    for entity in entities:
        span = entity.get("span") or []
        values = [surfaces[int(i)] for i in span if int(i) < len(surfaces)]
        slots[str(entity.get("type"))] = " ".join(values)
    return slots


def _load_slurp(capability: str) -> list[dict[str, Any]]:
    path = ROOTS["slurp"] / "dataset/slurp/test.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    schema = {
        "name": "execute_slurp_intent",
        "description": "Execute the spoken SLURP intent.",
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string"},
                "scenario": {"type": "string"},
                "slots": {"type": "object"},
            },
            "required": ["intent", "scenario", "slots"],
        },
    }
    if capability == "instruction_following":
        system = (
            "Listen to the spoken instruction. Return JSON only with intent and slots; "
            "do not explain. This SLURP intent/slot score is a diagnostic proxy, not a "
            "general instruction-following score."
        )
        metric = "intent_accuracy+slot_f1 (diagnostic proxy)"
    else:
        system = (
            "Listen to the spoken request and invoke the function below. Return JSON only "
            "as {\"tool\": string, \"arguments\": object}; no prose.\nTool schema:\n"
            + _json_dump(schema)
        )
        metric = "tool_exact_match (diagnostic proxy)"
    result = []
    for row in (rows[i] for i in _even_indices(len(rows), 10)):
        recordings = row.get("recordings") or []
        # Alternate headset/non-headset when possible to avoid selecting one file type.
        recording = recordings[0]
        if len(recordings) > 1 and int(row.get("slurp_id", 0)) % 2:
            recording = recordings[-1]
        audio = _find_audio(ROOTS["slurp"], str(recording["file"]))
        slots = _slurp_slots(row)
        if capability == "instruction_following":
            expected = {"intent": row["intent"], "slots": slots}
        else:
            expected = {
                "tool": "execute_slurp_intent",
                "arguments": {"intent": row["intent"], "scenario": row["scenario"], "slots": slots},
            }
        prompt = [_message("system", text=system), _message("user", audio=str(audio))]
        result.append(
            _base_sample(
                capability=capability,
                benchmark="slurp",
                subset="test",
                native_id=str(row["slurp_id"]),
                audio_paths=[str(audio)],
                prompt=prompt,
                reference={"sentence": row["sentence"], "intent": row["intent"], "scenario": row["scenario"], "slots": slots},
                expected=expected,
                metric=metric,
                meta={"recording": recording, "native": {k: row.get(k) for k in ("slurp_id", "action", "scenario")}},
            )
        )
    return result


def _vocal_constraint(question: str) -> dict[str, Any]:
    q = question.strip()
    lower = q.lower()
    constraints: dict[str, Any] = {}
    if "repeat" in lower or "重复" in q or "朗诵" in q or "朗读" in q:
        # The VocalBench prompts use both an English colon and Chinese full
        # stops to introduce the text to repeat.  Keep the target as metadata
        # for a token-overlap audit; emotion/speed remain audio-output
        # requirements and are intentionally marked unsupported below.
        tail = re.search(
            r"(?:question|问题|句子|这句诗|下面这句话|下面说的问题)\s*[：:]\s*(.*)$",
            q,
            flags=re.I | re.S,
        )
        if tail:
            constraints["repeat_target"] = tail.group(1).strip().strip('"“”')
        else:
            # Chinese plain-repeat form: ``……这句诗。<target>``.
            tail = re.search(
                r"(?:这句诗|下面这句话|下面说的问题)[。．]\s*(.+)$",
                q,
                flags=re.S,
            )
            if tail:
                constraints["repeat_target"] = tail.group(1).strip().strip('"“”')
            else:
                tail = re.search(
                    r"(?:without any explanation|不要回答|不要解释|angry emotion|half my speaking speed|"
                    r"twice my speaking speed|normal tone)[。.!！?？]\s*(.+)$",
                    q,
                    flags=re.I | re.S,
                )
                if tail:
                    constraints["repeat_target"] = tail.group(1).strip().strip('"“”')
    m = re.search(r"(?:at least|至少)\s*(\d+)\s*(?:lines?|行)", lower + q.lower())
    if m:
        constraints["min_lines"] = int(m.group(1))
    m = re.search(r"(?:exactly|正好)\s*(\d+)\s*(?:lines?|行)", lower + q.lower())
    if m:
        constraints["exact_lines"] = int(m.group(1))
    m = re.search(r"(?:fewer than|less than|少于)\s*(\d+)\s*(?:words?|字)", lower + q.lower())
    if m:
        constraints["max_words"] = int(m.group(1))
    if "no comma" in lower or "不要使用逗号" in q or "不使用逗号" in q:
        constraints["no_comma"] = True
    if "lowercase" in lower or "小写" in q:
        constraints["lowercase"] = True
    if "bullet" in lower or "项目符号" in q:
        constraints["bullets"] = True
    m = re.search(
        r"(?:give me|name|list|提供|给我|划分)\s*(?:exactly\s*)?"
        r"(?:five|5|三|四|五|六|七|八|九|十|\d+)\s*"
        r"(?:suggestions?|names?|categories?|items?|个类别|个类别|项|条)",
        q,
        flags=re.I,
    )
    if m:
        number_words = {
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
            "七": 7,
            "八": 8,
            "九": 9,
            "十": 10,
        }
        token = re.search(
            r"(five|5|三|四|五|六|七|八|九|十|\d+)", m.group(0), re.I
        ).group(1)
        if token.isdigit():
            constraints["exact_items"] = int(token)
        else:
            constraints["exact_items"] = number_words.get(token, 5)
    m = re.search(
        r"(?:at least|至少)\s*(?:要\s*)?(?:出现\s*)?"
        r"(\d+|one|two|three|four|five|六|三|四|五)\s*(?:times?|次)",
        lower + q.lower(),
    )
    if m:
        # The keyword itself is extracted from the nearby quoted phrase.
        quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", q)
        count_token = m.group(1).lower()
        count_words = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "三": 3,
            "四": 4,
            "五": 5,
            "六": 6,
        }
        constraints["keyword_frequency"] = {
            "keyword": quoted[0] if quoted else None,
            "minimum": int(count_token) if count_token.isdigit() else count_words.get(count_token, 1),
        }
    m = re.search(r"first sentence should start with\s*[\"“]?([^\"”]+)", q, re.I)
    if m:
        constraints["starts_with"] = m.group(1).strip()
    m = re.search(r"第一(?:句话|句).*?(?:以|从)\s*[\"“]?([^\"”]+)", q)
    if m:
        constraints["starts_with"] = m.group(1).strip()
    excluded = re.search(
        r"(?:exclude|excluding|do not use|排除|不要).*?[\"“]([^\"”]+)[\"”]",
        q,
        re.I,
    )
    if excluded:
        constraints["forbidden_keywords"] = [x.strip() for x in re.split(r"[,，、]| and | 和 ", excluded.group(1)) if x.strip()]
    included = re.search(
        r"(?:include|包含).*?[\"“]([^\"”]+)[\"”]",
        q,
        re.I,
    )
    if included:
        constraints["required_keywords"] = [x.strip() for x in re.split(r"[,，、]| and | 和 ", included.group(1)) if x.strip()]
    # Chinese VocalBench-zh uses unquoted keyword lists.
    m = re.search(r"排除以下关键词\s*[：:]\s*([^。；;]+)", q)
    if m:
        constraints["forbidden_keywords"] = [
            x.strip() for x in re.split(r"[,，、]|和", m.group(1)) if x.strip()
        ]
    m = re.search(r"包含关键词\s*[：:]\s*([^。；;]+)", q)
    if m:
        constraints["required_keywords"] = [
            x.strip() for x in re.split(r"[,，、]|和", m.group(1)) if x.strip()
        ]
    # English VocalBench content prompts often leave the keyword list
    # unquoted (``exclude ... temple, Tokyo ... include ...``).
    m = re.search(
        r"(?:exclude|excluding|do not use).*?(?:keywords?)?\s+(.+?)\s*\.\s*"
        r"(?:include|包含)",
        q,
        re.I | re.S,
    )
    if m:
        constraints["forbidden_keywords"] = [
            re.sub(r"^(?:the\s+)?keywords?\s+", "", x.strip(), flags=re.I)
            for x in re.split(r",| and |、|，|和", m.group(1))
            if x.strip()
        ]
    m = re.search(
        r"(?:include|包含).*?(?:keywords?)?\s+(.+?)(?:\.|$)",
        q,
        re.I | re.S,
    )
    if m and "required_keywords" not in constraints:
        constraints["required_keywords"] = [
            re.sub(r"^(?:the\s+)?keywords?\s+", "", x.strip(), flags=re.I)
            for x in re.split(r",| and |、|，|和", m.group(1))
            if x.strip()
        ]
    if "without any explanation" in lower or "不要回答" in q or "不要解释" in q:
        constraints["no_explanation"] = True
    if "angry emotion" in lower or "伤心" in q or "生气" in q or "angry" in lower:
        constraints["unsupported_emotion"] = True
    if "speed" in lower or "语速" in q:
        constraints["unsupported_speed"] = True
    if re.search(r"(?:if yes|if no|如果是|如果不是|若是|若不是)", q, re.I):
        constraints["unsupported_conditional"] = True
    if re.search(r"(?:then|next|finally|接下来|然后|最后|步骤)", q, re.I):
        constraints["unsupported_multistep"] = True
    if re.search(r"(?:apology|apologize|道歉)", q, re.I):
        constraints["required_mode"] = "apology"
    return constraints


def _load_vocalbench(capability: str, zh: bool = False) -> list[dict[str, Any]]:
    key = "vocalbench_zh" if zh else "vocalbench"
    root = ROOTS[key]
    filename = "instruction_following.json" if capability == "instruction_following" else "multi_round.json"
    rows = json.loads((root / "json" / filename).read_text(encoding="utf-8"))
    if capability == "instruction_following":
        rows = _stratified(rows, lambda x: x.get("Sub-category") or x.get("Category") or "unknown")
        result = []
        system = (
            "Listen to the audio and follow the spoken instruction exactly. Return only "
            "the requested result, with no discussion of this evaluation prompt."
        )
        for row in rows:
            audio = root / "audio" / str(row["Audio"])
            constraints = _vocal_constraint(str(row.get("Question") or ""))
            result.append(
                _base_sample(
                    capability=capability,
                    benchmark=key,
                    subset="instruction_following",
                    native_id=row["Qid"],
                    audio_paths=[str(audio)],
                    prompt=[_message("system", text=system), _message("user", audio=str(audio))],
                    reference=None,
                    expected={"constraints": constraints},
                    metric="constraint_satisfaction (local audit; official LLM judge unavailable)",
                    meta={"question": row.get("Question"), "category": row.get("Category"), "sub_category": row.get("Sub-category"), "source": row.get("Source")},
                )
            )
        return result

    rows = _even_indices(len(rows), 10)
    result = []
    system = (
        "You are a conversational voice assistant. Use all prior turns and answer the "
        "latest spoken turn. Return only the answer; preserve the language of the user."
    )
    all_rows = json.loads((root / "json" / filename).read_text(encoding="utf-8"))
    for idx in rows:
        row = all_rows[idx]
        messages = [_message("system", text=system)]
        for turn in row.get("Context") or []:
            role = "assistant" if str(turn.get("from")).lower() == "assistant" else "user"
            messages.append(_message(role, text=str(turn.get("value") or "")))
        audio = root / "audio" / str(row["Audio"])
        messages.append(_message("user", audio=str(audio)))
        result.append(
            _base_sample(
                capability=capability,
                benchmark=key,
                subset="multi_round",
                native_id=row["Qid"],
                audio_paths=[str(audio)],
                prompt=messages,
                reference={"answer": row.get("Answer")},
                expected={"answer": row.get("Answer")},
                metric="rouge_l+token_f1 (diagnostic; official judge not reproduced)",
                meta={"question": row.get("Question"), "category": row.get("Category"), "source": row.get("Source"), "context_turns": len(row.get("Context") or [])},
            )
        )
    return result


def _bytes_from_audio(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, dict):
        return _bytes_from_audio(value.get("bytes"))
    if isinstance(value, str):
        # Pandas occasionally stringifies an Arrow bytes scalar.
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (bytes, bytearray)):
                return bytes(parsed)
        except Exception:
            pass
    raise TypeError(f"audio value does not contain bytes: {type(value).__name__}")


def _materialize_audio(value: Any, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _bytes_from_audio(value)
    if not destination.exists() or destination.stat().st_size != len(payload):
        destination.write_bytes(payload)
    return destination


def _load_voicebench_ifeval(work: Path) -> list[dict[str, Any]]:
    import pandas as pd

    path = ROOTS["voicebench"] / "ifeval/test-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    selected = [rows[i] for i in _even_indices(len(rows), 10)]
    result = []
    for row in selected:
        key = str(row["key"])
        audio = _materialize_audio(row["audio"], work / "audio" / "voicebench_ifeval" / f"{key}.wav")
        ids = row.get("instruction_id_list")
        kwargs = row.get("kwargs")
        if isinstance(ids, str):
            try:
                ids = ast.literal_eval(ids)
            except Exception:
                # NumPy's array string representation separates quoted items
                # with spaces (``['a' 'b']``), which is not valid Python
                # syntax.  Recover each quoted instruction id explicitly.
                quoted_ids = re.findall(r"['\"]([^'\"]+)['\"]", ids)
                ids = quoted_ids or [ids]
        if isinstance(kwargs, str):
            try:
                kwargs = ast.literal_eval(kwargs)
            except Exception:
                kwargs = []
        result.append(
            _base_sample(
                capability="instruction_following",
                benchmark="voicebench",
                subset="ifeval",
                native_id=key,
                audio_paths=[str(audio)],
                prompt=[
                    _message(
                        "system",
                        text=(
                            "Listen to the audio instruction and satisfy every explicit "
                            "format/content constraint. Return only the requested answer."
                        ),
                    ),
                    _message("user", audio=str(audio)),
                ],
                reference=None,
                expected={"instruction_ids": ids, "kwargs": kwargs, "prompt": row.get("prompt")},
                metric="IFEval constraint_pass_rate (official checker-compatible subset)",
                meta={"prompt": row.get("prompt"), "instruction_ids": ids, "kwargs": kwargs},
            )
        )
    return result


def _load_voicebench_mt(work: Path) -> list[dict[str, Any]]:
    import pandas as pd

    path = ROOTS["voicebench"] / "mtbench/test-00000-of-00001.parquet"
    rows = pd.read_parquet(path).to_dict("records")
    selected = [rows[i] for i in _even_indices(len(rows), 10)]
    result = []
    for row in selected:
        qid = str(row["question_id"])
        a1 = _materialize_audio(row["audio1"], work / "audio" / "voicebench_mtbench" / f"{qid}-audio1.wav")
        a2 = _materialize_audio(row["audio2"], work / "audio" / "voicebench_mtbench" / f"{qid}-audio2.wav")
        turns = row.get("turns")
        if hasattr(turns, "tolist"):
            turns = turns.tolist()
        if isinstance(turns, str):
            try:
                turns = ast.literal_eval(turns)
            except Exception:
                turns = [turns]
        turns = [str(x) for x in (turns or [])]
        messages = [
            _message("system", text="Use both audio turns and maintain the dialogue state. Answer the latest turn only."),
            # ``turns`` is the native transcript used by the judge.  Do not
            # place it beside the waveform: doing so would leak the answer
            # and turn this into text-conditioned QA rather than spoken
            # dialogue.  Keep the transcript in ``meta`` for audit only.
            _message("user", audio=str(a1)),
            _message("assistant", text="(previous response is represented by the first audio turn; do not repeat it)"),
            _message("user", audio=str(a2)),
        ]
        result.append(
            _base_sample(
                capability="multi_turn_dialogue",
                benchmark="voicebench",
                subset="mtbench",
                native_id=qid,
                audio_paths=[str(a1), str(a2)],
                prompt=messages,
                reference={"reference": row.get("reference")},
                expected={"turns": turns},
                metric="LLM judge (not independently computable: reference is null)",
                meta={"category": row.get("category"), "turns": turns},
            )
        )
    return result


def _audioagent_tools(root: Path, bench: str) -> list[dict[str, Any]]:
    path = root / bench / "benchmark/tool_schemas.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _audioagent_rows(view: str) -> list[dict[str, Any]]:
    filename = {"multi_turn_dialogue": "total_turns.json", "tool_call": "tool_call_turns.json", "clarification": "clarification_turns.json"}[view]
    return json.loads((ROOTS["audioagent"] / filename).read_text(encoding="utf-8"))


def _audioagent_turns(bench: str) -> list[dict[str, Any]]:
    return json.loads((ROOTS["audioagent"] / bench / "benchmark/turns.json").read_text(encoding="utf-8"))


def _load_audioagent(capability: str) -> list[dict[str, Any]]:
    view = {"multi_turn_dialogue": "multi_turn_dialogue", "tool_call": "tool_call", "clarification": "clarification"}[capability]
    selectors = _audioagent_rows(view)
    selected = [selectors[i] for i in _even_indices(len(selectors), 10)]
    result = []
    for selector in selected:
        bench = str(selector["bench"])
        turns = _audioagent_turns(bench)
        by_id = {int(row["turn_id"]): row for row in turns}
        target_id = int(selector["turn_id"])
        target = by_id[target_id]
        metadata = []
        for row in turns:
            if int(row["turn_id"]) <= target_id:
                metadata.append(row)
        # Preserve the preceding conversation, but cap it to avoid unbounded prompts.
        history_rows = metadata[:-1][-16:]
        messages = [
            _message(
                "system",
                text=(
                    "You are a spoken conference/home assistant. Use the dialogue history "
                    "and answer the latest turn."
                ),
            )
        ]
        if capability == "tool_call":
            tools = _audioagent_tools(ROOTS["audioagent"], bench)
            messages[0]["contents"][0]["value"] += (
                " Return JSON only as {\"tool\": string, \"arguments\": object}; "
                "if no tool is appropriate return {\"tool\": null, \"arguments\": {}}.\n"
                "Complete tool schemas:\n" + _json_dump(tools)
            )
        elif capability == "clarification":
            messages[0]["contents"][0]["value"] += (
                " If a required entity is ambiguous or missing, ask one targeted "
                "clarifying question before acting."
            )
        for row in history_rows:
            audio = ROOTS["audioagent"] / bench / str(row.get("file_name") or row.get("audio_file") or "")
            if not audio.is_file():
                audio = ROOTS["audioagent"] / bench / str(row.get("audio_file") or f"audio/turn_{int(row['turn_id']):03d}.wav")
            # ``input_text`` is a native transcript/reference.  Audio is the
            # model input; retain the transcript only in the sample metadata.
            messages.append(_message("user", audio=str(audio)))
            messages.append(_message("assistant", text=str(row.get("golden_text") or "")))
        current = ROOTS["audioagent"] / bench / str(selector.get("audio_file") or target.get("audio_file") or f"audio/turn_{target_id:03d}.wav")
        expected_call = target.get("required_function_call")
        if capability == "tool_call":
            expected = expected_call
        elif capability == "clarification":
            golden_text = str(target.get("golden_text") or "")
            # The native clarification view mixes true disambiguation turns
            # with answerable follow-up/tool turns.  Use the native golden
            # response's explicit question as a *diagnostic* target; it is
            # not presented as an official clarification label.
            should_clarify = bool(
                re.search(
                    r"\?|\bwhich\b|\bwhat would you like\b|\bwould you like\b|"
                    r"please tell me|just to confirm|which one",
                    golden_text,
                    re.I,
                )
            )
            expected = {"answer": golden_text, "should_clarify": should_clarify}
        else:
            expected = {"answer": target.get("golden_text")}
        messages.append(_message("user", audio=str(current)))
        result.append(
            _base_sample(
                capability=capability,
                benchmark="audioagentbench_suite",
                subset=view,
                native_id=f"{bench}:{target_id}",
                audio_paths=[str(current)] + [str(x.get("audio_file")) for x in []],
                prompt=messages,
                reference={"input_text": target.get("input_text"), "golden_text": target.get("golden_text"), "required_function_call": expected_call},
                expected=expected,
                metric=(
                    "tool_use_success+parameter_f1 (offline expected-call proxy)"
                    if capability == "tool_call"
                    else (
                        "clarification_decision+text_f1 (offline native-golden proxy)"
                        if capability == "clarification"
                        else "text_f1 (offline contextual-turn proxy)"
                    )
                ),
                meta={"bench": bench, "turn_id": target_id, "categories": target.get("categories"), "subcategory": target.get("subcategory")},
            )
        )
    # Ensure paths in the sample's audio list reflect every audio content item.
    for row in result:
        paths = []
        for message in row["prompt"]:
            for item in message.get("contents", []):
                if item.get("type") == "audio":
                    paths.append(item["value"])
        row["audio_paths"] = paths
        row["audio"] = [_audio_descriptor(x) for x in paths]
    return result


def _step_call(text: str) -> Any:
    # Native rows are arrays of turn objects; the last assistant value is the gold call.
    try:
        payload = text.strip()
        match = re.search(r"<tool_call>.*?\n([^\n]+)\n(\{.*?\})</tool_call>", payload, re.S)
        if match:
            return {"name": match.group(1).strip(), "args": json.loads(match.group(2))}
    except Exception:
        pass
    return None


def _step_schemas() -> list[dict[str, Any]]:
    return [
        {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}},
        {"name": "get_date_time", "parameters": {"type": "object", "properties": {}, "required": []}},
        {"name": "web_search", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "timbre_rag", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "calculate", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}},
    ]


def _load_stepeval() -> list[dict[str, Any]]:
    root = ROOTS["stepeval"]
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.jsonl")):
        if path.name.endswith(".jsonl"):
            payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for idx, session in enumerate(payload):
                if not isinstance(session, list) or not session:
                    continue
                final = session[-1]
                if str(final.get("from", "")).lower() != "assistant":
                    continue
                stem = path.stem
                target_tool = stem.rsplit("_", 1)[0]
                rows.append({"path": path, "session_index": idx, "session": session, "target_tool": target_tool, "gold_call": _step_call(str(final.get("text") or final.get("value") or ""))})
    # Cover all four tool families and both positive/negative subsets.
    selected: list[dict[str, Any]] = []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["path"].stem].append(row)
    for name in sorted(groups):
        if groups[name]:
            selected.append(groups[name][0])
    remaining = [row for row in rows if row not in selected]
    selected.extend(remaining[i] for i in _even_indices(len(remaining), max(0, 10 - len(selected))))
    selected = selected[:10]
    result = []
    schemas = _step_schemas()
    for row in selected:
        session = row["session"]
        messages = [_message("system", text=("You are a tool-calling assistant. Use the complete schemas below. " "Return JSON only as {\"tool\": string|null, \"arguments\": object}; " "for a negative/no-call case do not invoke the named target tool.\nSchemas:\n" + _json_dump(schemas)))]
        audio_paths = []
        for turn in session[:-1]:
            value = turn.get("value")
            if isinstance(value, dict) and value.get("type") == "wav":
                audio = root / "Step-Audio_Toolcall" / str(value.get("value"))
                audio_paths.append(str(audio))
                role = "user" if str(turn.get("from", "")).lower() in {"human", "user"} else "assistant"
                messages.append(_message(role, audio=str(audio)))
            elif str(turn.get("from", "")).lower() in {"human", "user"}:
                messages.append(_message("user", text=str(turn.get("text") or "")))
            else:
                messages.append(_message("assistant", text=str(turn.get("text") or "")))
        final_audio = session[-2].get("value") if len(session) >= 2 else None
        if isinstance(final_audio, dict):
            final_path = root / "Step-Audio_Toolcall" / str(final_audio.get("value"))
            audio_paths.append(str(final_path))
            # The final human turn is already in session[:-1]; no duplicate needed.
        expected = {"gold_call": row["gold_call"], "target_tool": row["target_tool"], "polarity": "positive" if "positive" in row["path"].stem else "negative"}
        result.append(_base_sample(capability="tool_call", benchmark="stepeval_audio_toolcall", subset=row["path"].stem, native_id=f"{row['path'].name}:{row['session_index']}", audio_paths=audio_paths, prompt=messages, reference={"session": session, "target_tool": row["target_tool"]}, expected=expected, metric="trigger_precision_recall+type_accuracy+parameter_accuracy (10-row diagnostic)", meta={"source_file": str(row["path"]), "session_id": session[0].get("session_id")}))
    return result


def _load_mtalk(work: Path) -> list[dict[str, Any]]:
    import pandas as pd

    path = ROOTS["mtalk"] / "data/test-00000-of-00001.parquet"
    df = pd.read_parquet(path)
    rows = df.to_dict("records")
    selected = _stratified(rows, lambda x: x.get("type"), 10)
    # Materialize all rows needed as history; ``number`` is reused across the
    # ambient/paralinguistic/semantic variants, so the condition ``type`` is
    # part of the conversation key.  Grouping by number alone would splice
    # unrelated variants into one dialogue and make the latest turn
    # ambiguous.
    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[(str(row.get("number")), str(row.get("type")))].append((idx, row))
    result = []
    # ``id`` is absent in the released MTalk parquet and several records have
    # identical dictionaries.  Resolve the selected object by identity (not
    # equality) and include its source row index in ``native_id`` so every
    # sampled turn remains addressable and gets a unique sample_id.
    row_indices_by_identity = {id(row): idx for idx, row in enumerate(rows)}
    for idx, row in enumerate(selected):
        row_index = row_indices_by_identity[id(row)]
        number = str(row.get("number"))
        type_name = str(row.get("type"))
        sequence = sorted(
            grouped[(number, type_name)], key=lambda x: str(x[1].get("turn"))
        )
        current_pos = next(i for i, (j, _) in enumerate(sequence) if j == row_index)
        messages = [
            _message(
                "system",
                text=(
                    "Use the preceding user audio turns as dialogue context and answer "
                    "the latest spoken turn. The native sample has no recorded assistant "
                    "turns; do not invent or repeat a placeholder. Return only your response."
                ),
            )
        ]
        audio_paths = []
        for hist_idx, hist in sequence[:current_pos]:
            audio = _materialize_audio(hist["audio"], work / "audio" / "mtalk" / f"{hist_idx}.wav")
            audio_paths.append(str(audio))
            # MTalk's transcription is for rubric/judge alignment, not an
            # additional model input.  Supplying it would make the audio
            # modality unverifiable.
            messages.append(_message("user", audio=str(audio)))
        audio = _materialize_audio(row["audio"], work / "audio" / "mtalk" / f"{row_index}.wav")
        audio_paths.append(str(audio))
        messages.append(_message("user", audio=str(audio)))
        native_id = f"{number}:row{row_index}:turn{row.get('turn')}"
        result.append(_base_sample(capability="multi_turn_dialogue", benchmark="mtalk_bench", subset=str(row.get("type")), native_id=native_id, audio_paths=audio_paths, prompt=messages, reference=None, expected={"transcription": row.get("transcription")}, metric="LLM judge+ELO (not computable from a single model; rubric retained)", meta={"type": row.get("type"), "type_full": row.get("type_full"), "number": number, "turn": row.get("turn"), "source_row_index": row_index, "rubric_prompt_general": row.get("rubric_prompt_general"), "rubric_prompt_specific": row.get("rubric_prompt_specific")}))
    return result


def _load_ear() -> list[dict[str, Any]]:
    path = ROOTS["ear"] / "manifests/ear_wdyl.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = _stratified(rows, lambda x: x.get("condition"), 10)
    result = []
    system = (
        "Listen carefully. If the requested key information is unavailable because of "
        "noise, say what is missing and ask one targeted clarification. If it is available, "
        "answer directly without guessing."
    )
    for row in selected:
        audio = Path(str(row["audio"]))
        result.append(_base_sample(capability="clarification", benchmark="ear_wdyl", subset=str(row.get("condition")), native_id=str(row["id"]), audio_paths=[str(audio)], prompt=[_message("system", text=system), _message("user", audio=str(audio))], reference=None, expected={"answerable": bool(row.get("answerable")), "masked_word": row.get("masked_word")}, metric="clarification_decision (方案待定; official question/gold/judge absent)", meta={k: row.get(k) for k in ("condition", "masked_word", "language", "task_id_clarification")}))
    return result


def _load_ihbench(work: Path) -> list[dict[str, Any]]:
    import pandas as pd

    root = ROOTS["ihbench"]
    selectors = json.loads((root / "interruptions_all.json").read_text(encoding="utf-8"))
    selected = [selectors[i] for i in _even_indices(len(selectors), 10)]
    conversations = pd.read_parquet(root / "conversations.parquet").set_index("conversation_id")
    baseline = pd.read_parquet(root / "baseline.parquet")
    baseline_map = {(str(r.conversation_id), int(r.turn)): str(r.response) for r in baseline.itertuples()}
    result = []
    for selector in selected:
        cid = str(selector["conversation_id"])
        turn = int(selector["interrupting_turn_index"])
        row = conversations.loc[cid]
        messages = [_message("system", text=str(row.get("system_message") or ""))]
        audio_paths = []
        # Include assistant/user turns through the interruption. The next assistant response is generated.
        for t in range(1, turn + 1):
            assistant = row.get(f"assistant_turn_{t}_transcript")
            if assistant:
                messages.append(_message("assistant", text=str(assistant)))
            audio_value = row.get(f"user_turn_{t}_audio")
            transcript = row.get(f"user_turn_{t}_transcript")
            if isinstance(audio_value, dict):
                audio = _materialize_audio(audio_value, work / "audio" / "ihbench" / f"{cid}-{t}.wav")
                audio_paths.append(str(audio))
                # The transcript is retained in the parquet for scoring, but
                # must not be sent next to the waveform during the smoke.
                messages.append(_message("user", audio=str(audio)))
            elif transcript:
                messages.append(_message("user", text=str(transcript)))
        rubrics = row.get(f"turn_{turn}_rq_rubrics")
        if hasattr(rubrics, "tolist"):
            rubrics = rubrics.tolist()
        result.append(_base_sample(capability="interruption", benchmark="ihbench", subset="all", native_id=f"{cid}:{turn}", audio_paths=audio_paths, prompt=messages, reference={"baseline": baseline_map.get((cid, turn)), "tf_rubric": row.get(f"turn_{turn}_tf_rubric"), "rq_rubrics": rubrics}, expected={"interruption_type": selector.get("interruption_type"), "tf_rubric": row.get(f"turn_{turn}_tf_rubric"), "rq_rubrics": rubrics}, metric="tf_win_rate+rq_pass_rate (offline recovery proxy; no real-time latency)", meta={"conversation_id": cid, "turn": turn, "domain": row.get("domain"), "goal": row.get("goal"), "interruption_type": selector.get("interruption_type")}))
    return result


def build_manifest(work: Path) -> list[dict[str, Any]]:
    builders: dict[tuple[str, str], Any] = {
        ("instruction_following", "fluent_speech_commands"): lambda: _load_fsc("instruction_following"),
        ("instruction_following", "slurp"): lambda: _load_slurp("instruction_following"),
        ("instruction_following", "vocalbench"): lambda: _load_vocalbench("instruction_following"),
        ("instruction_following", "vocalbench_zh"): lambda: _load_vocalbench("instruction_following", zh=True),
        ("instruction_following", "voicebench"): lambda: _load_voicebench_ifeval(work),
        ("tool_call", "audioagentbench_suite"): lambda: _load_audioagent("tool_call"),
        ("tool_call", "fluent_speech_commands"): lambda: _load_fsc("tool_call"),
        ("tool_call", "slurp"): lambda: _load_slurp("tool_call"),
        ("tool_call", "stepeval_audio_toolcall"): _load_stepeval,
        ("multi_turn_dialogue", "audioagentbench_suite"): lambda: _load_audioagent("multi_turn_dialogue"),
        ("multi_turn_dialogue", "mtalk_bench"): lambda: _load_mtalk(work),
        ("multi_turn_dialogue", "vocalbench"): lambda: _load_vocalbench("multi_turn_dialogue"),
        ("multi_turn_dialogue", "vocalbench_zh"): lambda: _load_vocalbench("multi_turn_dialogue", zh=True),
        ("multi_turn_dialogue", "voicebench"): lambda: _load_voicebench_mt(work),
        ("clarification", "audioagentbench_suite"): lambda: _load_audioagent("clarification"),
        ("clarification", "ear_wdyl"): _load_ear,
        ("interruption", "ihbench"): lambda: _load_ihbench(work),
    }
    rows: list[dict[str, Any]] = []
    for capability, benchmarks in TARGETS.items():
        for benchmark in benchmarks:
            key = (capability, benchmark)
            if key not in builders:
                raise KeyError(key)
            selected = builders[key]()
            if len(selected) != 10:
                raise RuntimeError(f"{key} selected {len(selected)} rows, expected 10")
            for ordinal, row in enumerate(selected):
                row["ordinal"] = ordinal
                row["source_manifest"] = str(MANIFEST_PATH)
                row["source_manifest_sha256"] = _sha256(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None
                for descriptor in row["audio"]:
                    if not descriptor["exists"]:
                        raise FileNotFoundError(f"{key} sample {row['sample_id']} audio missing: {descriptor['path']}")
            rows.extend(selected)
    return rows


def _parse_json_output(text: str) -> Any:
    text = str(text or "").strip()
    # Qwen tool-call adapters may emit the native XML-ish form even when the
    # prompt requests JSON (``<tool_call>function\nname\n{...}</tool_call>``).
    # Normalize that form before scoring so a serialization choice is not
    # mistaken for a missed invocation.  Keep the raw prediction unchanged
    # in the audit artifact.
    tagged = re.search(r"<tool_call>(.*?)</tool_call>", text, flags=re.S | re.I)
    if tagged:
        lines = [line.strip() for line in tagged.group(1).splitlines() if line.strip()]
        if lines and lines[0].lower() == "function":
            lines = lines[1:]
        if len(lines) >= 2:
            try:
                return {"tool": lines[0], "arguments": json.loads("\n".join(lines[1:]))}
            except Exception:
                pass
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    # Also recover the first balanced object from verbose model output.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate.strip())
        except Exception:
            continue
    return None


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _f1_tokens(a: Any, b: Any) -> float:
    ta = re.findall(r"\w+", _norm_text(a), flags=re.UNICODE)
    tb = re.findall(r"\w+", _norm_text(b), flags=re.UNICODE)
    if not ta or not tb:
        return 1.0 if ta == tb else 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _rouge_l(a: Any, b: Any) -> float:
    aa = _norm_text(a).split()
    bb = _norm_text(b).split()
    if not aa or not bb:
        return 1.0 if aa == bb else 0.0
    prev = [0] * (len(bb) + 1)
    for x in aa:
        cur = [0]
        for j, y in enumerate(bb, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(aa), lcs / len(bb)
    return 2 * p * r / (p + r) if p + r else 0.0


def _extract_call(value: Any) -> tuple[str | None, dict[str, Any]]:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, dict):
        return None, {}
    tool = value.get("tool") or value.get("name")
    args = value.get("arguments") or value.get("args") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return (str(tool) if tool is not None else None), args if isinstance(args, dict) else {}


def _call_list(value: Any) -> list[tuple[str | None, dict[str, Any]]]:
    """Normalize one call, a list of calls, or a tool_calls wrapper."""
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list):
        value = value["tool_calls"]
    if isinstance(value, list):
        return [_extract_call(item) for item in value]
    return [_extract_call(value)]


def _score(row: Mapping[str, Any], prediction: str) -> dict[str, Any]:
    parsed = _parse_json_output(prediction)
    expected = row.get("expected") or {}
    metric = str(row.get("metric") or "")
    score: dict[str, Any] = {"parsed": parsed, "status": "ok"}
    if "intent_accuracy" in metric:
        got_intent = parsed.get("intent") if isinstance(parsed, dict) else None
        got_slots = parsed.get("slots") if isinstance(parsed, dict) else {}
        exp_slots = expected.get("slots") or {}
        score.update({"intent_correct": int(_norm_text(got_intent) == _norm_text(expected.get("intent"))), "slot_f1": round(_f1_tokens(_json_dump(got_slots), _json_dump(exp_slots)), 4)})
        score["value"] = round((score["intent_correct"] + score["slot_f1"]) / 2, 4)
    elif "tool_exact_match" in metric:
        tool, args = _extract_call(parsed)
        exp_tool, exp_args = _extract_call(expected)
        score.update({"tool_correct": int(tool == exp_tool), "parameter_f1": round(_f1_tokens(_json_dump(args), _json_dump(exp_args)), 4)})
        score["value"] = int(tool == exp_tool and _norm_text(_json_dump(args)) == _norm_text(_json_dump(exp_args)))
    elif "rouge_l" in metric:
        ref = (row.get("reference") or {}).get("answer")
        score["rouge_l"] = round(_rouge_l(prediction, ref), 4)
        score["token_f1"] = round(_f1_tokens(prediction, ref), 4)
        score["value"] = score["rouge_l"]
    elif "IFEval" in metric:
        meta = row.get("meta") or {}
        expected_ifeval = row.get("expected") or {}
        p = str(prediction or "")
        checks = []
        unsupported = []
        ids = meta.get("instruction_ids") or expected_ifeval.get("instruction_ids") or []
        kw = meta.get("kwargs") or expected_ifeval.get("kwargs") or []
        if isinstance(ids, str):
            try:
                ids = ast.literal_eval(ids)
            except Exception:
                ids = [ids]
        if isinstance(kw, dict):
            kw = [kw]
        if isinstance(kw, str):
            try:
                kw = ast.literal_eval(kw)
            except Exception:
                kw = []
        ids = list(ids) if isinstance(ids, (list, tuple)) else [ids]
        kw = list(kw) if isinstance(kw, (list, tuple)) else [kw]
        if len(kw) < len(ids):
            kw.extend({} for _ in range(len(ids) - len(kw)))

        def word_count(text: str) -> int:
            return len(re.findall(r"\b\w+\b", text, flags=re.UNICODE))

        def sentence_count(text: str) -> int:
            return len([x for x in re.split(r"[.!?。！？]+", text) if x.strip()])

        for index, raw_id in enumerate(ids):
            instruction_id = str(raw_id)
            args = kw[index] if index < len(kw) and isinstance(kw[index], dict) else {}
            if instruction_id == "punctuation:no_comma":
                checks.append("," not in p)
            elif instruction_id == "change_case:english_lowercase":
                checks.append(not any(ch.isalpha() and ch.isupper() for ch in p))
            elif instruction_id == "keywords:forbidden_words":
                forbidden = args.get("forbidden_words") or []
                checks.append(not any(str(x).lower() in p.lower() for x in forbidden))
            elif instruction_id == "keywords:existence":
                required = args.get("keywords") or []
                checks.append(all(str(x).lower() in p.lower() for x in required))
            elif instruction_id == "length_constraints:number_words":
                n = args.get("num_words")
                relation = str(args.get("relation") or "")
                if n is None:
                    unsupported.append(instruction_id)
                elif relation == "less than":
                    checks.append(word_count(p) < float(n))
                else:
                    checks.append(word_count(p) >= float(n))
            elif instruction_id == "length_constraints:number_sentences":
                n = args.get("num_sentences")
                relation = str(args.get("relation") or "")
                if n is None:
                    unsupported.append(instruction_id)
                elif relation == "less than":
                    checks.append(sentence_count(p) < float(n))
                else:
                    checks.append(sentence_count(p) >= float(n))
            elif instruction_id == "detectable_format:number_bullet_lists":
                n = args.get("num_bullets")
                if n is None:
                    unsupported.append(instruction_id)
                else:
                    bullets = [x for x in p.splitlines() if re.match(r"\s*(?:[-*•]|\d+[.)])\s+", x)]
                    checks.append(len(bullets) == int(float(n)))
            elif instruction_id == "detectable_format:title":
                checks.append(bool(re.search(r"<<[^<>]+>>", p)))
            elif instruction_id == "detectable_content:number_placeholders":
                n = args.get("num_placeholders")
                checks.append(len(re.findall(r"\[[^\]]+\]", p)) >= int(float(n or 1)))
            elif instruction_id == "length_constraints:nth_paragraph_first_word":
                paragraphs = [x.strip() for x in re.split(r"\n\s*\n", p) if x.strip()]
                n = int(float(args.get("num_paragraphs") or 0))
                first_word = str(args.get("first_word") or "")
                checks.append(
                    len(paragraphs) == n
                    and bool(paragraphs)
                    and paragraphs[0].split()[0].lower().strip("*#\"'") == first_word.lower()
                )
            elif instruction_id == "change_case:capital_word_frequency":
                n = int(float(args.get("capital_frequency") or 0))
                checks.append(len(re.findall(r"\b[A-Z]{2,}\b", p)) >= n)
            elif instruction_id == "detectable_format:number_highlighted_sections":
                n = int(float(args.get("num_highlights") or 0))
                checks.append(len(re.findall(r"\*[^*\n]+\*", p)) >= n)
            elif instruction_id == "startend:quotation":
                stripped = p.strip()
                checks.append(len(stripped) >= 2 and stripped[0] == stripped[-1] == '"')
            elif instruction_id == "detectable_format:constrained_response":
                prompt_text = str(meta.get("prompt") or expected_ifeval.get("prompt") or "")
                choices = re.findall(r"My answer is (?:yes|no|maybe)\.", prompt_text, flags=re.I)
                checks.append(bool(choices) and p.strip().lower() in {x.lower() for x in choices})
            else:
                unsupported.append(instruction_id)
        score["checks"] = checks
        score["unsupported"] = unsupported
        score["supported_check_count"] = len(checks)
        score["value"] = sum(checks) / len(checks) if checks else None
        score["status"] = "partial_checker"
    elif "clarification_decision" in metric:
        expected_clarify = (expected or {}).get("should_clarify")
        if expected_clarify is None:
            # EAR uses ``answerable`` rather than a native should-ask label.
            expected_clarify = not bool((expected or {}).get("answerable"))
        asks = bool(
            re.search(
                r"\?|\bclarif|\bwhich\b|\bwhat do you mean\b|"
                r"\bwhich one\b|\bwould you like\b|请问|澄清|具体",
                prediction,
                re.I,
            )
        )
        answer = (expected or {}).get("answer")
        score.update(
            {
                "should_clarify": bool(expected_clarify),
                "asks_clarification": asks,
                "decision_correct": int(asks == bool(expected_clarify)),
                "text_f1": round(_f1_tokens(prediction, answer), 4)
                if answer
                else None,
            }
        )
        score["value"] = score["decision_correct"]
        score["status"] = "offline_proxy"
    elif "trigger_precision" in metric:
        tool, args = _extract_call(parsed)
        target = str(expected.get("target_tool") or "")
        positive = expected.get("polarity") == "positive"
        invoked = tool is not None
        target_hit = bool(tool and target in tool)
        score.update({"invoked": invoked, "target_tool": target, "target_hit": target_hit, "trigger_correct": int(target_hit if positive else not target_hit)})
        gold = expected.get("gold_call") or {}
        gname = gold.get("name") if isinstance(gold, dict) else None
        score["type_correct"] = int(tool == gname) if positive and gname else None
        score["value"] = score["trigger_correct"]
    elif "tool_use_success" in metric:
        predicted_calls = _call_list(parsed)
        expected_calls = _call_list(expected)
        # A null native call is an intentional no-tool case.  Treat a parsed
        # null call as the corresponding empty call rather than losing the
        # negative examples.  For multi-call turns, require the complete
        # ordered sequence and report an average parameter overlap as a
        # secondary diagnostic.
        if predicted_calls == [(None, {})] and expected_calls == [(None, {})]:
            parameter_f1 = 1.0
            calls_exact = 1
        else:
            pair_count = min(len(predicted_calls), len(expected_calls))
            overlaps = []
            for index in range(pair_count):
                got_tool, got_args = predicted_calls[index]
                exp_tool, exp_args = expected_calls[index]
                overlaps.append(_f1_tokens(_json_dump(got_args), _json_dump(exp_args)))
            parameter_f1 = sum(overlaps) / len(overlaps) if overlaps else 0.0
            calls_exact = int(
                len(predicted_calls) == len(expected_calls)
                and all(
                    got_tool == exp_tool
                    and _norm_text(_json_dump(got_args))
                    == _norm_text(_json_dump(exp_args))
                    for (got_tool, got_args), (exp_tool, exp_args) in zip(
                        predicted_calls, expected_calls
                    )
                )
            )
        score.update(
            {
                "predicted_call_count": len(predicted_calls),
                "expected_call_count": len(expected_calls),
                "calls_exact": calls_exact,
                "tool_correct": calls_exact,
                "parameter_f1": round(parameter_f1, 4),
                "value": calls_exact,
            }
        )
    elif "text_f1" in metric:
        reference = row.get("reference") or {}
        # AudioAgentBench stores its contextual target as ``golden_text``;
        # VocalBench stores the same field as ``answer``.  Support both so
        # the offline proxy is actually computed for every eligible sample.
        ref = reference.get("answer") or reference.get("golden_text")
        score["text_f1"] = round(_f1_tokens(prediction, ref), 4) if ref else None
        score["value"] = score["text_f1"]
        score["status"] = "offline_proxy"
    elif "constraint_satisfaction" in metric:
        constraints = ((expected or {}).get("constraints") or {})
        checks = []
        unsupported = []
        if constraints.get("no_comma"): checks.append("," not in prediction)
        if constraints.get("lowercase"): checks.append(prediction == prediction.lower())
        if constraints.get("bullets"): checks.append(bool(re.search(r"(^|\n)\s*[-*•]", prediction)))
        if constraints.get("min_lines"): checks.append(len([x for x in prediction.splitlines() if x.strip()]) >= constraints["min_lines"])
        if constraints.get("exact_lines"): checks.append(len([x for x in prediction.splitlines() if x.strip()]) == constraints["exact_lines"])
        if constraints.get("max_words"): checks.append(len(prediction.split()) < constraints["max_words"])
        if constraints.get("repeat_target"): checks.append(_f1_tokens(prediction, constraints["repeat_target"]) >= 0.6)
        if constraints.get("exact_items"):
            lines = [x for x in prediction.splitlines() if x.strip()]
            numbered = [x for x in lines if re.match(r"\s*(?:[-*•]|\d+[.)]|[一二三四五六七八九十]+[、.)])", x)]
            checks.append(len(numbered) == int(constraints["exact_items"]) or len(lines) == int(constraints["exact_items"]))
        frequency = constraints.get("keyword_frequency") or {}
        if frequency.get("keyword"):
            checks.append(
                len(re.findall(re.escape(str(frequency["keyword"])), prediction, re.I))
                >= int(frequency.get("minimum", 1))
            )
        if constraints.get("starts_with"):
            first = next((x.strip() for x in prediction.splitlines() if x.strip()), "")
            checks.append(first.startswith(str(constraints["starts_with"])))
        if constraints.get("forbidden_keywords"):
            checks.append(
                not any(
                    str(word).lower() in prediction.lower()
                    for word in constraints["forbidden_keywords"]
                )
            )
        if constraints.get("required_keywords"):
            checks.append(
                all(
                    str(word).lower() in prediction.lower()
                    for word in constraints["required_keywords"]
                )
            )
        if constraints.get("required_mode") == "apology":
            checks.append(bool(re.search(r"apolog|道歉|歉意", prediction, re.I)))
        if constraints.get("no_explanation"):
            # Whether a response contains an explanation is semantic; a
            # repeated target is the only safe local check.  Keep this as an
            # explicit unsupported dimension instead of silently passing it.
            unsupported.append("no_explanation")
        if constraints.get("unsupported_emotion"):
            unsupported.append("emotion/style (requires audio output or judge)")
        if constraints.get("unsupported_speed"):
            unsupported.append("speaking speed (requires audio output or timing)")
        if constraints.get("unsupported_conditional"):
            unsupported.append("conditional/semantic completeness (requires judge)")
        if constraints.get("unsupported_multistep"):
            unsupported.append("multi-step completeness (requires judge)")
        score["checks"] = checks
        score["unsupported"] = unsupported
        score["value"] = sum(checks) / len(checks) if checks else None
        score["status"] = "heuristic"
    elif "tf_win_rate" in metric:
        ref = ((row.get("reference") or {}).get("baseline") or "")
        score["token_f1_vs_baseline"] = round(_f1_tokens(prediction, ref), 4) if ref else None
        score["value"] = score["token_f1_vs_baseline"]
        score["status"] = "offline_proxy"
    else:
        score["value"] = None
        score["status"] = "judge_unavailable"
    return score


def prepare(args: argparse.Namespace) -> None:
    work = Path(args.work_root).resolve()
    work.mkdir(parents=True, exist_ok=True)
    rows = build_manifest(work)
    _write_jsonl(work / "samples.jsonl", rows)
    counts = Counter((row["capability"], row["benchmark"]) for row in rows)
    _write_json(work / "preflight.json", {"schema_version": "t4-smoke-preflight/1", "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "manifest": str(MANIFEST_PATH), "manifest_sha256": _sha256(MANIFEST_PATH) if MANIFEST_PATH.is_file() else None, "sample_count": len(rows), "counts": {f"{k[0]}::{k[1]}": v for k, v in sorted(counts.items())}, "audio_missing": [a for row in rows for a in row["audio"] if not a["exists"]], "targets": TARGETS})
    print(_json_dump({"work_root": str(work), "sample_count": len(rows), "counts": {f"{k[0]}::{k[1]}": v for k, v in sorted(counts.items())}}))


def _load_model_class():
    # Import the branch adapter only after the CPU-side manifest is complete.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from audio_evals.models.qwen3_omni import Qwen3Omni

    return Qwen3Omni


def infer(args: argparse.Namespace) -> None:
    work = Path(args.work_root).resolve()
    samples = _read_jsonl(work / "samples.jsonl")
    if not samples:
        raise RuntimeError("samples.jsonl is empty; run prepare first")
    selected = samples
    if args.capability:
        selected = [row for row in selected if row["capability"] == args.capability]
    if args.benchmark:
        selected = [row for row in selected if row["benchmark"] == args.benchmark]
    if args.shard_count > 1:
        selected = [row for i, row in enumerate(selected) if i % args.shard_count == args.shard_index]
    output = work / (args.output or f"predictions-{args.shard_index}.jsonl")
    done = {}
    if args.resume and output.is_file():
        for row in _read_jsonl(output):
            done[str(row.get("sample_id"))] = row
    cls = _load_model_class()
    model = cls(path=str(QWEN_CHECKPOINT), env_path=str(QWEN_ENV), requirements_path="", max_new_tokens=args.max_new_tokens, startup_timeout=3600, request_timeout=7200, request_log_interval=30, gpu_id=args.gpu_id)
    rows_out = []
    try:
        for ordinal, sample in enumerate(selected):
            if sample["sample_id"] in done:
                rows_out.append(done[sample["sample_id"]])
                continue
            started = time.time()
            try:
                prediction = model.inference(sample["prompt"])
                error = None
            except Exception as exc:
                prediction = ""
                error = f"{type(exc).__name__}: {exc}"
            rows_out.append({"sample_id": sample["sample_id"], "capability": sample["capability"], "benchmark": sample["benchmark"], "native_id": sample["native_id"], "prediction": str(prediction), "error": error, "latency_seconds": round(time.time() - started, 3), "model": "Qwen3-Omni-30B-A3B-Instruct", "checkpoint": str(QWEN_CHECKPOINT), "prompt_sha256": hashlib.sha256(_json_dump(sample["prompt"]).encode()).hexdigest()})
            _write_jsonl(output, rows_out)
            print(f"{ordinal + 1}/{len(selected)} {sample['sample_id']} error={error}", flush=True)
    finally:
        try:
            model.release()
        except Exception:
            pass
    print(_json_dump({"output": str(output), "rows": len(rows_out), "errors": sum(bool(r.get("error")) for r in rows_out)}))


def evaluate(args: argparse.Namespace) -> None:
    work = Path(args.work_root).resolve()
    samples = {row["sample_id"]: row for row in _read_jsonl(work / "samples.jsonl")}
    pred_rows: list[dict[str, Any]] = []
    for path in sorted(work.glob("predictions*.jsonl")):
        pred_rows.extend(_read_jsonl(path))
    # Last occurrence wins, which makes resume/sharded retries deterministic.
    predictions = {row["sample_id"]: row for row in pred_rows}
    audited = []
    for sample_id, sample in samples.items():
        pred = predictions.get(sample_id)
        if not pred:
            audited.append({"sample_id": sample_id, "capability": sample["capability"], "benchmark": sample["benchmark"], "error": "missing prediction", "score": {"status": "missing"}})
            continue
        scored = _score(sample, str(pred.get("prediction") or ""))
        audited.append({"sample_id": sample_id, "capability": sample["capability"], "benchmark": sample["benchmark"], "native_id": sample["native_id"], "prediction": pred.get("prediction"), "error": pred.get("error"), "reference": sample.get("reference"), "expected": sample.get("expected"), "meta": sample.get("meta"), "score": scored})
    _write_jsonl(work / "audited.jsonl", audited)
    summary: dict[str, Any] = {}
    for key, group in _groupby(audited, lambda r: (r["capability"], r["benchmark"])):
        values = [r["score"].get("value") for r in group if isinstance(r.get("score", {}).get("value"), (int, float))]
        summary[f"{key[0]}::{key[1]}"] = {"sample_count": len(group), "predicted": sum(not r.get("error") for r in group), "errors": sum(bool(r.get("error")) for r in group), "numeric_values": len(values), "mean_value": round(sum(values) / len(values), 4) if values else None, "statuses": dict(Counter(r.get("score", {}).get("status") for r in group))}
    _write_json(work / "summary.json", {"schema_version": "t4-smoke-summary/1", "summary": summary, "sample_count": len(audited), "missing_predictions": sum(r.get("error") == "missing prediction" for r in audited)})
    print(_json_dump({"summary": summary, "audited": str(work / "audited.jsonl")}))


def _groupby(rows: Sequence[dict[str, Any]], key):
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups.items()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--work-root", required=True)
    p.set_defaults(func=prepare)
    p = sub.add_parser("infer")
    p.add_argument("--work-root", required=True)
    p.add_argument("--output")
    p.add_argument("--capability")
    p.add_argument("--benchmark")
    p.add_argument("--gpu-id", default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.set_defaults(func=infer)
    p = sub.add_parser("evaluate")
    p.add_argument("--work-root", required=True)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
