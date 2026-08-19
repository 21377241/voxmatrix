import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
from datasets import IterableDataset, load_dataset

from audio_evals.dataset.huggingface import (
    _atomic_write_wav,
    _exclusive_lock,
    _valid_wav,
)
from audio_evals.dataset.prepared import PreparedAudioJsonl
from audio_evals.prepare_dataset import _write_manifest, prepare_dataset
from audio_evals.registry import registry
from mesh_eval.core.benchmark_map import apply_benchmark_fallback, benchmark_profile
from mesh_eval.core.schema import normalize_record
from mesh_eval.core.schema_v2 import (
    adapt_v1_to_v2,
    is_canonical_v2,
    validate_canonical_sample,
    validate_json_schema,
)


DEFAULT_DATASETS = [
    "aishell-1",
    "librispeech-test-clean",
    "fleurs-zh",
    "covost2-zh-en",
    "air-foundation",
    "speech-cmmlu",
    "hsk1",
    "mmau-test-mini",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="mesh_eval/data/benchmark_mini.jsonl")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--limit_per_dataset", type=int, default=12)
    parser.add_argument("--registry_path", default="")
    parser.add_argument("--use_bucket", default="diagnostic_evidence")
    parser.add_argument("--split", default="mini")
    parser.add_argument(
        "--loader",
        choices=["safe_hf", "registry", "prepared"],
        default="",
        help="safe_hf reads HF dataset specs and takes only the first N rows; "
        "registry calls the original dataset.load and can be heavy; prepared "
        "uses the durable prepare-dataset flow. Defaults to prepared in formal "
        "mode and safe_hf in diagnostic mode.",
    )
    parser.add_argument(
        "--mode",
        choices=["diagnostic", "formal"],
        default="diagnostic",
        help="Formal mode requires the prepared loader. Diagnostic mode keeps "
        "the bounded/streaming convenience path used by mini manifests.",
    )
    parser.add_argument(
        "--prepared_root",
        default="",
        help="Root for per-dataset prepared assets/manifests in formal mode.",
    )
    parser.add_argument(
        "--checksum", choices=["sha256", "none"], default="sha256"
    )
    parser.add_argument("--audio_root", default="raw/mesh_benchmark_mini")
    parser.add_argument("--no_streaming", action="store_true")
    parser.add_argument("--allow_missing_audio", action="store_true")
    parser.add_argument(
        "--allow_partial",
        action="store_true",
        help="Write successful datasets and exit zero even if another dataset fails. "
        "Intended only for exploratory diagnostics.",
    )
    parser.add_argument(
        "--evaluator_override",
        nargs="+",
        default=[],
        help="Override mapped evaluators, e.g. dump for inference-only preparation.",
    )
    return parser.parse_args()


def dataset_profile(dataset_name: str, default_task: str) -> Dict[str, Any]:
    try:
        return benchmark_profile(dataset_name)
    except KeyError:
        pass
    lowered = dataset_name.lower()
    task = default_task.lower()
    profile = {
        "task": "speech_understanding",
        "capability": "asr",
        "scenario": "phone",
        "condition": {
            "acoustic": "clean",
            "spatial": "near_field",
            "speaker": "single_speaker",
            "device": "phone_mic",
            "interaction": "single_turn",
        },
        "answer_type": "text",
        "metrics": ["wer"],
        "language": "en",
    }

    if any(key in lowered for key in ("aishell", "wenetspeech", "fleurs-zh", "hsk", "cmmlu")) or "zh" in task:
        profile["language"] = "zh"
        profile["metrics"] = ["cer"]
    if "meeting" in lowered:
        profile["scenario"] = "meeting"
        profile["condition"].update(
            {"acoustic": "overlap_speech", "spatial": "far_field", "device": "meeting_array"}
        )
    if "covost2" in lowered or task.startswith("sst2"):
        profile.update(
            {
                "capability": "speech_translation",
                "answer_type": "text",
                "metrics": ["bleu"],
            }
        )
        if "zh-en" in lowered or task.endswith("en"):
            profile["language"] = "zh"
            profile["target_language"] = "en"
    if any(key in lowered for key in ("air", "cmmlu", "hsk", "mmau")):
        profile.update(
            {
                "task": "agent",
                "capability": "qa",
                "answer_type": "classification",
                "metrics": ["accuracy"],
                "scenario": "phone",
            }
        )
    return profile


def first_present(doc: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if doc.get(key) not in (None, "", []):
            return doc[key]
    return ""


def apply_aliases(doc: Dict[str, Any], col_aliases: Dict[str, str]) -> Dict[str, Any]:
    result = dict(doc)
    for source, target in (col_aliases or {}).items():
        if source in result and target not in result:
            result[target] = result[source]
    return result


def get_dataset_spec(dataset_name: str) -> Dict[str, Any]:
    spec = registry._dataset.get(dataset_name)
    if spec is None:
        return {}
    return spec


def iter_hf_rows(spec: Dict[str, Any], limit: int, no_streaming: bool) -> Iterable[Dict[str, Any]]:
    if limit < 0:
        raise ValueError("limit_per_dataset must be non-negative")
    args = spec.get("args", {})
    local_path_value = str(args.get("local_path") or "")
    local_path = Path(local_path_value) if local_path_value else None
    local_data_dir = local_path / "data" if local_path else None
    if local_data_dir and local_data_dir.is_dir():
        split = str(args.get("split") or "train")
        if split in {"all", "*"}:
            parquet_files = sorted(local_data_dir.glob("*.parquet"))
        else:
            parquet_files = sorted(local_data_dir.glob(f"{split}-*.parquet"))
        if limit > 0:
            # Read metadata until enough rows are covered; do not fingerprint the
            # remaining shards in a multi-hundred-GB repository.
            import pyarrow.parquet as pq

            selected = []
            row_budget = 0
            for path in parquet_files:
                selected.append(path)
                row_budget += pq.ParquetFile(path).metadata.num_rows
                if row_budget >= limit:
                    break
            parquet_files = selected
        if parquet_files:
            import pyarrow.parquet as pq

            emitted = 0
            for path in parquet_files:
                parquet = pq.ParquetFile(path)
                batch_size = min(32, limit) if limit else 32
                for batch in parquet.iter_batches(batch_size=max(1, batch_size)):
                    for row in batch.to_pylist():
                        for column, audio in list(row.items()):
                            if not re.fullmatch(r"audio\d*", column):
                                continue
                            if isinstance(audio, dict) and audio.get("bytes"):
                                array, sampling_rate = sf.read(
                                    io.BytesIO(audio["bytes"]),
                                    dtype="float32",
                                    always_2d=False,
                                )
                                row[column] = {
                                    "array": array,
                                    "sampling_rate": sampling_rate,
                                    "path": audio.get("path"),
                                }
                        yield row
                        emitted += 1
                        if limit and emitted >= limit:
                            return
            return

    load_args = {"path": args["name"], "trust_remote_code": True}
    if args.get("subset"):
        load_args["name"] = args["subset"]
    if args.get("split") and args["split"] not in {"all", "*"}:
        load_args["split"] = args["split"]
    if args.get("data_files"):
        load_args["data_files"] = args["data_files"]
    if args.get("revision"):
        load_args["revision"] = args["revision"]
    if args.get("cache_dir"):
        load_args["cache_dir"] = args["cache_dir"]
    load_args["trust_remote_code"] = args.get("trust_remote_code", True)

    if no_streaming:
        if load_args.get("split") and limit:
            load_args["split"] = f"{load_args['split']}[:{limit}]"
        dataset = load_dataset(**load_args)
        emitted = 0
        datasets = dataset.values() if hasattr(dataset, "values") else [dataset]
        for split_dataset in datasets:
            for row in split_dataset:
                yield row
                emitted += 1
                if limit and emitted >= limit:
                    return
        return

    dataset = load_dataset(**load_args, streaming=True)
    if not isinstance(dataset, IterableDataset):
        # Some dataset scripts ignore streaming and return a DatasetDict.
        if hasattr(dataset, "values"):
            dataset = next(iter(dataset.values()))
    datasets = dataset.values() if hasattr(dataset, "values") else [dataset]
    emitted = 0
    for split_dataset in datasets:
        for row in split_dataset:
            if limit and emitted >= limit:
                return
            yield row
            emitted += 1


def save_audio_field(
    doc: Dict[str, Any],
    dataset_name: str,
    index: int,
    audio_root: str,
    allow_missing_audio: bool,
) -> Dict[str, Any]:
    result = dict(doc)
    found_audio = False
    audio_columns = sorted(
        key for key in result if re.fullmatch(r"audio\d*", key)
    )
    for column in audio_columns:
        audio = result.get(column)
        wav_column = "WavPath" if column == "audio" else f"WavPath{column[5:]}"
        if isinstance(audio, dict) and audio.get("array") is not None and audio.get(
            "sampling_rate"
        ):
            out_dir = os.path.join(audio_root, dataset_name)
            os.makedirs(out_dir, exist_ok=True)
            suffix = "" if column == "audio" else f"_{column}"
            out_path = os.path.abspath(
                os.path.join(out_dir, f"{index:05d}{suffix}.wav")
            )
            with _exclusive_lock(f"{out_path}.lock"):
                if not _valid_wav(out_path):
                    _atomic_write_wav(
                        out_path,
                        audio["array"],
                        int(audio["sampling_rate"]),
                        "PCM_16",
                    )
            result[wav_column] = out_path
            found_audio = True
        elif isinstance(audio, dict) and audio.get("path"):
            result[wav_column] = audio["path"]
            found_audio = True
        result.pop(column, None)

    for key in ("WavPath", "audio_path", "path", "wav", "wav_path"):
        if result.get(key):
            result["WavPath"] = result[key]
            found_audio = True
            break

    if found_audio or any(re.fullmatch(r"WavPath\d+", key) for key in result):
        return result

    if not allow_missing_audio:
        raise ValueError(f"missing audio field for {dataset_name} row {index}")
    result["WavPath"] = ""
    return result


def load_docs_safe_hf(
    dataset_name: str,
    spec: Dict[str, Any],
    limit: int,
    audio_root: str,
    no_streaming: bool,
    allow_missing_audio: bool,
) -> List[Dict[str, Any]]:
    cls = spec.get("cls") or spec.get("class")
    if cls != "audio_evals.dataset.huggingface.Huggingface":
        raise ValueError(f"safe_hf only supports Huggingface datasets, got {cls}")

    args = spec.get("args", {})
    docs = []
    for index, row in enumerate(iter_hf_rows(spec, limit, no_streaming)):
        row = apply_aliases(dict(row), args.get("col_aliases") or {})
        row = save_audio_field(
            row,
            dataset_name=dataset_name,
            index=index,
            audio_root=audio_root,
            allow_missing_audio=allow_missing_audio,
        )
        docs.append(row)
    return docs


def load_docs_registry(dataset_name: str, limit: int) -> tuple[List[Dict[str, Any]], str, str]:
    dataset = registry.get_dataset(dataset_name)
    if dataset is None:
        raise KeyError(f"dataset not found: {dataset_name}")
    return dataset.load(limit=limit), dataset.ref_col, dataset.task_name


def load_docs_prepared(
    dataset_name: str,
    limit: int,
    prepared_root: str,
    checksum: str,
) -> tuple[List[Dict[str, Any]], str, str]:
    dataset = registry.get_dataset(dataset_name)
    if dataset is None:
        raise KeyError(f"dataset not found: {dataset_name}")
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_name).strip("._")
    output_dir = os.path.join(prepared_root, safe_name or "dataset")
    prepare_dataset(
        dataset_name=dataset_name,
        output_dir=output_dir,
        limit=limit,
        checksum=checksum,
    )
    manifest = os.path.join(output_dir, "manifest.jsonl")
    prepared = PreparedAudioJsonl(
        manifest,
        default_task=dataset.task_name,
        ref_col=dataset.ref_col,
        col_aliases=dataset.col_aliases,
    )
    return prepared.load(), prepared.ref_col, prepared.task_name


def normalize_choices(sample: Dict[str, Any]) -> None:
    choices = sample.get("choices")
    if choices in (None, "", []):
        explicit = [
            sample.get("choice_a"),
            sample.get("choice_b"),
            sample.get("choice_c"),
            sample.get("choice_d"),
        ]
        choices = [item for item in explicit if item not in (None, "", [])]
    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except Exception:
            choices = [item.strip() for item in choices.splitlines() if item.strip()]
    if isinstance(choices, dict):
        choices = list(choices.values())
    if isinstance(choices, list):
        sample["choices"] = choices
        for index, value in enumerate(choices[:4]):
            sample[f"choice_{chr(97 + index)}"] = value


def convert_doc(
    dataset_name: str,
    dataset_ref_col: str,
    default_task: str,
    doc: Dict[str, Any],
    index: int,
    use_bucket: str,
    split: str,
    canonical_v2: bool = False,
) -> Dict[str, Any]:
    if is_canonical_v2(doc):
        return apply_benchmark_fallback(
            doc, dataset_profile(dataset_name, default_task), benchmark=dataset_name
        )
    profile = dataset_profile(dataset_name, default_task)
    reference = first_present(
        doc,
        [
            dataset_ref_col,
            "text",
            "sentence",
            "raw_transcription",
            "translation",
            "answer",
            "Answer",
            "ans_choice",
            "caption",
        ],
    )
    numbered_audio_paths = [
        (int(match.group(1)), value)
        for key, value in doc.items()
        if (match := re.fullmatch(r"WavPath(\d+)", key)) and value
    ]
    numbered_audio_paths.sort()
    primary_audio_path = doc.get("WavPath", "")
    if not primary_audio_path and numbered_audio_paths:
        primary_audio_path = numbered_audio_paths[0][1]
    input_obj: Dict[str, Any] = {"audio_path": primary_audio_path}
    if numbered_audio_paths:
        input_obj["audios"] = [
            {"uri": path, "turn": turn} for turn, path in numbered_audio_paths
        ]

    sample = {
        "sample_id": f"{dataset_name}_{index:05d}",
        "dataset": dataset_name,
        "input": input_obj,
        "reference": {"text": str(reference)},
        "use_bucket": use_bucket or profile.get("use_bucket", "diagnostic_evidence"),
        "split": split,
        "metadata": {
            "domain": dataset_name,
            "source_task": default_task,
        },
    }
    for field in ("task", "capability", "scenario", "condition", "answer_type"):
        if doc.get(field) not in (None, "", []):
            sample[field] = doc[field]
    if doc.get("metadata") and isinstance(doc["metadata"], dict):
        sample["metadata"].update(doc["metadata"])
    for field in ("metrics", "evaluators", "prompt_name"):
        if doc.get(field) not in (None, "", []):
            sample[field] = doc[field]
    sample = apply_benchmark_fallback(sample, profile, benchmark=dataset_name)
    if profile.get("target_language"):
        sample["metadata"]["target_language"] = profile["target_language"]

    for key in (
        "question",
        "QuestionText",
        "prompt",
        "instruction",
        "choice_a",
        "choice_b",
        "choice_c",
        "choice_d",
        "choices",
    ):
        if key in doc:
            sample[key] = doc[key]
    if "QuestionText" in sample and "question" not in sample:
        sample["question"] = sample["QuestionText"]
    if "prompt" in sample and "question" not in sample:
        sample["question"] = sample["prompt"]
    if "instruction" in sample and "question" not in sample:
        sample["question"] = sample["instruction"]
    normalize_choices(sample)

    for key, value in doc.items():
        if re.fullmatch(r"audio\d*", key):
            continue
        if key not in sample and key != "WavPath":
            sample[key] = value

    if canonical_v2:
        return adapt_v1_to_v2(sample, ref_col="text")
    return normalize_record(sample, ref_col="text", default_dataset_id=dataset_name)


def validate_formal_samples(
    dataset_name: str, samples: Iterable[Dict[str, Any]]
) -> None:
    """Fail closed unless every formal output satisfies both V2 validators."""
    for index, sample in enumerate(samples):
        sample_id = str(sample.get("sample_id") or index)
        errors = [
            f"python: {error}"
            for error in validate_canonical_sample(sample, formal=True)
        ]
        errors.extend(
            f"json_schema: {error}"
            for error in validate_json_schema(sample, "canonical")
        )
        if errors:
            raise ValueError(
                f"dataset {dataset_name} sample {sample_id} cannot produce "
                "canonical V2: " + "; ".join(errors)
            )


def main():
    args = get_args()
    if args.limit_per_dataset < 0:
        raise ValueError("limit_per_dataset must be non-negative")
    loader = args.loader or ("prepared" if args.mode == "formal" else "safe_hf")
    if args.mode == "formal" and loader != "prepared":
        raise ValueError("formal mode requires --loader prepared")
    if args.allow_partial and args.mode == "formal":
        raise ValueError("--allow_partial is only available in diagnostic mode")
    if args.registry_path:
        registry.add_registry_paths(args.registry_path.split())

    prepared_root = args.prepared_root
    if not prepared_root:
        output_path = Path(args.output).resolve()
        prepared_root = str(output_path.parent / f"{output_path.stem}.prepared")

    rows = []
    summary = {}
    for dataset_name in args.datasets:
        spec = get_dataset_spec(dataset_name)
        if not spec:
            summary[dataset_name] = {"status": "missing_dataset"}
            continue
        spec_args = spec.get("args", {})
        dataset_ref_col = spec_args.get("ref_col", "text")
        default_task = spec_args.get("default_task", "")
        try:
            if loader == "safe_hf":
                docs = load_docs_safe_hf(
                    dataset_name=dataset_name,
                    spec=spec,
                    limit=args.limit_per_dataset,
                    audio_root=args.audio_root,
                    no_streaming=args.no_streaming,
                    allow_missing_audio=args.allow_missing_audio,
                )
            elif loader == "registry":
                docs, dataset_ref_col, default_task = load_docs_registry(
                    dataset_name, args.limit_per_dataset
                )
            else:
                docs, dataset_ref_col, default_task = load_docs_prepared(
                    dataset_name,
                    args.limit_per_dataset,
                    prepared_root,
                    args.checksum,
                )
            converted = [
                convert_doc(
                    dataset_name=dataset_name,
                    dataset_ref_col=dataset_ref_col,
                    default_task=default_task,
                    doc=doc,
                    index=index,
                    use_bucket=args.use_bucket,
                    split=args.split,
                    canonical_v2=args.mode == "formal",
                )
                for index, doc in enumerate(docs)
            ]
            if args.evaluator_override:
                for sample in converted:
                    if is_canonical_v2(sample):
                        hints = dict(sample.get("evaluation_hints") or {})
                        hints["evaluators"] = list(args.evaluator_override)
                        hints["source"] = "cli_evaluator_override"
                        sample["evaluation_hints"] = hints
                    else:
                        sample["evaluators"] = list(args.evaluator_override)
            if args.mode == "formal":
                validate_formal_samples(dataset_name, converted)
        except Exception as exc:
            summary[dataset_name] = {"status": "error", "error": str(exc)}
            continue
        rows.extend(converted)
        summary[dataset_name] = {"status": "ok", "count": len(converted)}

    _write_manifest(Path(args.output).resolve(), rows)

    result = {
        "output": args.output,
        "mode": args.mode,
        "loader": loader,
        "total": len(rows),
        "summary": summary,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    failures = {
        name: item for name, item in summary.items() if item.get("status") != "ok"
    }
    if failures and not args.allow_partial:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
