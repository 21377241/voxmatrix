import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.core.schema import normalize_record
from mesh_eval.core.schema_v2 import (
    adapt_v1_to_v2,
    validate_canonical_sample,
    validate_json_schema,
)
from mesh_eval.scripts.build_manifest_from_registry import dataset_profile
from audio_evals.config import get_environment
from audio_evals.prepare_dataset import _write_manifest


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source",
        action="append",
        help="Dataset and event jsonl in the form dataset_name=/abs/path/result.jsonl",
    )
    parser.add_argument("--limit_per_dataset", type=int, default=20)
    parser.add_argument(
        "--audio_base_dir",
        default=get_environment("VOXMATRIX_LEGACY_AUDIO_ROOT", str(REPO_ROOT)),
        help="Base directory used to resolve relative audio paths stored in old prompt events.",
    )
    parser.add_argument("--use_bucket", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--schema-version",
        choices=["1.0", "2.0"],
        default="2.0",
        help="New manifests are canonical V2 by default; V1 is compatibility-only.",
    )
    parser.add_argument(
        "--replay-events-output",
        default="",
        help="V2 replay sidecar path. Defaults to <output>.replay.jsonl.",
    )
    return parser.parse_args()


def parse_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"source must be dataset=path, got: {value}")
    dataset, path = value.split("=", 1)
    return dataset.strip(), path.strip()


def read_events(path: str) -> Dict[int, Dict[str, Any]]:
    rows: Dict[int, Dict[str, Any]] = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            rows[int(event["id"])][event["type"]] = event.get("data", {})
    return rows


def find_audio_path(prompt_data: Dict[str, Any]) -> str:
    content = prompt_data.get("content")
    if isinstance(content, list):
        for message in content:
            for item in message.get("contents", []) or message.get("content", []):
                if isinstance(item, dict) and item.get("type") == "audio":
                    return item.get("value") or item.get("audio") or ""
    return ""


def resolve_audio_path(path: str, base_dir: str) -> str:
    if not path:
        return ""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def metrics_from_eval(eval_data: Dict[str, Any], profile: Dict[str, Any]) -> List[str]:
    metrics = []
    if "cer%" in eval_data:
        metrics.append("cer")
    if "wer%" in eval_data:
        metrics.append("wer")
    if "match" in eval_data:
        metrics.append("accuracy")
    return metrics or profile["metrics"]


def build_rows_for_source(
    dataset_name: str,
    path: str,
    limit: int,
    audio_base_dir: str,
    use_bucket: str,
    split: str,
    schema_version: str = "2.0",
    replay_events: Optional[List[Dict[str, Any]]] = None,
    event_id_offset: int = 0,
) -> List[Dict[str, Any]]:
    profile = dataset_profile(dataset_name, "")
    rows = read_events(path)
    output = []
    for index in sorted(rows):
        if len(output) >= limit:
            break
        row = rows[index]
        prompt = row.get("prompt")
        eval_data = row.get("eval")
        if not prompt or not eval_data or "ref" not in eval_data:
            continue
        inference_data = row.get("inference") or {}
        post_process_data = row.get("post_process") or {}
        prediction = post_process_data.get("content")
        if prediction in (None, ""):
            prediction = inference_data.get("content")
        if prediction in (None, ""):
            prediction = eval_data.get("pred")
        if prediction in (None, ""):
            continue
        audio_path = resolve_audio_path(find_audio_path(prompt), audio_base_dir)
        if not os.path.isfile(audio_path):
            continue
        sample = {
            "sample_id": f"{dataset_name}_{index:05d}",
            "dataset": dataset_name,
            "task": profile["task"],
            "capability": profile["capability"],
            "scenario": profile["scenario"],
            "condition": profile["condition"],
            "input": {"audio_path": audio_path},
            "reference": {"text": str(eval_data["ref"])},
            "answer_type": profile["answer_type"],
            "metrics": metrics_from_eval(eval_data, profile),
            "evaluators": profile.get("evaluators", []),
            "prompt_name": profile.get("prompt", ""),
            "use_bucket": use_bucket or profile.get("use_bucket", "diagnostic_evidence"),
            "split": split,
            "metadata": {
                "language": profile["language"],
                "domain": dataset_name,
                "source_event_jsonl": path,
                "source_event_id": index,
            },
        }
        if schema_version == "2.0":
            converted = adapt_v1_to_v2(sample, ref_col="text")
            provenance = dict(converted.get("provenance") or {})
            provenance["replay_source"] = {
                "source": "verified_derived",
                "confidence": 1.0,
                "derivation_version": "event-replay-builder@2",
                "evidence": path,
            }
            converted["provenance"] = provenance
            errors = validate_canonical_sample(converted, formal=False)
            errors.extend(validate_json_schema(converted, "canonical"))
            if errors:
                raise ValueError(
                    f"source event {path} id={index} cannot produce canonical V2: "
                    + "; ".join(errors)
                )
            output.append(converted)
            if replay_events is not None:
                event_id = event_id_offset + len(output) - 1
                replay_events.extend(
                    [
                        {
                            "type": "inference",
                            "id": event_id,
                            "data": {
                                "content": inference_data.get("content", prediction),
                                **(
                                    {"runtime": inference_data["runtime"]}
                                    if isinstance(inference_data.get("runtime"), dict)
                                    else {}
                                ),
                            },
                        },
                        {
                            "type": "post_process",
                            "id": event_id,
                            "data": {"content": prediction},
                        },
                    ]
                )
        else:
            sample["eval_info"] = {
                "inference": {"content": prediction},
                "post_process": {"content": prediction},
            }
            legacy = normalize_record(
                sample, ref_col="text", default_dataset_id=dataset_name
            )
            legacy["schema_version"] = "1.0"
            output.append(legacy)
    return output


def main():
    args = get_args()
    source_values = list(args.source or [])
    if not source_values:
        raise ValueError("at least one --source dataset=path is required")
    all_rows = []
    replay_events = []
    summary = {}
    for source in source_values:
        dataset_name, path = parse_source(source)
        rows = build_rows_for_source(
            dataset_name=dataset_name,
            path=path,
            limit=args.limit_per_dataset,
            audio_base_dir=args.audio_base_dir,
            use_bucket=args.use_bucket,
            split=args.split,
            schema_version=args.schema_version,
            replay_events=replay_events,
            event_id_offset=len(all_rows),
        )
        all_rows.extend(rows)
        summary[dataset_name] = {"count": len(rows), "path": path}

    _write_manifest(Path(args.output).resolve(), all_rows)
    replay_output = ""
    if args.schema_version == "2.0":
        replay_output = args.replay_events_output or str(
            Path(args.output).with_suffix(".replay.jsonl")
        )
        _write_manifest(Path(replay_output).resolve(), replay_events)
    print(
        json.dumps(
            {
                "output": args.output,
                "schema_version": args.schema_version,
                "replay_events_output": replay_output,
                "total": len(all_rows),
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
