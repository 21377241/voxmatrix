import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mesh_eval.core.benchmark_map import benchmark_profile


DEFAULT_SUFFIXES = {".wav", ".flac"}


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def index_candidates(
    candidate_dir: Path, suffixes: Iterable[str] = DEFAULT_SUFFIXES
) -> Dict[str, List[Path]]:
    allowed = {suffix.lower() for suffix in suffixes}
    candidates: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(candidate_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in allowed:
            candidates[path.stem].append(path.resolve())
    return candidates


def candidate_keys(row: Dict[str, Any]) -> List[str]:
    values = [row.get("sample_id"), row.get("filename"), row.get("ans")]
    keys = []
    for value in values:
        if value not in (None, ""):
            key = Path(str(value)).stem
            if key and key not in keys:
                keys.append(key)
    return keys


def audio_validation_error(path: Path) -> str:
    try:
        info = sf.info(path)
    except Exception as exc:
        return str(exc)
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        return (
            f"invalid audio metadata: frames={info.frames}, "
            f"samplerate={info.samplerate}, channels={info.channels}"
        )
    return ""


def attach_candidates(
    rows: List[Dict[str, Any]],
    candidate_dir: Path,
    benchmark_map: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_index = index_candidates(candidate_dir)
    output_rows = []
    missing = []
    ambiguous = []
    invalid = []
    counts = Counter()

    for index, source_row in enumerate(rows):
        row = dict(source_row)
        match = None
        for key in candidate_keys(row):
            paths = candidate_index.get(key, [])
            if len(paths) == 1:
                match = paths[0]
                break
            if len(paths) > 1:
                ambiguous.append(
                    {
                        "index": index,
                        "sample_id": row.get("sample_id", ""),
                        "key": key,
                        "paths": [str(path) for path in paths],
                    }
                )
                break
        if match is None:
            if not ambiguous or ambiguous[-1]["index"] != index:
                missing.append(
                    {
                        "index": index,
                        "sample_id": row.get("sample_id", ""),
                        "keys": candidate_keys(row),
                    }
                )
            continue

        validation_error = audio_validation_error(match)
        if validation_error:
            invalid.append(
                {
                    "index": index,
                    "sample_id": row.get("sample_id", ""),
                    "path": str(match),
                    "error": validation_error,
                }
            )
            continue

        benchmark = str(row.get("dataset_id") or row.get("dataset"))
        profile = benchmark_profile(benchmark, benchmark_map)
        row["metrics"] = profile["metrics"]
        row["evaluators"] = profile["evaluators"]
        row["prompt_name"] = profile.get("prompt", row.get("prompt_name", ""))
        eval_info = dict(row.get("eval_info") or {})
        eval_info.pop("eval", None)
        eval_info["inference"] = {"content": str(match)}
        eval_info["post_process"] = {"content": str(match)}
        row["eval_info"] = eval_info
        metadata = dict(row.get("metadata") or {})
        metadata["candidate_audio_path"] = str(match)
        row["metadata"] = metadata
        output_rows.append(row)
        counts[benchmark] += 1

    report = {
        "source_count": len(rows),
        "candidate_file_count": sum(len(paths) for paths in candidate_index.values()),
        "attached_count": len(output_rows),
        "missing_count": len(missing),
        "ambiguous_count": len(ambiguous),
        "invalid_count": len(invalid),
        "counts": dict(sorted(counts.items())),
        "missing": missing[:20],
        "ambiguous": ambiguous[:20],
        "invalid": invalid[:20],
    }
    return output_rows, report


def write_manifest(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 sample_id/filename/ans 文件名挂接 Seed-TTS 候选音频"
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark-map", default="")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    if not args.candidate_dir.is_dir():
        raise SystemExit(f"candidate directory does not exist: {args.candidate_dir}")
    rows = read_manifest(args.manifest)
    output_rows, report = attach_candidates(
        rows, args.candidate_dir, benchmark_map=args.benchmark_map
    )
    report.update(
        {
            "manifest": str(args.manifest),
            "candidate_dir": str(args.candidate_dir),
            "output": str(args.output),
        }
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.allow_partial and (
        report["missing_count"]
        or report["ambiguous_count"]
        or report["invalid_count"]
    ):
        raise SystemExit(1)
    write_manifest(args.output, output_rows)


if __name__ == "__main__":
    main()
