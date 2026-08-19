"""从 LibriSpeech 目录解析 .trans.txt，生成 raw_index.jsonl。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def source_data_root() -> str:
    """Return the canonical source root while accepting the legacy option."""
    return (
        os.environ.get("VOXMATRIX_SOURCE_DATA_ROOT")
        or os.environ.get("ULTRAEVAL_SOURCE_DATA_ROOT")
        or "data/source_datasets"
    )


def parse_subset_name(subset: str) -> dict:
    is_other = 1 if "other" in subset else 0
    is_clean = 1 if "clean" in subset else 0
    if subset.startswith("train"):
        split = "train"
    elif subset.startswith("dev"):
        split = "dev"
    elif subset.startswith("test"):
        split = "test"
    else:
        split = "test"
    return {
        "subset": subset,
        "split": split,
        "subset_is_other": is_other,
        "subset_is_clean": is_clean,
    }


def probe_duration_sec(audio_path: str) -> float | None:
    """尽量读取时长；失败则返回 None（不阻断索引）。"""
    path = Path(audio_path)
    if not path.is_file():
        return None
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate and info.frames:
            return round(info.frames / float(info.samplerate), 3)
    except Exception:
        pass
    try:
        # flac/wav 裸读失败时忽略
        import wave

        if path.suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as w:
                return round(w.getnframes() / float(w.getframerate()), 3)
    except Exception:
        pass
    return None


def iter_records(root: Path, subset: str, limit: int = 0, *, with_duration: bool = True):
    subset_dir = root / subset
    if not subset_dir.is_dir():
        raise FileNotFoundError(f"子集不存在: {subset_dir}")

    meta = parse_subset_name(subset)
    count = 0
    for trans_path in sorted(subset_dir.rglob("*.trans.txt")):
        with open(trans_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, _, text = line.partition(" ")
                text = text.strip()
                audio_flac = trans_path.parent / f"{utt_id}.flac"
                if audio_flac.exists():
                    audio_path = str(audio_flac)
                else:
                    candidates = list(trans_path.parent.glob(f"{utt_id}.*"))
                    audio_path = str(candidates[0]) if candidates else str(audio_flac)

                try:
                    rel = str(Path(audio_path).relative_to(root))
                except ValueError:
                    rel = audio_path
                rec = {
                    "id": utt_id,
                    "text": text,
                    "audio_path": audio_path,
                    "audio_filepath": rel,
                    **meta,
                }
                if with_duration:
                    dur = probe_duration_sec(audio_path)
                    if dur is not None:
                        rec["duration_sec"] = dur
                yield rec
                count += 1
                if limit and count >= limit:
                    return


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 LibriSpeech 索引")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            source_data_root()
        ).expanduser()
        / "01_basic_transcription"
        / "LibriSpeech",
    )
    parser.add_argument("--subset", default="train-clean-100")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in iter_records(args.root, args.subset, limit=args.limit):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"索引完成: {n} 条 → {args.output} (subset={args.subset})")


if __name__ == "__main__":
    main()
