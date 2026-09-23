#!/usr/bin/env python3
"""Build 30-60s long_form_asr windows to differentiate from utterance-level asr.

For session+timestamp indexes (AISHELL-4 / AliMeeting / MISP / SBCSAE): crop
continuous windows from the parent recording and concatenate in-window refs.

For AMI / CHiME-6 (already utterance wavs): pack consecutive clips in time order
until duration is in [30, 60] and ffmpeg-concat the audio.
"""

from __future__ import annotations

import json
import re
import subprocess
import wave
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
AUDIO_ROOT = NATIVE / "long_form_windows"
FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"
MIN_DUR = 30.0
MAX_DUR = 60.0
LIMIT = 10

UTT_INDEXES = {
    "aishell_4": ROOT / "mesh_eval" / "data" / "native" / "long_form_timelines" / "aishell_4.jsonl",
    "alimeeting": ROOT / "mesh_eval" / "data" / "native" / "long_form_timelines" / "alimeeting.jsonl",
    "misp": ROOT / "mesh_eval" / "data" / "native" / "long_form_timelines" / "misp.jsonl",
    "sbcsae": ROOT / "mesh_eval" / "data" / "native" / "long_form_timelines" / "sbcsae.jsonl",
}

AMI_MANIFEST = Path(
    "/mnt/afs/oss_data/datasets/03_multi_speaker/AMI/ami_ihm_test_asr.jsonl"
)
AMI_ROOT = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/AMI")
CHIME_MANIFEST = Path(
    "/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME-6/chime6_ref_array_test_asr.jsonl"
)
CHIME_ROOT = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME-6")


def audio_duration(path: Path) -> float:
    """Duration in seconds. AMI float WAV is not readable by stdlib wave."""
    try:
        with wave.open(str(path), "rb") as handle:
            return handle.getnframes() / float(handle.getframerate())
    except Exception:
        import soundfile as sf

        return float(sf.info(str(path)).duration)


def ffmpeg_crop(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if end <= start:
        raise ValueError(f"bad interval {start}-{end} for {src}")
    cmd = [
        FFMPEG,
        "-y",
        "-ss",
        f"{start:.6f}",
        "-to",
        f"{end:.6f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    # MISP CSOBx3 raw pcm: 8ch s32le @ 16kHz → downmix to mono (same as utt_clips / smoke fix).
    # Wrong decode as mono s16le / 16ch s16le yields clipped white noise.
    if src.suffix.lower() == ".pcm":
        cmd = [
            FFMPEG,
            "-y",
            "-f",
            "s32le",
            "-ar",
            "16000",
            "-ac",
            "8",
            "-ss",
            f"{start:.6f}",
            "-to",
            f"{end:.6f}",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(dst),
        ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 1000:
        raise RuntimeError(
            f"ffmpeg crop failed {src} [{start},{end}]\n{proc.stderr[-1000:]}"
        )


def ffmpeg_concat(paths: list[Path], dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    listing = dst.with_suffix(".txt")
    listing.write_text(
        "".join(f"file '{path}'\n" for path in paths), encoding="utf-8"
    )
    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(listing),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 1000:
        raise RuntimeError(f"ffmpeg concat failed\n{proc.stderr[-1000:]}")


def pack_session_rows(rows: list[dict]) -> list[tuple[float, float, list[dict]]]:
    """Non-overlapping windows with duration in [MIN_DUR, MAX_DUR]."""
    rows = sorted(rows, key=lambda r: float(r["start_time"]))
    windows = []
    i = 0
    while i < len(rows):
        start = float(rows[i]["start_time"])
        j = i
        end = float(rows[i]["end_time"])
        while j + 1 < len(rows):
            nxt_end = float(rows[j + 1]["end_time"])
            if nxt_end - start > MAX_DUR:
                break
            j += 1
            end = nxt_end
            if end - start >= MIN_DUR:
                # Prefer the first legal window (>=30s, <=60s) then stop growing.
                break
        dur = end - start
        if dur < MIN_DUR:
            break
        if MIN_DUR <= dur <= MAX_DUR:
            windows.append((start, end, rows[i : j + 1]))
        i = j + 1
    return windows


def join_text(rows: list[dict]) -> str:
    parts = []
    for row in rows:
        text = str(row.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def build_from_utt_index(dataset: str, index: Path, limit: int = LIMIT) -> list[dict]:
    by_src: dict[str, list[dict]] = defaultdict(list)
    with index.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("src_audio")
            if not src or row.get("start_time") is None or row.get("end_time") is None:
                continue
            if not Path(src).is_file():
                continue
            by_src[str(src)].append(row)

    out_rows: list[dict] = []
    audio_dir = AUDIO_ROOT / dataset
    # Round-robin across sessions for diversity.
    session_windows = []
    for src, rows in by_src.items():
        wins = pack_session_rows(rows)
        for start, end, pack in wins:
            session_windows.append((src, start, end, pack))

    # Prefer spreading across source files.
    by_file: dict[str, list] = defaultdict(list)
    for item in session_windows:
        by_file[item[0]].append(item)
    interleaved = []
    files = list(by_file.keys())
    idxs = {f: 0 for f in files}
    while len(interleaved) < len(session_windows):
        progressed = False
        for f in files:
            i = idxs[f]
            if i < len(by_file[f]):
                interleaved.append(by_file[f][i])
                idxs[f] = i + 1
                progressed = True
                if len(interleaved) >= limit * 3:
                    break
        if not progressed or len(interleaved) >= limit * 3:
            break

    for src, start, end, pack in interleaved:
        if len(out_rows) >= limit:
            break
        text = join_text(pack)
        if not text:
            continue
        sample_id = f"{dataset}_lf_{len(out_rows):04d}"
        dst = audio_dir / f"{sample_id}.wav"
        if not dst.is_file():
            print(f"[crop] {dataset} {sample_id} {end-start:.1f}s from {Path(src).name}")
            ffmpeg_crop(Path(src), dst, start, end)
        dur = audio_duration(dst)
        if not (MIN_DUR - 0.5 <= dur <= MAX_DUR + 0.5):
            print(f"[skip] {sample_id} duration {dur:.2f}s out of range")
            continue
        out_rows.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "capability": "long_form_asr",
                "protocol": "long_form_asr_window_30_60",
                "WavPath": str(dst),
                "text": text,
                "src_audio": src,
                "start_time": start,
                "end_time": end,
                "audio_duration_seconds": round(dur, 3),
                "n_utterances": len(pack),
                "window_min_seconds": MIN_DUR,
                "window_max_seconds": MAX_DUR,
            }
        )
    return out_rows


def _ami_sort_key(path: str) -> tuple:
    # eval_ami_en2002a_h00_mee073_0019663_0019800.wav
    match = re.search(r"_(\d+)_(\d+)\.wav$", path)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def _chime_sort_key(path: str) -> tuple:
    # chime6_worn_test_S01_P01_000030720_000204480.wav
    match = re.search(r"_(\d+)_(\d+)\.wav$", path)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def _group_key_ami(path: str) -> str:
    # meeting id folder
    parts = Path(path).parts
    for part in parts:
        if re.fullmatch(r"[A-Z]{2}\d{4}[a-z]", part):
            return part
    return Path(path).parent.name


def _group_key_chime(path: str) -> str:
    # .../worn/test/S01/...
    parts = Path(path).parts
    for part in parts:
        if re.fullmatch(r"S\d+", part):
            return part
    return Path(path).parent.name


def build_from_utt_files(
    dataset: str,
    manifest: Path,
    *,
    group_key,
    sort_key,
    limit: int = LIMIT,
    max_scan: int = 5000,
    audio_root: Path | None = None,
) -> list[dict]:
    by_group: dict[str, list[dict]] = defaultdict(list)
    with manifest.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= max_scan:
                break
            if not line.strip():
                continue
            doc = json.loads(line)
            inp = doc.get("input") or {}
            wav = inp.get("audio_path") or doc.get("WavPath")
            text = str((doc.get("reference") or {}).get("text") or doc.get("text") or "").strip()
            if not wav or not text:
                # msswift / messages style (AMI, CHiME-6 L0)
                for msg in doc.get("messages") or []:
                    for part in msg.get("content") or []:
                        if isinstance(part, dict) and part.get("audio_path"):
                            wav = part["audio_path"]
                        if isinstance(part, dict) and part.get("text"):
                            text = str(part["text"]).strip()
            if not wav or not text:
                continue
            wav_path = Path(wav)
            if not wav_path.is_file() and audio_root is not None:
                wav_path = audio_root / wav
            if not wav_path.is_file():
                continue
            wav = str(wav_path)
            by_group[group_key(wav)].append(
                {
                    "WavPath": wav,
                    "text": text,
                    "sample_id": doc.get("sample_id") or doc.get("id"),
                    "sort": sort_key(wav),
                }
            )

    out_rows: list[dict] = []
    audio_dir = AUDIO_ROOT / dataset
    groups = list(by_group.keys())
    # interleave groups
    cursors = {g: 0 for g in groups}
    for g in groups:
        by_group[g].sort(key=lambda r: r["sort"])

    while len(out_rows) < limit and groups:
        progressed = False
        for g in list(groups):
            rows = by_group[g]
            i = cursors[g]
            if i >= len(rows):
                continue
            pack = []
            paths = []
            total = 0.0
            j = i
            while j < len(rows) and total < MIN_DUR:
                path = Path(rows[j]["WavPath"])
                try:
                    dur = audio_duration(path)
                except Exception:
                    j += 1
                    continue
                if total + dur > MAX_DUR and pack:
                    break
                pack.append(rows[j])
                paths.append(path)
                total += dur
                j += 1
                if total >= MIN_DUR and total <= MAX_DUR:
                    break
            cursors[g] = j if j > i else i + 1
            if not pack or total < MIN_DUR or total > MAX_DUR:
                continue
            text = join_text(pack)
            if not text:
                continue
            sample_id = f"{dataset}_lf_{len(out_rows):04d}"
            dst = audio_dir / f"{sample_id}.wav"
            if not dst.is_file():
                print(
                    f"[concat] {dataset} {sample_id} {total:.1f}s "
                    f"n={len(pack)} group={g}"
                )
                ffmpeg_concat(paths, dst)
            dur = audio_duration(dst)
            if not (MIN_DUR - 0.5 <= dur <= MAX_DUR + 0.5):
                print(f"[skip] {sample_id} duration {dur:.2f}s out of range")
                continue
            out_rows.append(
                {
                    "sample_id": sample_id,
                    "dataset": dataset,
                    "capability": "long_form_asr",
                    "protocol": "long_form_asr_window_30_60",
                    "WavPath": str(dst),
                    "text": text,
                    "audio_duration_seconds": round(dur, 3),
                    "n_utterances": len(pack),
                    "source_group": g,
                    "window_min_seconds": MIN_DUR,
                    "window_max_seconds": MAX_DUR,
                }
            )
            progressed = True
            if len(out_rows) >= limit:
                break
        if not progressed:
            break
    return out_rows


def write_manifest(dataset: str, rows: list[dict]) -> Path:
    path = NATIVE / f"{dataset}_long_form_30_60.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def main(only: set[str] | None = None) -> None:
    AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {}
    want = only

    for dataset, index in UTT_INDEXES.items():
        if want and dataset not in want:
            continue
        rows = build_from_utt_index(dataset, index, LIMIT)
        path = write_manifest(dataset, rows)
        durs = [r["audio_duration_seconds"] for r in rows]
        summary[dataset] = {
            "n": len(rows),
            "dur": (min(durs), max(durs)) if durs else None,
            "manifest": str(path),
        }
        print(f"[done] {dataset} n={len(rows)} dur={summary[dataset]['dur']}")

    if not want or "ami" in want:
        ami_rows = build_from_utt_files(
            "ami",
            AMI_MANIFEST,
            group_key=_group_key_ami,
            sort_key=_ami_sort_key,
            audio_root=AMI_ROOT,
        )
        summary["ami"] = {
            "n": len(ami_rows),
            "dur": (
                min(r["audio_duration_seconds"] for r in ami_rows),
                max(r["audio_duration_seconds"] for r in ami_rows),
            )
            if ami_rows
            else None,
            "manifest": str(write_manifest("ami", ami_rows)),
        }
        print(f"[done] ami n={len(ami_rows)} dur={summary['ami']['dur']}")

    if not want or "chime_6" in want:
        chime_rows = build_from_utt_files(
            "chime_6",
            CHIME_MANIFEST,
            group_key=_group_key_chime,
            sort_key=_chime_sort_key,
            audio_root=CHIME_ROOT,
        )
        summary["chime_6"] = {
            "n": len(chime_rows),
            "dur": (
                min(r["audio_duration_seconds"] for r in chime_rows),
                max(r["audio_duration_seconds"] for r in chime_rows),
            )
            if chime_rows
            else None,
            "manifest": str(write_manifest("chime_6", chime_rows)),
        }
        print(f"[done] chime_6 n={len(chime_rows)} dur={summary['chime_6']['dur']}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    missing = [k for k, v in summary.items() if v["n"] < LIMIT]
    if missing:
        raise SystemExit(f"insufficient windows for {missing}: {summary}")


if __name__ == "__main__":
    import sys

    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")} or None
    main(only)
