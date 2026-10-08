#!/usr/bin/env python3
"""Build ASR eval samples that point at real data (not utt_clips / FullEval / smoke).

For each runnable ASR benchmark, write mesh_eval/data/native/asr_<bench>.jsonl
(limit=10) with WavPath resolving to L0 short audio, or a crop under
asr_eval_clips/ derived from L0 long audio + timestamps.

utt_clips remains an optional side ledger for slicing stats; capability eval
does not load it.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NATIVE = ROOT / "mesh_eval" / "data" / "native"
CLIPS = NATIVE / "asr_eval_clips"
FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"
LIMIT = 10
MIN_UTT_SECONDS = 1.5

AISHELL1_JSONL = Path(
    "/mnt/afs/oss_data/datasets/01_basic_transcription/AIShell-1/aishell1_test_asr.jsonl"
)
AISHELL1_ROOT = Path("/mnt/afs/oss_data/datasets/01_basic_transcription/AIShell-1")
AISHELL5_JSONL = Path(
    "/mnt/afs/oss_data/datasets/02_complex_acoustic/AISHELL-5/AISHELL-5_eval1_msswift.jsonl"
)
AISHELL5_ROOT = Path("/mnt/afs/oss_data/datasets/02_complex_acoustic/AISHELL-5")
LIBRI_ROOT = Path("/mnt/afs/eval_data/01_basic_transcription/LibriSpeech/test-clean")
MAGIC_PART = Path("/mnt/afs/eval_data/04_dialogue_slu/MagicHub/DataPartition/test.tsv")
MAGIC_WAV = Path("/mnt/afs/eval_data/04_dialogue_slu/MagicHub/MDT2021S003/WAV")
MAGIC_TXT = Path("/mnt/afs/eval_data/04_dialogue_slu/MagicHub/MDT2021S003/TXT")
SBCSAE_JSONL = Path("/mnt/afs/eval_data/benchmarks/SBCSAE_Public_Speech/manifests/asr.jsonl")
CV_EN_ROOT = Path(
    "/mnt/afs/oss_data/datasets/06_english_spoken_accent/CommonVoice_26/"
    "raw/extracted/26.0/en/cv-corpus-26.0-2026-06-12/en"
)
# Official TEST_NET shorts are not unpacked on disk (aes.tgz only); reuse the
# already-decoded segment wavs once into asr_eval_clips as L0-derived eval audio.
WENET_SMOKE_AUDIO = Path(
    "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/wenetspeech/audio"
)
WENET_SMOKE_MANIFEST = Path(
    "/mnt/afs/users/wangyl/VoxMatrix/smoke_asr_extra/wenetspeech/manifest.jsonl"
)


def ffmpeg_crop(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if end <= start:
        raise ValueError(f"bad interval {start}-{end}")
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
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 500:
        raise RuntimeError(f"crop failed {src} [{start},{end}]\n{proc.stderr[-800:]}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def build_aishell_1(limit: int = LIMIT) -> list[dict]:
    rows = []
    with AISHELL1_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            doc = json.loads(line)
            rel = None
            text = ""
            for msg in doc.get("messages") or []:
                for part in msg.get("content") or []:
                    if isinstance(part, dict) and part.get("audio_path"):
                        rel = part["audio_path"]
                    if isinstance(part, dict) and part.get("text"):
                        text = str(part["text"]).strip()
            if not rel or not text:
                continue
            wav = AISHELL1_ROOT / rel
            if not wav.is_file():
                continue
            rows.append(
                {
                    "sample_id": doc.get("id") or f"aishell_1_{len(rows):04d}",
                    "dataset": "aishell_1",
                    "capability": "asr",
                    "WavPath": str(wav),
                    "text": text,
                    "source_index": str(AISHELL1_JSONL),
                }
            )
    return rows


def stride_pick(candidates: list, limit: int) -> list:
    """Evenly spaced picks so samples span sessions instead of the first file."""
    step = max(1, len(candidates) // limit)
    return candidates[::step][:limit]


def build_aishell_5(limit: int = LIMIT) -> list[dict]:
    candidates = []
    with AISHELL5_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            meta = doc.get("metadata") or {}
            start = meta.get("start_time")
            end = meta.get("end_time")
            audios = doc.get("audios") or []
            text = ""
            for msg in doc.get("messages") or []:
                if msg.get("role") == "assistant":
                    text = str(msg.get("content") or "").strip()
            if start is None or end is None or not audios or not text:
                continue
            if float(end) - float(start) < MIN_UTT_SECONDS:
                continue
            candidates.append((doc, float(start), float(end), audios, text))
    rows = []
    clip_dir = CLIPS / "aishell_5"
    for doc, start, end, audios, text in stride_pick(candidates, limit):
        src = AISHELL5_ROOT / audios[0]
        if not src.is_file():
            continue
        sid = str(doc.get("id") or f"aishell_5_{len(rows):04d}")
        dst = clip_dir / f"{sid}.wav"
        if not dst.is_file():
            ffmpeg_crop(src, dst, start, end)
        rows.append(
            {
                "sample_id": sid,
                "dataset": "aishell_5",
                "capability": "asr",
                "WavPath": str(dst),
                "text": text,
                "src_audio": str(src),
                "start_time": start,
                "end_time": end,
                "source_index": str(AISHELL5_JSONL),
            }
        )
    return rows


def build_librispeech(limit: int = LIMIT) -> list[dict]:
    rows = []
    for trans in sorted(LIBRI_ROOT.glob("*/*/*.trans.txt")):
        with trans.open(encoding="utf-8") as handle:
            for line in handle:
                if len(rows) >= limit:
                    return rows
                parts = line.strip().split(maxsplit=1)
                if len(parts) != 2:
                    continue
                utt_id, text = parts
                flac = trans.parent / f"{utt_id}.flac"
                if not flac.is_file():
                    continue
                rows.append(
                    {
                        "sample_id": utt_id,
                        "dataset": "librispeech",
                        "capability": "asr",
                        "WavPath": str(flac),
                        "text": text,
                        "source_index": str(trans),
                        "subset": "test-clean",
                    }
                )
    return rows


def build_magicdata(limit: int = LIMIT) -> list[dict]:
    # DataPartition/test.tsv lists dialogue wav basenames; TXT has [start,end] lines.
    names = []
    with MAGIC_PART.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i == 0 and line.startswith("../"):
                continue
            name = line.strip().split("\t")[0].strip()
            if name.endswith(".wav"):
                names.append(name)
    candidates = []
    for name in names:
        wav = MAGIC_WAV / name
        txt = MAGIC_TXT / name.replace(".wav", ".txt")
        if not wav.is_file() or not txt.is_file():
            continue
        with txt.open(encoding="utf-8") as handle:
            for j, line in enumerate(handle):
                match = re.match(
                    r"\[([0-9.]+),([0-9.]+)\]\t[^\t]+\t[^\t]+\t(.+)$", line.strip()
                )
                if not match:
                    continue
                start, end, text = float(match.group(1)), float(match.group(2)), match.group(3).strip()
                text = re.sub(r"\[.*?\]", "", text).replace("[+]", "").strip()
                if not text or end - start < MIN_UTT_SECONDS:
                    continue
                candidates.append((name, j, wav, txt, start, end, text))
    rows = []
    clip_dir = CLIPS / "magicdata_ramc"
    for name, j, wav, txt, start, end, text in stride_pick(candidates, limit):
        sid = f"magicdata_ramc_{Path(name).stem}_{j:04d}"
        dst = clip_dir / f"{sid}.wav"
        if not dst.is_file():
            ffmpeg_crop(wav, dst, start, end)
        rows.append(
            {
                "sample_id": sid,
                "dataset": "magicdata_ramc",
                "capability": "asr",
                "WavPath": str(dst),
                "text": text,
                "src_audio": str(wav),
                "start_time": start,
                "end_time": end,
                "source_index": str(txt),
            }
        )
    return rows


def build_sbcsae(limit: int = LIMIT) -> list[dict]:
    candidates = []
    with SBCSAE_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            if doc.get("scenario_secondary") != "kitchen_dining":
                continue
            text = str(doc.get("text") or "").strip()
            start = doc.get("start_seconds")
            end = doc.get("end_seconds")
            if not text or not doc.get("audio") or start is None or end is None:
                continue
            if float(end) - float(start) < MIN_UTT_SECONDS:
                continue
            candidates.append((doc, Path(doc["audio"]), float(start), float(end), text))
    rows = []
    clip_dir = CLIPS / "sbcsae"
    for doc, src, start, end, text in stride_pick(candidates, limit):
        if not src.is_file():
            continue
        sid = str(doc.get("id") or f"sbcsae_{len(rows):04d}")
        dst = clip_dir / f"{sid}.wav"
        if not dst.is_file():
            ffmpeg_crop(src, dst, start, end)
        rows.append(
            {
                "sample_id": sid,
                "dataset": "sbcsae",
                "capability": "asr",
                "WavPath": str(dst),
                "text": text,
                "src_audio": str(src),
                "start_time": start,
                "end_time": end,
                "source_index": str(SBCSAE_JSONL),
            }
        )
    return rows


def build_common_voice_en(limit: int = LIMIT) -> list[dict]:
    # smoke ASR #6: English CV clips (not zh-HK / multi-locale annotation draw).
    tsv = CV_EN_ROOT / "test.tsv"
    clips = CV_EN_ROOT / "clips"
    rows = []
    with tsv.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if len(rows) >= limit:
                break
            path = row.get("path") or ""
            sentence = (row.get("sentence") or "").strip()
            wav = clips / path
            if not sentence or not wav.is_file():
                continue
            rows.append(
                {
                    "sample_id": f"cv_en_{Path(path).stem}",
                    "dataset": "common_voice",
                    "capability": "asr",
                    "WavPath": str(wav),
                    "text": sentence,
                    "locale": "en",
                    "source_index": str(tsv),
                }
            )
    return rows


def build_wenetspeech(limit: int = LIMIT) -> list[dict]:
    """TEST_NET segments: copy decoded wavs into asr_eval_clips (L0 aes not unpacked)."""
    rows = []
    clip_dir = CLIPS / "wenetspeech"
    clip_dir.mkdir(parents=True, exist_ok=True)
    with WENET_SMOKE_MANIFEST.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            doc = json.loads(line)
            src = Path(doc["WavPath"])
            if not src.is_file():
                continue
            sid = str(doc.get("sample_id") or src.stem)
            dst = clip_dir / src.name
            if not dst.is_file():
                shutil.copy2(src, dst)
            rows.append(
                {
                    "sample_id": sid,
                    "dataset": "wenetspeech",
                    "capability": "asr",
                    "WavPath": str(dst),
                    "text": str(doc.get("text") or "").strip(),
                    "subset": "TEST_NET",
                    "source_index": "WenetSpeech TEST_NET (decoded segment wavs)",
                }
            )
    return rows


BUILDERS = {
    "aishell_1": build_aishell_1,
    "aishell_5": build_aishell_5,
    "librispeech": build_librispeech,
    "magicdata_ramc": build_magicdata,
    "sbcsae": build_sbcsae,
    "common_voice": build_common_voice_en,
    "wenetspeech": build_wenetspeech,
}


def main(only: set[str] | None = None) -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, builder in BUILDERS.items():
        if only and name not in only:
            continue
        rows = builder(LIMIT)
        path = NATIVE / f"asr_{name}.jsonl"
        write_jsonl(path, rows)
        summary[name] = {"n": len(rows), "manifest": str(path)}
        print(f"[done] {name} n={len(rows)} -> {path}")
        if len(rows) < LIMIT:
            raise SystemExit(f"insufficient samples for {name}: {len(rows)}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import sys

    only = {arg for arg in sys.argv[1:] if not arg.startswith("-")} or None
    main(only)
