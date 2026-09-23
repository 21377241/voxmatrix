#!/usr/bin/env python3
"""Build long_form utterance timelines from L0 (TextGrid / official asr jsonl).

Output: mesh_eval/data/native/long_form_timelines/<bench>.jsonl
Fields: src_audio, start_time, end_time, text
Used by build_long_form_windows.py — does not read utt_clips.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "mesh_eval" / "data" / "native" / "long_form_timelines"


def parse_textgrid(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    intervals = []
    # Match intervals [n]: xmin xmax text
    pattern = re.compile(
        r"intervals\s*\[\d+\]\s*:\s*xmin\s*=\s*([0-9.]+)\s*xmax\s*=\s*([0-9.]+)\s*text\s*=\s*\"([^\"]*)\"",
        re.MULTILINE,
    )
    for m in pattern.finditer(text):
        start, end, lab = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if not lab or lab in {"<%>", "<sil>", "<noise>"}:
            continue
        intervals.append((start, end, lab))
    return intervals


def write_rows(name: str, rows: list[dict]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.jsonl"
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def build_aishell_4() -> int:
    root = Path("/mnt/afs/eval_data/03_multi_speaker/AISHELL-4/test")
    wav_dir = root / "wav"
    tg_dir = root / "TextGrid"
    rows = []
    for tg in sorted(tg_dir.glob("*.TextGrid")):
        wav = wav_dir / f"{tg.stem}.flac"
        if not wav.is_file():
            continue
        for start, end, text in parse_textgrid(tg):
            rows.append(
                {
                    "src_audio": str(wav),
                    "start_time": start,
                    "end_time": end,
                    "text": text,
                    "dataset": "aishell_4",
                }
            )
    write_rows("aishell_4", rows)
    return len(rows)


def build_alimeeting() -> int:
    # Prefer Test_Ali far textgrids
    roots = [
        Path("/mnt/afs/eval_data/03_multi_speaker/AliMeeting/Test_Ali/Test_Ali_far"),
        Path("/mnt/afs/eval_data/03_multi_speaker/AliMeeting/Eval_Ali/Eval_Ali_far"),
    ]
    rows = []
    for root in roots:
        audio_dir = root / "audio_dir"
        tg_dir = root / "textgrid_dir"
        if not tg_dir.is_dir():
            continue
        for tg in sorted(tg_dir.glob("*.TextGrid")):
            # audio name often matches stem
            wav = audio_dir / f"{tg.stem}.wav"
            if not wav.is_file():
                cands = list(audio_dir.glob(f"{tg.stem}*.wav"))
                wav = cands[0] if cands else wav
            if not wav.is_file():
                continue
            for start, end, text in parse_textgrid(tg):
                rows.append(
                    {
                        "src_audio": str(wav),
                        "start_time": start,
                        "end_time": end,
                        "text": text,
                        "dataset": "alimeeting",
                    }
                )
    write_rows("alimeeting", rows)
    return len(rows)


def build_sbcsae() -> int:
    src = Path("/mnt/afs/eval_data/benchmarks/SBCSAE_Public_Speech/manifests/asr.jsonl")
    rows = []
    with src.open(encoding="utf-8") as handle:
        for line in handle:
            doc = json.loads(line)
            if doc.get("scenario_secondary") != "kitchen_dining":
                continue
            audio = doc.get("audio")
            start = doc.get("start_seconds")
            end = doc.get("end_seconds")
            text = str(doc.get("text") or "").strip()
            if not audio or start is None or end is None or not text:
                continue
            if not Path(audio).is_file():
                continue
            rows.append(
                {
                    "src_audio": str(audio),
                    "start_time": float(start),
                    "end_time": float(end),
                    "text": text,
                    "dataset": "sbcsae",
                }
            )
    write_rows("sbcsae", rows)
    return len(rows)


def build_misp_from_l0_index() -> int:
    """MISP L0: audiobench_asr_norm.json with pcm + timestamps."""
    root = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/MISP")
    src = root / "audiobench_asr_norm.json"
    rows = []
    if not src.is_file():
        write_rows("misp", rows)
        return 0
    data = json.loads(src.read_text(encoding="utf-8"))
    for doc in data:
        start = doc.get("start_time")
        end = doc.get("end_time")
        audios = doc.get("audios") or []
        text = ""
        for msg in doc.get("messages") or []:
            if msg.get("role") == "assistant":
                text = str(msg.get("content") or "").strip()
        if start is None or end is None or not audios or not text:
            continue
        audio_p = root / audios[0]
        if not audio_p.is_file():
            continue
        rows.append(
            {
                "src_audio": str(audio_p),
                "start_time": float(start),
                "end_time": float(end),
                "text": text,
                "dataset": "misp",
            }
        )
    write_rows("misp", rows)
    return len(rows)


def main() -> None:
    counts = {
        "aishell_4": build_aishell_4(),
        "alimeeting": build_alimeeting(),
        "sbcsae": build_sbcsae(),
        "misp": build_misp_from_l0_index(),
    }
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    missing = [k for k, n in counts.items() if n < 10]
    if missing:
        raise SystemExit(f"timeline too small: {missing} {counts}")


if __name__ == "__main__":
    main()
