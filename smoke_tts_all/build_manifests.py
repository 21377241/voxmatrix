#!/usr/bin/env python3
"""Build speech_generation smoke: 10 LibriTTS test-clean texts (content_acc only)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
N = 10
LIBRI = Path("/mnt/afs/eval_data/01_basic_transcription/LibriTTS/test-clean")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_stride(rows: list, n: int = N) -> list:
    if len(rows) <= n:
        return rows
    step = max(1, len(rows) // n)
    return [rows[i * step] for i in range(n)][:n]


def main() -> None:
    rows: list[dict] = []
    for txt in sorted(LIBRI.glob("*/*/*.normalized.txt")):
        text = txt.read_text(encoding="utf-8").strip()
        if not text or len(text.split()) < 4 or len(text.split()) > 40:
            continue
        wav = txt.with_suffix("").with_suffix(".wav")
        # file is xxx.normalized.txt -> stem ends with .normalized; wav is sibling without that
        wav = Path(str(txt).replace(".normalized.txt", ".wav"))
        sid = wav.stem if wav.suffix == ".wav" else txt.name.replace(".normalized.txt", "")
        if not Path(str(txt).replace(".normalized.txt", ".wav")).is_file():
            continue
        ref_wav = Path(str(txt).replace(".normalized.txt", ".wav"))
        rows.append(
            {
                "sample_id": f"libritts_{sid}",
                "dataset": "libritts",
                "subset": "test-clean",
                "protocol": "tts_content_only",
                "text": text,
                "language": "en",
                "ref_wav": str(ref_wav),
            }
        )
    picked = sample_stride(rows, N)
    if len(picked) < N:
        raise RuntimeError(f"libritts: only {len(picked)}")
    write_jsonl(ROOT / "libritts" / "manifest.jsonl", picked)
    print(f"[tts] wrote {len(picked)}")


if __name__ == "__main__":
    main()
