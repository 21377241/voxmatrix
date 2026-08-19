"""为跨 Task 试点构建小样 raw_index。"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def source_data_root() -> str:
    """Return the canonical source root while accepting the legacy option."""
    return (
        os.environ.get("VOXMATRIX_SOURCE_DATA_ROOT")
        or os.environ.get("ULTRAEVAL_SOURCE_DATA_ROOT")
        or "data/source_datasets"
    )


def index_voxceleb1(root: Path, limit: int, seed: int = 42) -> list[dict]:
    """从说话人目录构造同人/异人验证对。"""
    wav_root = root / "juliuscn" / "voxceleb" / "vox1" / "wav"
    if not wav_root.is_dir():
        raise FileNotFoundError(wav_root)
    speakers = sorted([p for p in wav_root.iterdir() if p.is_dir()])
    rng = random.Random(seed)
    pairs = []
    # 同人
    for spk in speakers:
        utts = sorted(spk.rglob("*.wav"))
        if len(utts) < 2:
            continue
        a, b = rng.sample(utts, 2)
        pairs.append(
            {
                "id": f"{spk.name}_same_{a.stem}_{b.stem}",
                "audio_path_a": str(a),
                "audio_path_b": str(b),
                "audio_path": str(a),
                "same_speaker": True,
                "split": "test",
            }
        )
        if len(pairs) >= limit // 2:
            break
    # 异人
    while len(pairs) < limit and len(speakers) >= 2:
        s1, s2 = rng.sample(speakers, 2)
        u1 = sorted(s1.rglob("*.wav"))
        u2 = sorted(s2.rglob("*.wav"))
        if not u1 or not u2:
            continue
        a, b = rng.choice(u1), rng.choice(u2)
        pairs.append(
            {
                "id": f"{s1.name}_{s2.name}_diff_{a.stem}_{b.stem}",
                "audio_path_a": str(a),
                "audio_path_b": str(b),
                "audio_path": str(a),
                "same_speaker": False,
                "split": "test",
            }
        )
    return pairs[:limit]


def index_slurp(root: Path, limit: int) -> list[dict]:
    path = root / "SLURP_test_msswift.jsonl"
    if not path.exists():
        path = root / "SLURP_devel_msswift.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            audios = obj.get("audios") or []
            if not audios:
                continue
            assistant = ""
            for m in obj.get("messages") or []:
                if m.get("role") == "assistant":
                    assistant = m.get("content") or ""
            try:
                anno = json.loads(assistant) if isinstance(assistant, str) else assistant
            except json.JSONDecodeError:
                anno = {}
            audio_rel = audios[0]
            audio_path = str(root / audio_rel)
            rows.append(
                {
                    "id": Path(audio_rel).stem,
                    "audio_path": audio_path,
                    "record_path": audio_path,
                    "text": anno.get("transcription", ""),
                    "scenario": anno.get("scenario") or anno.get("intent", ""),
                    "action": anno.get("action", ""),
                    "intent": anno.get("intent", ""),
                    "entities": anno.get("entities", []),
                    "split": "test",
                }
            )
            if len(rows) >= limit:
                break
    return rows


def index_libritts(root: Path, limit: int) -> list[dict]:
    path = root / "libritts_train_clean_100_tts.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = None
            audio_rel = None
            for m in obj.get("messages") or []:
                for c in m.get("content") or []:
                    if c.get("text"):
                        text = c["text"]
                    if c.get("audio_path"):
                        audio_rel = c["audio_path"]
            if not text or not audio_rel:
                continue
            rows.append(
                {
                    "id": obj.get("id") or Path(audio_rel).stem,
                    "text": text,
                    "audio_path": str(root / audio_rel),
                    "split": "train",
                    "subset": "train-clean-100",
                }
            )
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["voxceleb1", "slurp", "libritts"])
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()

    source_root = Path(
        source_data_root()
    ).expanduser()
    defaults = {
        "voxceleb1": source_root
        / "03_multi_speaker"
        / "VoxCeleb_1_2_multi_speaker_multi_speaker",
        "slurp": source_root / "04_dialogue_slu" / "SLURP_repo",
        "libritts": source_root / "01_basic_transcription" / "LibriTTS",
    }
    root = args.root or defaults[args.dataset]
    if args.dataset == "voxceleb1":
        rows = index_voxceleb1(root, args.limit)
    elif args.dataset == "slurp":
        rows = index_slurp(root, args.limit)
    else:
        rows = index_libritts(root, args.limit)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{args.dataset} 索引 {len(rows)} 条 → {args.output}")


if __name__ == "__main__":
    main()
