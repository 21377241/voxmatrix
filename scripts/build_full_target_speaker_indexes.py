#!/usr/bin/env python3
"""Build the full target_speaker index from AISHELL-5 Eval1 + Eval2 (adapted).

AISHELL-5 has no enrollment / command / should_execute labels, so pairs are
constructed here:

* Each session is one car with two speakers (near-field DA0x headsets) and
  synchronised far-field mics (DX01C01..DX04C01) whose TextGrid holds both
  speakers' utterances.
* Enrollment: near-field DA audio of the target speaker from session s1,
  single-speaker utterances (no other-speaker overlap within +-0.1 s)
  concatenated with 0.3 s silence to 5-10 s. One clip per (speaker, session).
* Command: one far-field DX01C01 utterance (1.0-8 s, >=3 characters, no
  other-speaker overlap within +-0.1 s) from a different session s2 != s1.
* Positive: command by the target speaker. Negative: command by the other
  speaker of the same car (same session s2), i.e. the hardest in-car impostor.
* Each speaker gets n positives and n negatives, n = min(PAIRS_PER_SPEAKER,
  own pool, partner pool): in-car speech overlaps heavily, so a few speakers
  have fewer than 50 clean commands. A command utterance is used at most once
  as a positive (for its speaker) and once as a negative (for the partner).

Audio goes to /mnt/afs/eval_data/target_speaker/aishell_5/, the index to
mesh_eval/data/full/target_speaker/, and a balanced 10-pair Native subset to
mesh_eval/data/native/target_speaker_aishell_5.jsonl.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from suites.native.windowed_ondevice import Seg, parse_textgrid_spk_text  # noqa: E402

AISHELL5 = Path("/mnt/afs/oss_data/datasets/02_complex_acoustic/AISHELL-5")
AUDIO_OUT = Path("/mnt/afs/eval_data/target_speaker/aishell_5")
FULL_OUT = ROOT / "mesh_eval" / "data" / "full" / "target_speaker" / "target_speaker_aishell_5.jsonl"
NATIVE_OUT = ROOT / "mesh_eval" / "data" / "native" / "target_speaker_aishell_5.jsonl"
NATIVE_LEGACY_DIR = ROOT / "mesh_eval" / "data" / "native" / "legacy_target_speaker"

SPLITS = ("Eval1", "Eval2")
FAR_FIELD = "DX01C01"
SR = 16000
GUARD = 0.1
ENROLL_MIN, ENROLL_MAX = 5.0, 10.0
ENROLL_UTT_MIN, ENROLL_UTT_MAX = 1.0, 8.0
CMD_MIN, CMD_MAX = 1.0, 8.0
CMD_MIN_CHARS = 3
GAP = 0.3
PAIRS_PER_SPEAKER = 50
_CHAR_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")


@dataclass
class SessionData:
    split: str
    session: str
    near: dict[str, Path]
    segs: list[Seg]


def load_sessions() -> list[SessionData]:
    sessions: list[SessionData] = []
    for split in SPLITS:
        for sdir in sorted((AISHELL5 / split).iterdir()):
            far_tg = sdir / f"{FAR_FIELD}.TextGrid"
            if not far_tg.is_file() or not (sdir / f"{FAR_FIELD}.wav").is_file():
                continue
            near: dict[str, Path] = {}
            for tg in sorted(sdir.glob("DA*.TextGrid")):
                speakers = {s.speaker for s in parse_textgrid_spk_text(tg)}
                wav = tg.with_suffix(".wav")
                if len(speakers) == 1 and wav.is_file():
                    near[speakers.pop()] = wav
            segs = parse_textgrid_spk_text(far_tg)
            if len(near) == 2 and {s.speaker for s in segs} == set(near):
                sessions.append(SessionData(split, sdir.name, near, segs))
    return sessions


def isolated(seg: Seg, segs: list[Seg]) -> bool:
    return not any(
        o.speaker != seg.speaker and o.end > seg.start - GUARD and o.start < seg.end + GUARD
        for o in segs
    )


def n_chars(text: str) -> int:
    return len(_CHAR_RE.findall(text))


def read_span(path: Path, start: float, end: float) -> np.ndarray:
    audio, sr = sf.read(str(path), start=int(round(start * SR)), stop=int(round(end * SR)), dtype="float32")
    if sr != SR:
        raise RuntimeError(f"unexpected sr={sr} for {path}")
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")


def build_enrollment(sess: SessionData, speaker: str, rng: random.Random) -> dict | None:
    utts = [
        s for s in sess.segs
        if s.speaker == speaker
        and ENROLL_UTT_MIN <= s.end - s.start <= ENROLL_UTT_MAX
        and isolated(s, sess.segs)
    ]
    if not utts:
        return None
    start_idx = rng.randrange(len(utts))
    chosen: list[Seg] = []
    total = 0.0
    for seg in utts[start_idx:] + utts[:start_idx]:
        dur = seg.end - seg.start
        extra = dur + (GAP if chosen else 0.0)
        if total + extra > ENROLL_MAX:
            continue
        chosen.append(seg)
        total += extra
        if total >= ENROLL_MIN:
            break
    if total < ENROLL_MIN:
        return None
    chosen.sort(key=lambda s: s.start)
    gap = np.zeros(int(GAP * SR), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(chosen):
        if i:
            parts.append(gap)
        parts.append(read_span(sess.near[speaker], seg.start, seg.end))
    path = AUDIO_OUT / "enroll" / f"{sess.split.lower()}_{sess.session}_{speaker}.wav"
    write_wav(path, np.concatenate(parts))
    return {
        "path": str(path),
        "duration": round(total, 3),
        "n_utts": len(chosen),
        "channel": sess.near[speaker].stem,
        "segments": [[round(s.start, 3), round(s.end, 3)] for s in chosen],
    }


def command_pool(sess: SessionData, speaker: str) -> list[Seg]:
    return [
        s for s in sess.segs
        if s.speaker == speaker
        and CMD_MIN <= s.end - s.start <= CMD_MAX
        and n_chars(s.text) >= CMD_MIN_CHARS
        and isolated(s, sess.segs)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-speaker", type=int, default=PAIRS_PER_SPEAKER)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    sessions = load_sessions()
    by_speaker: dict[str, list[SessionData]] = defaultdict(list)
    partner: dict[str, str] = {}
    for sess in sessions:
        a, b = sorted(sess.near)
        partner[a], partner[b] = b, a
        by_speaker[a].append(sess)
        by_speaker[b].append(sess)
    speakers = sorted(by_speaker)

    enroll: dict[tuple[str, str, str], dict] = {}
    for spk in speakers:
        for sess in by_speaker[spk]:
            clip = build_enrollment(sess, spk, rng)
            if clip:
                enroll[(spk, sess.split, sess.session)] = clip

    # Each (speaker, session) pool has an independent shuffle per role, so an
    # utterance is used at most once as a positive and once as a negative.
    pools: dict[tuple[str, str, str, str], list[Seg]] = {}
    pool_size: Counter = Counter()
    for spk in speakers:
        for sess in by_speaker[spk]:
            cands = command_pool(sess, spk)
            pool_size[spk] += len(cands)
            for role in ("pos", "neg"):
                shuffled = list(cands)
                rng.shuffle(shuffled)
                pools[(spk, sess.split, sess.session, role)] = shuffled
    n_pairs = {
        spk: min(args.pairs_per_speaker, pool_size[spk], pool_size[partner[spk]])
        for spk in speakers
    }

    rows_by_speaker: dict[str, list[dict]] = defaultdict(list)
    shortages: list[str] = []
    for spk in speakers:
        sess_list = [s for s in by_speaker[spk] if (spk, s.split, s.session) in enroll]
        n_sess = len(sess_list)
        for i in range(n_pairs[spk]):
            s1 = sess_list[i % n_sess]
            for kind in ("pos", "neg"):
                cmd_spk = spk if kind == "pos" else partner[spk]
                pool_kind = "pos" if kind == "pos" else "neg"
                seg = None
                for offset in range(1, n_sess):
                    s2 = sess_list[(i + offset + (i // n_sess)) % n_sess]
                    if s2 is s1:
                        continue
                    pool = pools[(cmd_spk, s2.split, s2.session, pool_kind)]
                    if pool:
                        seg = pool.pop()
                        break
                if seg is None:
                    shortages.append(f"{spk} {kind} #{i}")
                    continue
                sid = f"aishell5_ts_{s1.split.lower()}_{spk}_{kind}_{i:02d}"
                cmd_path = AUDIO_OUT / "cmd" / (
                    f"{s2.split.lower()}_{s2.session}_{cmd_spk}_{int(round(seg.start * 1000)):08d}.wav"
                )
                if not cmd_path.is_file():
                    write_wav(cmd_path, read_span(s2.near[cmd_spk].parent / f"{FAR_FIELD}.wav", seg.start, seg.end))
                clip = enroll[(spk, s1.split, s1.session)]
                rows_by_speaker[spk].append(
                    {
                        "sample_id": sid,
                        "dataset": "aishell_5",
                        "subset": s1.split.lower(),
                        "protocol": "enrollment_command_pair",
                        "WavPath": clip["path"],
                        "WavPath2": str(cmd_path),
                        "text": kind == "pos",
                        "should_execute": kind == "pos",
                        "enroll_speaker": spk,
                        "command_speaker": cmd_spk,
                        "capability": "target_speaker",
                        "pair_type": "positive" if kind == "pos" else "negative_same_car",
                        "enroll_session": s1.session,
                        "enroll_channel": clip["channel"],
                        "enroll_duration": clip["duration"],
                        "enroll_n_utts": clip["n_utts"],
                        "enroll_segments": clip["segments"],
                        "command_session": s2.session,
                        "command_channel": FAR_FIELD,
                        "command_start": round(seg.start, 3),
                        "command_end": round(seg.end, 3),
                        "command_duration": round(seg.end - seg.start, 3),
                        "command_text": seg.text,
                        "source": "AISHELL-5 adapted (scripts/build_full_target_speaker_indexes.py)",
                    }
                )

    for spk in speakers:
        by_label: dict[bool, list[dict]] = defaultdict(list)
        for row in rows_by_speaker[spk]:
            by_label[row["should_execute"]].append(row)
        keep = min(len(by_label[True]), len(by_label[False]))
        kept = {id(r) for r in by_label[True][:keep] + by_label[False][:keep]}
        rows_by_speaker[spk] = [r for r in rows_by_speaker[spk] if id(r) in kept]

    # Interleave so any prefix stays balanced across speakers and labels.
    rows: list[dict] = []
    longest = max(len(v) for v in rows_by_speaker.values())
    for j in range(longest):
        for spk in speakers:
            if j < len(rows_by_speaker[spk]):
                rows.append(rows_by_speaker[spk][j])

    FULL_OUT.parent.mkdir(parents=True, exist_ok=True)
    FULL_OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    # Native: 10 distinct target speakers, alternating positive / negative, first pair each.
    native: list[dict] = []
    for k, spk in enumerate(speakers[:10]):
        want = "positive" if k % 2 == 0 else "negative_same_car"
        native.append(next(r for r in rows_by_speaker[spk] if r["pair_type"] == want))
    if NATIVE_OUT.is_file() and not (NATIVE_LEGACY_DIR / NATIVE_OUT.name).is_file():
        NATIVE_LEGACY_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(NATIVE_OUT, NATIVE_LEGACY_DIR / NATIVE_OUT.name)
    NATIVE_OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in native), encoding="utf-8")

    summary = {
        "sessions": Counter(s.split for s in sessions),
        "speakers": speakers,
        "partners": {s: partner[s] for s in speakers},
        "enroll_clips": len(enroll),
        "rows": len(rows),
        "labels": Counter(r["pair_type"] for r in rows),
        "command_pool": {s: pool_size[s] for s in speakers},
        "pairs_per_label": n_pairs,
        "per_speaker": {s: len(v) for s, v in rows_by_speaker.items()},
        "shortages": shortages[:20],
        "n_shortages": len(shortages),
        "native_rows": len(native),
        "seed": args.seed,
    }
    (AUDIO_OUT / "build_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
