#!/usr/bin/env python3
"""Boundary-aligned nested S/M/L/XL windows for long_form_asr / diar / attribution.

Writes crops + indexes under /mnt/afs/eval_data/windowed_ondevice by default.
Does not depend on the retired 30–60s long_form builders or smoke_diar scripts.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"
SR = 16000
EPS = 0.05
SILENCE_GAP = 0.3
DEFAULT_OUT = Path("/mnt/afs/eval_data/windowed_ondevice")

BUCKETS: dict[str, tuple[float, float]] = {
    "S": (25.0, 45.0),
    "M": (50.0, 90.0),
    "L": (100.0, 150.0),
    "XL": (240.0, 330.0),
}
BUCKET_ORDER = ("S", "M", "L", "XL")

AISHELL4 = Path("/mnt/afs/eval_data/03_multi_speaker/AISHELL-4")
ALIMEETING = Path("/mnt/afs/eval_data/03_multi_speaker/AliMeeting")
AMI = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/AMI")
CHIME6 = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME-6")
CHIME_RAW = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME")

SKIP_TEXT = {
    "",
    "<%>",
    "<sil>",
    "<noise>",
    "<NOISE>",
    "<knock>",
    "<laughter>",
    "<DEAF>",
    "<$>",
    "<#>",
    "<->",
    "<_>",
    "*",
    "**",
}
_ANGLE_TOKEN_RE = re.compile(r"<[^>\s]+>")


def clean_transcript(text: str) -> str:
    """Drop angle-bracket meeting tags (<sil>, <% >, …) from gold text."""
    cleaned = _ANGLE_TOKEN_RE.sub(" ", str(text or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


@dataclass
class Seg:
    start: float
    end: float
    speaker: str
    text: str = ""


@dataclass
class Session:
    dataset: str
    meeting_id: str
    src_audio: Path | None
    segs: list[Seg]
    # AMI: utterance clips for overlay; CHiME: (start,end,spk,unit)
    ami_utts: list[tuple[float, float, str, Path, str]] = field(default_factory=list)
    chime_utts: list[tuple[float, float, str, int]] = field(default_factory=list)
    mode: str = "crop"  # crop | overlay | chime


@dataclass
class BucketWin:
    bucket: str
    t0: float
    t1: float

    @property
    def duration(self) -> float:
        return self.t1 - self.t0


@dataclass
class Family:
    family_id: str
    dataset: str
    meeting_id: str
    t0: float
    windows: dict[str, BucketWin]
    src_audio: Path | None
    mode: str


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def parse_textgrid_spk_text(path: Path) -> list[Seg]:
    """Speaker-named IntervalTiers → Seg(start,end,speaker,text)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    items = re.split(r"\n\s*item \[(\d+)\]:", text)
    segs: list[Seg] = []
    for i in range(1, len(items), 2):
        body = items[i + 1]
        name_m = re.search(r'name = "([^"]+)"', body)
        if not name_m:
            continue
        spk = name_m.group(1)
        if spk in {"内容层", "角色层"}:
            continue
        for m in re.finditer(
            r"intervals \[(\d+)\]:\s*xmin = ([^\n]+)\s*xmax = ([^\n]+)\s*text = \"([^\"]*)\"",
            body,
        ):
            xmin, xmax = float(m.group(2)), float(m.group(3))
            lab = clean_transcript(m.group(4))
            if xmax <= xmin or not lab or lab in SKIP_TEXT:
                continue
            segs.append(Seg(xmin, xmax, spk, lab))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def parse_rttm(path: Path) -> list[Seg]:
    segs: list[Seg] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        start, dur, spk = float(parts[3]), float(parts[4]), parts[7]
        if dur > 0:
            segs.append(Seg(start, start + dur, spk, ""))
    segs.sort(key=lambda s: (s.start, s.end))
    return segs


def boundaries(segs: list[Seg], tmax: float | None = None) -> list[float]:
    pts = {0.0}
    for s in segs:
        pts.add(round(s.start, 6))
        pts.add(round(s.end, 6))
    # silence midpoints for gaps >= SILENCE_GAP
    ordered = sorted(segs, key=lambda s: s.start)
    for a, b in zip(ordered, ordered[1:]):
        gap = b.start - a.end
        if gap >= SILENCE_GAP:
            pts.add(round(a.end + gap / 2.0, 6))
    if tmax is not None:
        pts.add(round(tmax, 6))
    return sorted(pts)


def segs_in_window(segs: list[Seg], t0: float, t1: float) -> list[Seg]:
    out: list[Seg] = []
    for s in segs:
        if s.end <= t0 + EPS or s.start >= t1 - EPS:
            continue
        out.append(s)
    return out


def refs_complete(segs: list[Seg], t0: float, t1: float) -> bool:
    for s in segs:
        intersects = s.end > t0 + EPS and s.start < t1 - EPS
        if not intersects:
            continue
        if s.start < t0 - EPS or s.end > t1 + EPS:
            return False
    return True


def window_stats(segs: list[Seg], t0: float, t1: float) -> dict:
    inside = segs_in_window(segs, t0, t1)
    dur = max(1e-6, t1 - t0)
    speech = 0.0
    overlap = 0.0
    # sweep for speech / overlap
    events: list[tuple[float, int]] = []
    for s in inside:
        events.append((max(s.start, t0), 1))
        events.append((min(s.end, t1), -1))
    events.sort()
    active = 0
    prev = t0
    for t, delta in events:
        if t > prev and active > 0:
            speech += t - prev
            if active > 1:
                overlap += t - prev
        active += delta
        prev = t
    spks = {s.speaker for s in inside}
    return {
        "n_speakers": len(spks),
        "speech_ratio": speech / dur,
        "overlap_ratio": overlap / dur,
        "n_segments": len(inside),
    }


def find_t1(t0: float, lo: float, hi: float, bounds: list[float], segs: list[Seg]) -> float | None:
    cands = [b for b in bounds if t0 + lo - 1e-9 <= b <= t0 + hi + 1e-9 and b > t0 + 1e-6]
    for t1 in sorted(cands):
        if refs_complete(segs, t0, t1):
            return t1
    return None


def try_nested_family(
    segs: list[Seg], t0: float, bounds: list[float]
) -> dict[str, BucketWin] | None:
    windows: dict[str, BucketWin] = {}
    prev_t1 = t0
    for bucket in BUCKET_ORDER:
        lo, hi = BUCKETS[bucket]
        t1 = find_t1(t0, lo, hi, bounds, segs)
        if t1 is None or t1 <= prev_t1:
            break
        if not refs_complete(segs, t0, t1):
            break
        windows[bucket] = BucketWin(bucket, t0, t1)
        prev_t1 = t1
    if "S" not in windows:
        return None
    stats = window_stats(segs, t0, windows["S"].t1)
    if stats["n_speakers"] < 2 or stats["speech_ratio"] < 0.3:
        return None
    return windows


def select_families(
    session: Session,
    *,
    family_gap: float,
    max_families: int,
    seed: int,
) -> list[Family]:
    del seed  # reserved for future stratified sampling
    segs = session.segs
    if not segs:
        return []
    tmax = max(s.end for s in segs)
    bounds = boundaries(segs, tmax)
    min_s = BUCKETS["S"][0]
    cands: list[tuple[float, dict[str, BucketWin]]] = []
    for t0 in bounds:
        if t0 + min_s > tmax + 1e-6:
            continue
        wins = try_nested_family(segs, t0, bounds)
        if wins:
            cands.append((t0, wins))
    if not cands:
        return []

    thirds = [0.0, tmax / 3.0, 2.0 * tmax / 3.0, tmax]
    picked: list[tuple[float, dict[str, BucketWin]]] = []

    def ok_gap(t0: float) -> bool:
        return all(abs(t0 - p[0]) >= family_gap for p in picked)

    for i in range(3):
        lo, hi = thirds[i], thirds[i + 1]
        pool = [(t0, w) for t0, w in cands if lo <= t0 < hi and ok_gap(t0)]
        if not pool:
            continue
        pool.sort(key=lambda x: (-len(x[1]), x[0]))
        picked.append(pool[0])
        if len(picked) >= max_families:
            break

    rest = sorted(((t0, w) for t0, w in cands if ok_gap(t0)), key=lambda x: x[0])
    if rest:
        step = max(1, len(rest) // max(1, max_families))
        for i in range(0, len(rest), step):
            t0, w = rest[i]
            if ok_gap(t0):
                picked.append((t0, w))
            if len(picked) >= max_families:
                break

    picked.sort(key=lambda x: x[0])
    picked = picked[:max_families]
    # slight shuffle of ranks already fixed; keep time order
    out: list[Family] = []
    for idx, (t0, wins) in enumerate(picked):
        fid = f"{session.dataset}_{session.meeting_id}_f{idx:02d}"
        out.append(
            Family(
                family_id=fid,
                dataset=session.dataset,
                meeting_id=session.meeting_id,
                t0=t0,
                windows=wins,
                src_audio=session.src_audio,
                mode=session.mode,
            )
        )
    return out


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
        str(SR),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 1000:
        raise RuntimeError(f"ffmpeg crop failed {src} [{start},{end}]\n{proc.stderr[-800:]}")


def ffmpeg_prefix(src: Path, dst: Path, duration: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        str(src),
        "-t",
        f"{duration:.6f}",
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-c:a",
        "pcm_s16le",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size < 1000:
        raise RuntimeError(f"ffmpeg prefix failed {src} dur={duration}\n{proc.stderr[-800:]}")


def write_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio.astype(np.float32), SR)


def overlay_mix(
    utts: list[tuple[float, float, str, Path, str]], t0: float, t1: float
) -> np.ndarray | None:
    n = int(round((t1 - t0) * SR))
    if n <= 0:
        return None
    mix = np.zeros(n, dtype=np.float32)
    placed = 0
    for start, end, _spk, path, _text in utts:
        if end <= t0 or start >= t1:
            continue
        try:
            audio, file_sr = sf.read(str(path), dtype="float32")
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != SR:
            duration = len(audio) / float(file_sr)
            new_len = max(1, int(round(duration * SR)))
            x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
            x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        rel_a = max(0.0, t0 - start)
        rel_b = min(end - start, t1 - start)
        if rel_b - rel_a < 0.05:
            continue
        i0 = int(round(rel_a * SR))
        i1 = int(round(rel_b * SR))
        chunk = audio[i0:i1]
        dest = int(round((start + rel_a - t0) * SR))
        j0 = max(0, dest)
        j1 = min(n, dest + len(chunk))
        if j1 <= j0:
            continue
        take = chunk[j0 - dest : j1 - dest]
        mix[j0:j1] += take
        placed += 1
    if placed < 1:
        return None
    return mix


def resolve_chime_array(session: str, unit: int) -> list[Path] | None:
    for sub in ("eval", "dev"):
        base = CHIME_RAW / f"CHiME6_{sub}" / "CHiME6" / "audio" / sub
        chs = [base / f"{session}_U{unit:02d}.CH{c}.wav" for c in range(1, 5)]
        if all(p.is_file() for p in chs):
            return chs
    return None


def crop_chime(session: str, unit: int, t0: float, t1: float) -> np.ndarray:
    chs = resolve_chime_array(session, unit)
    if not chs:
        raise FileNotFoundError(f"CHiME array missing: {session} U{unit:02d}")
    i0 = int(round(t0 * SR))
    n = int(round((t1 - t0) * SR))
    acc = None
    for path in chs:
        audio, file_sr = sf.read(str(path), start=i0, stop=i0 + n, dtype="float32")
        if file_sr != SR:
            raise RuntimeError(f"unexpected sr={file_sr} for {path}")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) < n:
            pad = np.zeros(n, dtype=np.float32)
            pad[: len(audio)] = audio
            audio = pad
        else:
            audio = audio[:n]
        acc = audio if acc is None else acc + audio
    assert acc is not None
    return (acc / 4.0).astype(np.float32)


def majority_chime_unit(
    utts: list[tuple[float, float, str, int]], t0: float, t1: float
) -> int | None:
    units = []
    for start, end, _spk, unit in utts:
        if end <= t0 or start >= t1:
            continue
        units.append(unit)
    if not units:
        return None
    return Counter(units).most_common(1)[0][0]


def relative_segments(segs: list[Seg], t0: float, t1: float) -> list[dict]:
    out = []
    for s in segs_in_window(segs, t0, t1):
        out.append(
            {
                "start": round(max(0.0, s.start - t0), 3),
                "end": round(min(t1 - t0, s.end - t0), 3),
                "speaker": s.speaker,
            }
        )
    return out


def relative_utterances(segs: list[Seg], t0: float, t1: float) -> list[dict]:
    out = []
    for i, s in enumerate(segs_in_window(segs, t0, t1)):
        text = clean_transcript(s.text)
        if not text:
            continue
        out.append(
            {
                "id": str(i),
                "speaker": s.speaker,
                "text": text,
                "start": round(max(0.0, s.start - t0), 3),
                "end": round(min(t1 - t0, s.end - t0), 3),
            }
        )
    return out


def join_asr_text(segs: list[Seg], t0: float, t1: float) -> str:
    parts = [
        clean_transcript(s.text)
        for s in segs_in_window(segs, t0, t1)
        if clean_transcript(s.text)
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Session loaders
# ---------------------------------------------------------------------------


def load_aishell_4() -> list[Session]:
    root = AISHELL4 / "test"
    wav_dir = root / "wav"
    tg_dir = root / "TextGrid"
    sessions: list[Session] = []
    for tg in sorted(tg_dir.glob("*.TextGrid")):
        wav = wav_dir / f"{tg.stem}.flac"
        if not wav.is_file():
            continue
        segs = parse_textgrid_spk_text(tg)
        if not segs:
            rttm = tg_dir / f"{tg.stem}.rttm"
            if rttm.is_file():
                segs = parse_rttm(rttm)
        if not segs:
            continue
        sessions.append(
            Session("aishell_4", tg.stem, wav, segs, mode="crop")
        )
    return sessions


def load_alimeeting() -> list[Session]:
    sessions: list[Session] = []
    roots = [
        ALIMEETING / "Test_Ali" / "Test_Ali_far",
        ALIMEETING / "Eval_Ali" / "Eval_Ali_far",
    ]
    for root in roots:
        audio_dir = root / "audio_dir"
        tg_dir = root / "textgrid_dir"
        if not tg_dir.is_dir():
            continue
        for tg in sorted(tg_dir.glob("*.TextGrid")):
            wavs = sorted(audio_dir.glob(f"{tg.stem}_MS*.wav"))
            if not wavs:
                continue
            segs = parse_textgrid_spk_text(tg)
            if not segs:
                continue
            mid = f"{root.name}:{tg.stem}"
            sessions.append(Session("alimeeting", mid, wavs[0], segs, mode="crop"))
    return sessions


def _parse_ami_utt(path: Path) -> tuple[str, float, float, str] | None:
    parts = path.stem.split("_")
    if len(parts) < 7:
        return None
    meeting, spk = parts[2], parts[4]
    try:
        start = int(parts[5]) / 100.0
        end = int(parts[6]) / 100.0
    except ValueError:
        return None
    if end <= start:
        return None
    return meeting, start, end, spk


def load_ami() -> list[Session]:
    text_by_rel: dict[str, str] = {}
    for split in ("test", "dev"):
        manifest = AMI / f"ami_sdm_{split}_asr.jsonl"
        if not manifest.is_file():
            continue
        for line in manifest.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                ap = row["messages"][0]["content"][0]["audio_path"]
                tx = row["messages"][1]["content"][0]["text"] or ""
            except (KeyError, IndexError, TypeError):
                continue
            text_by_rel[ap] = tx

    by_meeting: dict[str, list[tuple[float, float, str, Path, str]]] = defaultdict(list)
    for split in ("test", "dev"):
        root = AMI / "sdm" / split
        if not root.is_dir():
            continue
        for wav in root.glob("*/*.wav"):
            parsed = _parse_ami_utt(wav)
            if not parsed:
                continue
            meeting, start, end, spk = parsed
            rel = str(wav.relative_to(AMI))
            text = clean_transcript(text_by_rel.get(rel, ""))
            by_meeting[meeting].append((start, end, spk, wav, text))

    sessions: list[Session] = []
    for meeting, utts in sorted(by_meeting.items()):
        utts = sorted(utts, key=lambda x: x[0])
        segs = [Seg(s, e, spk, text) for s, e, spk, _p, text in utts]
        sessions.append(
            Session(
                "ami",
                meeting,
                None,
                segs,
                ami_utts=utts,
                mode="overlay",
            )
        )
    return sessions


def _parse_chime_utt(path: Path) -> tuple[str, float, float, str, int] | None:
    parts = path.stem.split("_")
    try:
        idx = next(i for i, p in enumerate(parts) if re.fullmatch(r"S\d+", p))
        session = parts[idx]
        unit = int(parts[idx + 1][1:])
        spk = parts[idx + 2]
        start = int(parts[idx + 3]) / float(SR)
        end = int(parts[idx + 4]) / float(SR)
    except (StopIteration, ValueError, IndexError):
        return None
    if end <= start:
        return None
    return session, start, end, spk, unit


def load_chime_6() -> list[Session]:
    text_by_rel: dict[str, str] = {}
    for name in ("chime6_ref_array_test_asr.jsonl", "chime6_ref_array_dev_asr.jsonl"):
        manifest = CHIME6 / name
        if not manifest.is_file():
            continue
        for line in manifest.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                ap = row["messages"][0]["content"][0]["audio_path"]
                tx = row["messages"][1]["content"][0]["text"] or ""
            except (KeyError, IndexError, TypeError):
                continue
            text_by_rel[ap] = tx

    by_meeting: dict[str, list[tuple[float, float, str, int, str]]] = defaultdict(list)
    for split in ("test", "dev"):
        root = CHIME6 / "ref_array" / split
        if not root.is_dir():
            continue
        for wav in root.glob("*/*.wav"):
            parsed = _parse_chime_utt(wav)
            if not parsed:
                continue
            meeting, start, end, spk, unit = parsed
            rel = str(wav.relative_to(CHIME6))
            text = clean_transcript(text_by_rel.get(rel, ""))
            by_meeting[meeting].append((start, end, spk, unit, text))

    sessions: list[Session] = []
    for meeting, utts in sorted(by_meeting.items()):
        utts = sorted(utts, key=lambda x: x[0])
        segs = [Seg(s, e, spk, text) for s, e, spk, _u, text in utts]
        chime_utts = [(s, e, spk, u) for s, e, spk, u, _t in utts]
        sessions.append(
            Session(
                "chime_6",
                meeting,
                None,
                segs,
                chime_utts=chime_utts,
                mode="chime",
            )
        )
    return sessions


LOADERS = {
    "aishell_4": load_aishell_4,
    "alimeeting": load_alimeeting,
    "ami": load_ami,
    "chime_6": load_chime_6,
}


def materialize_family_audio(
    family: Family,
    session: Session,
    crop_dir: Path,
) -> dict[str, Path]:
    """Crop XL then prefix S/M/L. Returns bucket → wav path."""
    paths: dict[str, Path] = {}
    if "XL" in family.windows:
        top = family.windows["XL"]
    else:
        top = family.windows[max(family.windows, key=lambda b: BUCKET_ORDER.index(b))]

    top_path = crop_dir / f"{family.family_id}_{top.bucket}.wav"
    if family.mode == "crop":
        assert family.src_audio is not None
        if not top_path.is_file() or top_path.stat().st_size < 1000:
            ffmpeg_crop(family.src_audio, top_path, top.t0, top.t1)
    elif family.mode == "overlay":
        mix = overlay_mix(session.ami_utts, top.t0, top.t1)
        if mix is None:
            raise RuntimeError(f"overlay failed {family.family_id}")
        write_wav(top_path, mix)
    elif family.mode == "chime":
        unit = majority_chime_unit(session.chime_utts, top.t0, top.t1)
        if unit is None:
            raise RuntimeError(f"no chime unit {family.family_id}")
        audio = crop_chime(session.meeting_id, unit, top.t0, top.t1)
        write_wav(top_path, audio)
    else:
        raise ValueError(family.mode)
    paths[top.bucket] = top_path

    for bucket, win in family.windows.items():
        if bucket == top.bucket:
            continue
        dst = crop_dir / f"{family.family_id}_{bucket}.wav"
        # prefix duration relative to t0
        dur = win.t1 - win.t0
        if not dst.is_file() or dst.stat().st_size < 1000:
            ffmpeg_prefix(top_path, dst, dur)
        paths[bucket] = dst
    return paths


def build_index_rows(
    family: Family,
    session: Session,
    wav_paths: dict[str, Path],
) -> tuple[list[dict], list[dict], list[dict], dict]:
    asr_rows: list[dict] = []
    diar_rows: list[dict] = []
    attr_rows: list[dict] = []
    fam_meta = {
        "crop_family_id": family.family_id,
        "dataset": family.dataset,
        "meeting_id": family.meeting_id,
        "t0": family.t0,
        "buckets": {},
        "mode": family.mode,
        "src_audio": str(family.src_audio) if family.src_audio else None,
    }
    for bucket, win in family.windows.items():
        stats = window_stats(session.segs, win.t0, win.t1)
        wav = wav_paths[bucket]
        base = {
            "sample_id": f"{family.family_id}_{bucket}",
            "crop_family_id": family.family_id,
            "dataset": family.dataset,
            "meeting_id": family.meeting_id,
            "bucket": bucket,
            "protocol": f"windowed_ondevice_{bucket}",
            "WavPath": str(wav),
            "src_audio": str(family.src_audio) if family.src_audio else None,
            "start_time": win.t0,
            "end_time": win.t1,
            "window_start": win.t0,
            "audio_duration_seconds": round(win.duration, 3),
            "n_speakers": stats["n_speakers"],
            "speech_ratio": round(stats["speech_ratio"], 4),
            "overlap_ratio": round(stats["overlap_ratio"], 4),
        }
        fam_meta["buckets"][bucket] = {
            "t0": win.t0,
            "t1": win.t1,
            "duration": round(win.duration, 3),
            **stats,
        }
        text_asr = join_asr_text(session.segs, win.t0, win.t1)
        if text_asr:
            asr_rows.append(
                {
                    **base,
                    "capability": "long_form_asr",
                    "text": text_asr,
                    "n_utterances": stats["n_segments"],
                }
            )
        diar_segs = relative_segments(session.segs, win.t0, win.t1)
        if stats["n_speakers"] >= 2 and diar_segs:
            diar_rows.append(
                {
                    **base,
                    "capability": "diarization",
                    "text": {"segments": diar_segs},
                    "n_segments": len(diar_segs),
                }
            )
        utts = relative_utterances(session.segs, win.t0, win.t1)
        if stats["n_speakers"] >= 2 and utts:
            attr_rows.append(
                {
                    **base,
                    "capability": "speaker_attribution",
                    "text": {"utterances": utts},
                    "n_utterances": len(utts),
                }
            )
    return asr_rows, diar_rows, attr_rows, fam_meta


def build_dataset(
    dataset: str,
    out_root: Path,
    *,
    family_gap: float = 150.0,
    max_families: int = 6,
    seed: int = 42,
) -> dict:
    loader = LOADERS[dataset]
    sessions = loader()
    crop_dir = out_root / "crops" / dataset
    crop_dir.mkdir(parents=True, exist_ok=True)

    # timelines
    tl_rows = []
    for sess in sessions:
        for s in sess.segs:
            tl_rows.append(
                {
                    "dataset": dataset,
                    "meeting_id": sess.meeting_id,
                    "src_audio": str(sess.src_audio) if sess.src_audio else None,
                    "start_time": s.start,
                    "end_time": s.end,
                    "speaker": s.speaker,
                    "text": s.text,
                }
            )
    write_jsonl(out_root / "timelines" / f"{dataset}.jsonl", tl_rows)

    asr_by_b: dict[str, list[dict]] = defaultdict(list)
    diar_by_b: dict[str, list[dict]] = defaultdict(list)
    attr_by_b: dict[str, list[dict]] = defaultdict(list)
    families_meta: list[dict] = []
    n_families = 0
    errors: list[str] = []

    for sess in sessions:
        fams = select_families(
            sess, family_gap=family_gap, max_families=max_families, seed=seed
        )
        for fam in fams:
            try:
                wavs = materialize_family_audio(fam, sess, crop_dir)
                asr, diar, attr, meta = build_index_rows(fam, sess, wavs)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{fam.family_id}: {exc}")
                continue
            n_families += 1
            families_meta.append(meta)
            for row in asr:
                asr_by_b[row["bucket"]].append(row)
            for row in diar:
                diar_by_b[row["bucket"]].append(row)
            for row in attr:
                attr_by_b[row["bucket"]].append(row)

    write_jsonl(out_root / "families" / f"{dataset}_families.jsonl", families_meta)
    counts = {"long_form_asr": {}, "diarization": {}, "speaker_attribution": {}}
    for b in BUCKET_ORDER:
        counts["long_form_asr"][b] = write_jsonl(
            out_root / "indexes" / "long_form_asr" / f"{dataset}_{b}.jsonl",
            asr_by_b.get(b, []),
        )
        counts["diarization"][b] = write_jsonl(
            out_root / "indexes" / "diarization" / f"{dataset}_{b}.jsonl",
            diar_by_b.get(b, []),
        )
        counts["speaker_attribution"][b] = write_jsonl(
            out_root / "indexes" / "speaker_attribution" / f"{dataset}_{b}.jsonl",
            attr_by_b.get(b, []),
        )

    return {
        "dataset": dataset,
        "sessions": len(sessions),
        "families": n_families,
        "counts": counts,
        "errors": errors[:20],
        "n_errors": len(errors),
    }


def build_all(
    datasets: list[str] | None = None,
    out_root: Path = DEFAULT_OUT,
    **kwargs,
) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    selected = datasets or list(LOADERS.keys())
    summary = out_root / "summaries" / "build_report.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    report: dict = {"out_root": str(out_root), "datasets": {}}
    if summary.is_file():
        try:
            prev = json.loads(summary.read_text(encoding="utf-8"))
            if isinstance(prev.get("datasets"), dict):
                report["datasets"].update(prev["datasets"])
        except Exception:
            pass
    for name in selected:
        print(f"[windowed_ondevice] building {name} ...", flush=True)
        report["datasets"][name] = build_dataset(name, out_root, **kwargs)
        print(json.dumps(report["datasets"][name], ensure_ascii=False, indent=2), flush=True)
        summary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
