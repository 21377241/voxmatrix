#!/usr/bin/env python3
"""Build diarization smoke manifests: 10 short multi-speaker windows per benchmark.

Cut/overlay ONLY after sampling to avoid scanning/cutting thousands of windows.

Datasets
--------
- aishell_4 / alimeeting / misp: original long audio → ffmpeg 30s cut
- ami: SDM utt clips → overlay reconstruct 30s mix (no continuous mix on disk)
- chime_6: continuous ref-array session → 4ch mean crop 30s (utt times for gold)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
CLIPS = ROOT / "clips"
N = 10
WIN = 30.0
STEP = 20.0
SR = 16000
FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"

AISHELL4 = Path("/mnt/afs/eval_data/03_multi_speaker/AISHELL-4")
ALIMEETING = Path("/mnt/afs/eval_data/03_multi_speaker/AliMeeting")
AMI = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/AMI")
CHIME6 = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME-6")
CHIME_RAW = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME")
MISP = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/MISP")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _existing_ids(name: str) -> list[str] | None:
    manifest = ROOT / name / "manifest.jsonl"
    if not manifest.is_file():
        return None
    ids = [json.loads(line)["sample_id"] for line in manifest.open(encoding="utf-8")]
    return ids or None


def sample_stride(rows: list, n: int = N) -> list:
    if len(rows) <= n:
        return rows
    step = max(1, len(rows) // n)
    return [rows[i * step] for i in range(n)][:n]


def parse_rttm(path: Path) -> list[tuple[float, float, str]]:
    segs: list[tuple[float, float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("SPEAKER"):
            continue
        parts = line.split()
        start = float(parts[3])
        dur = float(parts[4])
        spk = parts[7]
        if dur > 0:
            segs.append((start, start + dur, spk))
    return segs


def parse_textgrid_speaker_tiers(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    items = re.split(r"\n\s*item \[(\d+)\]:", text)
    segs: list[tuple[float, float, str]] = []
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
            xmin, xmax, lab = float(m.group(2)), float(m.group(3)), m.group(4).strip()
            if xmax > xmin and lab:
                segs.append((xmin, xmax, spk))
    return segs


def parse_misp_role_tier(path: Path) -> list[tuple[float, float, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    items = re.split(r"\n\s*item \[(\d+)\]:", text)
    segs: list[tuple[float, float, str]] = []
    skip = {"", "<NOISE>", "<knock>", "<laughter>", "<DEAF>", "*", "**"}
    for i in range(1, len(items), 2):
        body = items[i + 1]
        name_m = re.search(r'name = "([^"]+)"', body)
        if not name_m or name_m.group(1) != "角色层":
            continue
        for m in re.finditer(
            r"intervals \[(\d+)\]:\s*xmin = ([^\n]+)\s*xmax = ([^\n]+)\s*text = \"([^\"]*)\"",
            body,
        ):
            xmin, xmax, lab = float(m.group(2)), float(m.group(3)), m.group(4).strip()
            if xmax <= xmin or lab in skip:
                continue
            lab = lab.replace("<", "").replace(">", "").strip() or "spk"
            segs.append((xmin, xmax, lab))
    return segs


def find_windows(
    segs: list[tuple[float, float, str]],
    win: float = WIN,
    step: float = STEP,
    min_spk: int = 2,
    min_speech: float = 4.0,
) -> list[float]:
    if not segs:
        return []
    tmax = max(e for _, e, _ in segs)
    out: list[float] = []
    t = 0.0
    while t + win <= tmax + 1e-6:
        cov: dict[str, float] = defaultdict(float)
        for s, e, spk in segs:
            a, b = max(s, t), min(e, t + win)
            if b > a:
                cov[spk] += b - a
        if len(cov) >= min_spk and sum(cov.values()) >= min_speech:
            out.append(t)
        t += step
    return out


def clip_segments(
    segs: list[tuple[float, float, str]], win_start: float, win_end: float
) -> list[dict]:
    out: list[dict] = []
    for s, e, spk in segs:
        a, b = max(s, win_start), min(e, win_end)
        if b - a < 0.05:
            continue
        out.append(
            {
                "start": round(a - win_start, 3),
                "end": round(b - win_start, 3),
                "speaker": spk,
            }
        )
    return out


def ffmpeg_cut(src: Path, dst: Path, start: float, dur: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 1000:
        return
    cmd = [
        FFMPEG,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{dur:.3f}",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(SR),
        "-vn",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_wav(path: Path, audio: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio.astype(np.float32), sr)


def _select(cands: list[dict], name: str) -> list[dict]:
    existing = _existing_ids(name)
    if existing:
        by_id = {c["sample_id"]: c for c in cands}
        if all(sid in by_id for sid in existing):
            return [by_id[sid] for sid in existing]
    return sample_stride(cands, N)


def _materialize_cut(row: dict, segs: list[tuple[float, float, str]]) -> dict:
    ws = float(row["window_start"])
    rel = clip_segments(segs, ws, ws + WIN)
    clip = Path(row["WavPath"])
    ffmpeg_cut(Path(row["source_wav"]), clip, ws, WIN)
    row = dict(row)
    # ref_col=text only — do NOT also set `reference` (EvalTask._eval duplicate kwarg).
    row["text"] = {"segments": rel}
    row["audio_duration_seconds"] = float(WIN)
    row["n_speakers"] = len({s["speaker"] for s in rel})
    row["n_segments"] = len(rel)
    return row


# ---------------------------------------------------------------------------
def build_aishell_4() -> list[dict]:
    print("  scan aishell_4 rttm...", flush=True)
    wav_dir = AISHELL4 / "test" / "wav"
    rttm_dir = AISHELL4 / "test" / "TextGrid"
    cands: list[dict] = []
    segs_by_stem: dict[str, list] = {}
    for rttm in sorted(rttm_dir.glob("*.rttm")):
        stem = rttm.stem
        wav = wav_dir / f"{stem}.flac"
        if not wav.is_file():
            continue
        segs = parse_rttm(rttm)
        segs_by_stem[stem] = segs
        for ws in find_windows(segs):
            sid = f"aishell4_diar_{stem}_w{int(ws):04d}"
            cands.append(
                {
                    "sample_id": sid,
                    "dataset": "aishell_4",
                    "subset": "test",
                    "protocol": "rttm_window",
                    "window_start": ws,
                    "source_wav": str(wav),
                    "WavPath": str(CLIPS / "aishell_4" / f"{sid}.wav"),
                    "_stem": stem,
                }
            )
    picked = _select(cands, "aishell_4")
    print(f"  cut {len(picked)}/{len(cands)} aishell_4 clips...", flush=True)
    out = []
    for row in picked:
        out.append(_materialize_cut(row, segs_by_stem[row.pop("_stem")]))
    if len(out) < N:
        raise RuntimeError(f"aishell_4: only {len(out)}")
    return out


def build_alimeeting() -> list[dict]:
    print("  scan alimeeting textgrid...", flush=True)
    cands: list[dict] = []
    segs_by_key: dict[str, list] = {}
    for split in ("Eval_Ali_far", "Test_Ali_far"):
        root = (
            ALIMEETING
            / ("Eval_Ali" if split.startswith("Eval") else "Test_Ali")
            / split
        )
        audio_dir = root / "audio_dir"
        tg_dir = root / "textgrid_dir"
        for tg in sorted(tg_dir.glob("*.TextGrid")):
            wavs = sorted(audio_dir.glob(f"{tg.stem}_MS*.wav"))
            if not wavs:
                continue
            wav = wavs[0]
            segs = parse_textgrid_speaker_tiers(tg)
            key = f"{split}:{tg.stem}"
            segs_by_key[key] = segs
            for ws in find_windows(segs):
                sid = f"alimeeting_diar_{tg.stem}_w{int(ws):04d}"
                cands.append(
                    {
                        "sample_id": sid,
                        "dataset": "alimeeting",
                        "subset": split,
                        "protocol": "textgrid_window",
                        "window_start": ws,
                        "source_wav": str(wav),
                        "WavPath": str(CLIPS / "alimeeting" / f"{sid}.wav"),
                        "_key": key,
                    }
                )
    picked = _select(cands, "alimeeting")
    print(f"  cut {len(picked)}/{len(cands)} alimeeting clips...", flush=True)
    out = []
    for row in picked:
        out.append(_materialize_cut(row, segs_by_key[row.pop("_key")]))
    if len(out) < N:
        raise RuntimeError(f"alimeeting: only {len(out)}")
    return out


def _parse_ami_utt(path: Path) -> tuple[str, float, float, str] | None:
    parts = path.stem.split("_")
    if len(parts) < 7:
        return None
    meeting = parts[2]
    spk = parts[4]
    try:
        start = int(parts[5]) / 100.0
        end = int(parts[6]) / 100.0
    except ValueError:
        return None
    if end <= start:
        return None
    return meeting, start, end, spk


def _parse_chime_utt(path: Path) -> tuple[str, float, float, str, int] | None:
    """Return (session, start_s, end_s, speaker, array_unit)."""
    parts = path.stem.split("_")
    try:
        idx = next(i for i, p in enumerate(parts) if re.fullmatch(r"S\d+", p))
        session = parts[idx]
        unit = int(parts[idx + 1][1:])  # U02 → 2
        spk = parts[idx + 2]
        start = int(parts[idx + 3]) / float(SR)
        end = int(parts[idx + 4]) / float(SR)
    except (StopIteration, ValueError, IndexError):
        return None
    if end <= start:
        return None
    return session, start, end, spk, unit


def _resolve_chime_array(session: str, unit: int) -> list[Path] | None:
    for split, sub in (("eval", "eval"), ("dev", "dev")):
        base = CHIME_RAW / f"CHiME6_{split}" / "CHiME6" / "audio" / sub
        chs = [base / f"{session}_U{unit:02d}.CH{c}.wav" for c in range(1, 5)]
        if all(p.is_file() for p in chs):
            return chs
    return None


def _crop_chime_ref_array(session: str, unit: int, win_start: float) -> np.ndarray:
    """Equal-weight mean of the 4 ref-array channels over [win_start, win_start+WIN]."""
    chs = _resolve_chime_array(session, unit)
    if not chs:
        raise FileNotFoundError(f"CHiME continuous array missing: {session} U{unit:02d}")
    i0 = int(round(win_start * SR))
    n = int(round(WIN * SR))
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


def _chime_window_segs(
    utts: list[tuple[float, float, str, int]], win_start: float, win_end: float
) -> tuple[list[dict], int] | None:
    """Gold segments from utt times; majority ref-array unit in the window."""
    segs: list[dict] = []
    units: list[int] = []
    for start, end, spk, unit in utts:
        if end <= win_start or start >= win_end:
            continue
        a = max(start, win_start)
        b = min(end, win_end)
        if b - a < 0.05:
            continue
        segs.append(
            {
                "start": round(a - win_start, 3),
                "end": round(b - win_start, 3),
                "speaker": spk,
            }
        )
        units.append(unit)
    if len(segs) < 2 or len({s["speaker"] for s in segs}) < 2 or not units:
        return None
    unit = Counter(units).most_common(1)[0][0]
    return segs, unit


def _overlay_window(
    utts: list[tuple[float, float, str, Path]], win_start: float, win_end: float
) -> tuple[np.ndarray, list[dict]] | None:
    n = int(round((win_end - win_start) * SR))
    mix = np.zeros(n, dtype=np.float32)
    segs: list[dict] = []
    placed = 0
    for start, end, spk, path in utts:
        if end <= win_start or start >= win_end:
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
        abs_start = start
        rel_a = max(0.0, win_start - abs_start)
        rel_b = min(end - abs_start, win_end - abs_start)
        if rel_b - rel_a < 0.05:
            continue
        i0 = int(round(rel_a * SR))
        i1 = int(round(rel_b * SR))
        chunk = audio[i0:i1]
        dest = int(round((abs_start + rel_a - win_start) * SR))
        j1 = min(n, dest + len(chunk))
        j0 = max(0, dest)
        if j1 <= j0:
            continue
        take = chunk[j0 - dest : j1 - dest]
        mix[j0:j1] += take
        segs.append(
            {
                "start": round(abs_start + rel_a - win_start, 3),
                "end": round(abs_start + rel_b - win_start, 3),
                "speaker": spk,
            }
        )
        placed += 1
    if placed < 2 or len({s["speaker"] for s in segs}) < 2:
        return None
    return mix, segs


def _utt_window_candidates(
    name: str,
    sessions: dict[str, list[tuple[float, float, str, Path]]],
    protocol: str,
) -> list[dict]:
    cands: list[dict] = []
    for meeting, utts in sorted(sessions.items()):
        if len(utts) < 4:
            continue
        utts_sorted = sorted(utts, key=lambda x: x[0])
        # quick window search using segment times only (no audio I/O)
        segs = [(s, e, spk) for s, e, spk, _ in utts_sorted]
        for ws in find_windows(segs, min_speech=2.0):
            sid = f"{name}_diar_{meeting}_w{int(ws):04d}"
            cands.append(
                {
                    "sample_id": sid,
                    "dataset": name,
                    "subset": "reconstructed_window",
                    "protocol": protocol,
                    "window_start": ws,
                    "WavPath": str(CLIPS / name / f"{sid}.wav"),
                    "_meeting": meeting,
                }
            )
    return cands


def _materialize_overlay(
    row: dict, sessions: dict[str, list[tuple[float, float, str, Path]]]
) -> dict | None:
    meeting = row.pop("_meeting")
    ws = float(row["window_start"])
    built = _overlay_window(sessions[meeting], ws, ws + WIN)
    if built is None:
        return None
    mix, segs = built
    clip = Path(row["WavPath"])
    if not clip.is_file() or clip.stat().st_size < 1000:
        write_wav(clip, mix)
    row = dict(row)
    row["text"] = {"segments": segs}
    row["audio_duration_seconds"] = float(WIN)
    row["n_speakers"] = len({s["speaker"] for s in segs})
    row["n_segments"] = len(segs)
    return row


def _materialize_chime_crop(
    row: dict, sessions: dict[str, list[tuple[float, float, str, int]]], *, force: bool
) -> dict | None:
    meeting = row.pop("_meeting")
    ws = float(row["window_start"])
    built = _chime_window_segs(sessions[meeting], ws, ws + WIN)
    if built is None:
        return None
    segs, unit = built
    mix = _crop_chime_ref_array(meeting, unit, ws)
    clip = Path(row["WavPath"])
    if force or not clip.is_file() or clip.stat().st_size < 1000:
        write_wav(clip, mix)
    chs = _resolve_chime_array(meeting, unit)
    row = dict(row)
    row["subset"] = "session_window"
    row["protocol"] = "ref_array_window"
    row["text"] = {"segments": segs}
    row["audio_duration_seconds"] = float(WIN)
    row["n_speakers"] = len({s["speaker"] for s in segs})
    row["n_segments"] = len(segs)
    row["array_unit"] = unit
    if chs:
        row["source_wav"] = str(chs[0])
    return row


def build_ami() -> list[dict]:
    print("  index ami sdm/test...", flush=True)
    sessions: dict[str, list[tuple[float, float, str, Path]]] = defaultdict(list)
    for wav in (AMI / "sdm" / "test").glob("*/*.wav"):
        parsed = _parse_ami_utt(wav)
        if not parsed:
            continue
        meeting, start, end, spk = parsed
        sessions[meeting].append((start, end, spk, wav))
    cands = _utt_window_candidates("ami", sessions, "utt_overlay_sdm")
    picked = _select(cands, "ami")
    print(f"  overlay {len(picked)}/{len(cands)} ami windows...", flush=True)
    out = []
    for row in picked:
        item = _materialize_overlay(row, sessions)
        if item:
            out.append(item)
    # top up if some overlays failed
    if len(out) < N:
        for row in cands:
            if row["sample_id"] in {r["sample_id"] for r in out}:
                continue
            item = _materialize_overlay(dict(row), sessions)
            if item:
                out.append(item)
            if len(out) >= N:
                break
    if len(out) < N:
        raise RuntimeError(f"ami: only {len(out)}")
    return out[:N]


def build_chime_6(*, force_recut: bool = True) -> list[dict]:
    print("  index chime_6 ref_array/test...", flush=True)
    sessions: dict[str, list[tuple[float, float, str, int]]] = defaultdict(list)
    for wav in (CHIME6 / "ref_array" / "test").glob("*/*.wav"):
        parsed = _parse_chime_utt(wav)
        if not parsed:
            continue
        meeting, start, end, spk, unit = parsed
        sessions[meeting].append((start, end, spk, unit))
    # candidate search only needs (start,end,spk); reuse helper via dummy path
    sessions_for_cands: dict[str, list[tuple[float, float, str, Path]]] = {
        m: [(s, e, spk, Path(".")) for s, e, spk, _u in utts]
        for m, utts in sessions.items()
    }
    cands = _utt_window_candidates("chime_6", sessions_for_cands, "ref_array_window")
    for c in cands:
        c["subset"] = "session_window"
    picked = _select(cands, "chime_6")
    print(f"  crop {len(picked)}/{len(cands)} chime_6 windows from continuous...", flush=True)
    out = []
    for row in picked:
        item = _materialize_chime_crop(dict(row), sessions, force=force_recut)
        if item:
            out.append(item)
    if len(out) < N:
        for row in cands:
            if row["sample_id"] in {r["sample_id"] for r in out}:
                continue
            item = _materialize_chime_crop(dict(row), sessions, force=force_recut)
            if item:
                out.append(item)
            if len(out) >= N:
                break
    if len(out) < N:
        raise RuntimeError(f"chime_6: only {len(out)}")
    return out[:N]


def build_misp() -> list[dict]:
    print("  scan misp textgrid...", flush=True)
    cands: list[dict] = []
    segs_by_stem: dict[str, list] = {}
    for tg in sorted((MISP / "eval-F8N").glob("*/*/*.TextGrid")):
        wav = tg.with_suffix(".wav")
        if not wav.is_file():
            continue
        segs = parse_misp_role_tier(tg)
        segs_by_stem[tg.stem] = segs
        wins = find_windows(segs, min_spk=2, min_speech=3.0)
        if not wins:
            wins = find_windows(segs, min_spk=1, min_speech=5.0)[:8]
        for ws in wins:
            sid = f"misp_diar_{tg.stem}_w{int(ws):04d}"
            cands.append(
                {
                    "sample_id": sid,
                    "dataset": "misp",
                    "subset": "eval_f8n",
                    "protocol": "role_tier_window",
                    "window_start": ws,
                    "source_wav": str(wav),
                    "WavPath": str(CLIPS / "misp" / f"{sid}.wav"),
                    "_stem": tg.stem,
                    "note": "gold from 角色层 (主说话人/OVERCLEAR), not named speakers",
                }
            )
    picked = _select(cands, "misp")
    print(f"  cut {len(picked)}/{len(cands)} misp clips...", flush=True)
    out = []
    for row in picked:
        out.append(_materialize_cut(row, segs_by_stem[row.pop("_stem")]))
    if len(out) < N:
        raise RuntimeError(f"misp: only {len(out)}")
    return out


def main(only: list[str] | None = None) -> None:
    CLIPS.mkdir(parents=True, exist_ok=True)
    builders = {
        "aishell_4": build_aishell_4,
        "alimeeting": build_alimeeting,
        "ami": build_ami,
        "chime_6": build_chime_6,
        "misp": build_misp,
    }
    names = only or list(builders)
    for name in names:
        if name not in builders:
            raise SystemExit(f"unknown dataset: {name}; choose from {list(builders)}")
    summary_path = ROOT / "summary.json"
    if summary_path.is_file() and only:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.setdefault("datasets", [])
        by_name = {d["name"]: d for d in summary["datasets"]}
    else:
        summary = {
            "capability": "diarization",
            "window_seconds": WIN,
            "samples_per_benchmark": N,
            "datasets": [],
        }
        by_name = {}
    for name in names:
        fn = builders[name]
        print(f"[build] {name} ...", flush=True)
        rows = fn()
        manifest = ROOT / name / "manifest.jsonl"
        write_jsonl(manifest, rows)
        spk_hist = Counter(row["n_speakers"] for row in rows)
        by_name[name] = {
            "name": name,
            "manifest": str(manifest),
            "n": len(rows),
            "n_speakers_hist": dict(spk_hist),
            "protocol": rows[0]["protocol"],
        }
        print(f"[ok] {name}: {len(rows)} spk_hist={dict(spk_hist)}", flush=True)
    order = ["aishell_4", "alimeeting", "ami", "chime_6", "misp"]
    summary["datasets"] = [by_name[n] for n in order if n in by_name]
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main(sys.argv[1:] or None)
