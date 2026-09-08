#!/usr/bin/env python3
"""Build speaker_attribution smoke manifests (10 samples × 5 benchmarks).

Reuses smoke_diar_all 30s clips where possible; rebuilds ref as
{"utterances":[{"id","speaker","text"}]}. cn_celeb is SID (protocol mismatch)
and uses classification gold (speaker_id).
"""
from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIAR = Path("/mnt/afs/users/wangyl/VoxMatrix/smoke_diar_all")
CLIPS = ROOT / "clips"
N = 10
WIN = 30.0
SR = 16000
FFMPEG = "/mnt/afs/conda/envs/audioeval/bin/ffmpeg"

AISHELL4 = Path("/mnt/afs/eval_data/03_multi_speaker/AISHELL-4")
ALIMEETING = Path("/mnt/afs/eval_data/03_multi_speaker/AliMeeting")
AMI_ASR = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/AMI/ami_sdm_test_asr.jsonl")
CHIME6_ASR = Path(
    "/mnt/afs/oss_data/datasets/03_multi_speaker/CHiME-6/chime6_ref_array_test_asr.jsonl"
)
AISHELL5_ROOT = Path("/mnt/afs/oss_data/datasets/02_complex_acoustic/AISHELL-5")
AISHELL5_INDEX = AISHELL5_ROOT / "AISHELL-5_eval1_msswift.jsonl"
CN_CELEB_ROOT = Path("/mnt/afs/oss_data/datasets/03_multi_speaker/CN-Celeb")
CN_CELEB_JSON = CN_CELEB_ROOT / "audiobench.json"

SKIP_TEXT = {"", "<%>", "<sil>", "<NOISE>", "<knock>", "<laughter>", "<DEAF>", "*", "**"}


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


def ffmpeg_cut(src: Path, dst: Path, start: float, dur: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 1000:
        return
    cmd = [
        FFMPEG, "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
        "-i", str(src), "-ac", "1", "-ar", str(SR), "-vn", str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def parse_textgrid_utts(path: Path) -> list[tuple[float, float, str, str]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    items = re.split(r"\n\s*item \[(\d+)\]:", text)
    utts: list[tuple[float, float, str, str]] = []
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
            if xmax <= xmin or lab in SKIP_TEXT:
                continue
            clean = re.sub(r"<[^>]+>", "", lab).strip()
            if not clean:
                continue
            utts.append((xmin, xmax, spk, clean))
    return utts


def clip_utterances(
    utts: list[tuple[float, float, str, str]], win_start: float, win_end: float
) -> list[dict]:
    out: list[dict] = []
    for s, e, spk, txt in utts:
        a, b = max(s, win_start), min(e, win_end)
        if b - a < 0.05:
            continue
        # keep utterance if majority of speech is in window or text is short
        cover = (b - a) / max(1e-6, e - s)
        if cover < 0.5 and (e - s) > 1.0:
            # still include truncated text for long spans that overlap
            pass
        out.append(
            {
                "id": str(len(out)),
                "speaker": spk,
                "text": txt,
                "start": round(a - win_start, 3),
                "end": round(b - win_start, 3),
            }
        )
    out.sort(key=lambda x: (x["start"], x["end"]))
    for i, item in enumerate(out):
        item["id"] = str(i)
    return out


def _load_diar_manifest(name: str) -> list[dict]:
    path = DIAR / name / "manifest.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def build_aishell_4() -> list[dict]:
    rows = []
    for diar in _load_diar_manifest("aishell_4"):
        stem = Path(diar["source_wav"]).stem
        tg = AISHELL4 / "test" / "TextGrid" / f"{stem}.TextGrid"
        utts = parse_textgrid_utts(tg)
        ws = float(diar["window_start"])
        utterances = clip_utterances(utts, ws, ws + WIN)
        if len({u["speaker"] for u in utterances}) < 2 or not utterances:
            continue
        sid = diar["sample_id"].replace("_diar_", "_attr_")
        clip = Path(diar["WavPath"])
        assert clip.is_file(), clip
        rows.append(
            {
                "sample_id": sid,
                "dataset": "aishell_4",
                "subset": "test",
                "protocol": "textgrid_window",
                "language": "zh",
                "window_start": ws,
                "WavPath": str(clip),
                "text": {"utterances": utterances},
                "n_speakers": len({u["speaker"] for u in utterances}),
                "n_utterances": len(utterances),
            }
        )
    if len(rows) < N:
        raise RuntimeError(f"aishell_4 attr: only {len(rows)}")
    return rows[:N]


def _alimeeting_tg_for_wav(source_wav: str) -> Path:
    wav = Path(source_wav)
    # .../Eval_Ali_far/audio_dir/R8001_M8004_MS801.wav -> textgrid_dir/R8001_M8004.TextGrid
    stem = wav.stem
    meeting = re.sub(r"_MS\d+$", "", stem)
    tg = wav.parent.parent / "textgrid_dir" / f"{meeting}.TextGrid"
    if not tg.is_file():
        raise FileNotFoundError(tg)
    return tg


def build_alimeeting() -> list[dict]:
    rows = []
    for diar in _load_diar_manifest("alimeeting"):
        tg = _alimeeting_tg_for_wav(diar["source_wav"])
        utts = parse_textgrid_utts(tg)
        ws = float(diar["window_start"])
        utterances = clip_utterances(utts, ws, ws + WIN)
        if len({u["speaker"] for u in utterances}) < 2 or not utterances:
            continue
        sid = diar["sample_id"].replace("_diar_", "_attr_")
        clip = Path(diar["WavPath"])
        assert clip.is_file(), clip
        rows.append(
            {
                "sample_id": sid,
                "dataset": "alimeeting",
                "subset": diar.get("subset", "Eval_Ali_far"),
                "protocol": "textgrid_window",
                "language": "zh",
                "window_start": ws,
                "WavPath": str(clip),
                "text": {"utterances": utterances},
                "n_speakers": len({u["speaker"] for u in utterances}),
                "n_utterances": len(utterances),
            }
        )
    if len(rows) < N:
        raise RuntimeError(f"alimeeting attr: only {len(rows)}")
    return rows[:N]


def _parse_ami_id(sid: str) -> tuple[str, str, float, float] | None:
    # ami_sdm_test_EN2002a_sdm_FEO072_0035475_0035713
    m = re.search(
        r"ami_sdm_test_([A-Za-z0-9]+)_sdm_([A-Za-z0-9]+)_(\d+)_(\d+)$", sid
    )
    if not m:
        return None
    meeting, spk, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    return meeting.lower(), spk, a / 100.0, b / 100.0


def build_ami() -> list[dict]:
    by_meeting: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    with AMI_ASR.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            parsed = _parse_ami_id(obj["id"])
            if not parsed:
                continue
            meeting, spk, start, end = parsed
            txt = obj["messages"][1]["content"][0]["text"]
            if not txt:
                continue
            by_meeting[meeting].append((start, end, spk, txt))

    rows = []
    for diar in _load_diar_manifest("ami"):
        # ami_diar_en2002a_w0000
        m = re.match(r"ami_diar_([a-z0-9]+)_w(\d+)$", diar["sample_id"])
        if not m:
            continue
        meeting = m.group(1)
        ws = float(diar["window_start"])
        utts = by_meeting.get(meeting, [])
        utterances = clip_utterances(utts, ws, ws + WIN)
        if len({u["speaker"] for u in utterances}) < 2 or not utterances:
            continue
        sid = diar["sample_id"].replace("_diar_", "_attr_")
        clip = Path(diar["WavPath"])
        assert clip.is_file(), clip
        rows.append(
            {
                "sample_id": sid,
                "dataset": "ami",
                "subset": "sdm_overlay_window",
                "protocol": "utt_overlay_sdm",
                "language": "en",
                "window_start": ws,
                "WavPath": str(clip),
                "text": {"utterances": utterances},
                "n_speakers": len({u["speaker"] for u in utterances}),
                "n_utterances": len(utterances),
            }
        )
    if len(rows) < N:
        raise RuntimeError(f"ami attr: only {len(rows)}")
    return rows[:N]


def _parse_chime_asr_id(sid: str) -> tuple[str, str, float, float] | None:
    # chime6_ref_array_test_S01_U02_P03_000000000_000021920
    m = re.search(
        r"chime6_ref_array_test_(S\d+)_U\d+_(P\d+)_(\d+)_(\d+)$", sid
    )
    if not m:
        return None
    session, spk, a, b = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
    return session, spk, a / float(SR), b / float(SR)


def build_chime_6() -> list[dict]:
    """Reuse continuous-cropped diar clips; gold from ref_array ASR transcripts."""
    by_session: dict[str, list[tuple[float, float, str, str]]] = defaultdict(list)
    with CHIME6_ASR.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            parsed = _parse_chime_asr_id(obj["id"])
            if not parsed:
                continue
            session, spk, start, end = parsed
            txt = obj["messages"][1]["content"][0]["text"]
            if not txt or not str(txt).strip():
                continue
            by_session[session].append((start, end, spk, str(txt).strip()))

    rows = []
    for diar in _load_diar_manifest("chime_6"):
        m = re.match(r"chime_6_diar_(S\d+)_w(\d+)$", diar["sample_id"])
        if not m:
            continue
        session = m.group(1)
        ws = float(diar["window_start"])
        utterances = clip_utterances(by_session.get(session, []), ws, ws + WIN)
        if len({u["speaker"] for u in utterances}) < 2 or not utterances:
            continue
        sid = diar["sample_id"].replace("_diar_", "_attr_")
        clip = Path(diar["WavPath"])
        assert clip.is_file(), clip
        rows.append(
            {
                "sample_id": sid,
                "dataset": "chime_6",
                "subset": diar.get("subset", "session_window"),
                "protocol": diar.get("protocol", "ref_array_window"),
                "language": "en",
                "window_start": ws,
                "WavPath": str(clip),
                "text": {"utterances": utterances},
                "n_speakers": len({u["speaker"] for u in utterances}),
                "n_utterances": len(utterances),
                "audio_duration_seconds": float(WIN),
                "array_unit": diar.get("array_unit"),
            }
        )
    if len(rows) < N:
        raise RuntimeError(f"chime_6 attr: only {len(rows)}")
    return rows[:N]


def build_aishell_5() -> list[dict]:
    by_wav: dict[str, list[dict]] = defaultdict(list)
    with AISHELL5_INDEX.open(encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            rel = obj["audios"][0]
            meta = obj["metadata"]
            txt = obj["messages"][-1]["content"]
            if not txt or not str(txt).strip():
                continue
            by_wav[rel].append(
                {
                    "speaker": meta["speaker_id"],
                    "start": float(meta["start_time"]),
                    "end": float(meta["end_time"]),
                    "text": str(txt).strip(),
                }
            )

    cands: list[dict] = []
    for rel, items in by_wav.items():
        speakers = {x["speaker"] for x in items}
        if len(speakers) < 2 or len(items) < 4:
            continue
        wav = AISHELL5_ROOT / rel
        if not wav.is_file():
            continue
        tmax = max(x["end"] for x in items)
        t = 0.0
        while t + WIN <= tmax + 1e-6:
            win_items = [
                x for x in items if x["end"] > t and x["start"] < t + WIN
            ]
            spks = {x["speaker"] for x in win_items}
            if len(spks) >= 2 and len(win_items) >= 2:
                stem = Path(rel).stem
                sid = f"aishell5_attr_{Path(rel).parent.name}_{stem}_w{int(t):04d}"
                cands.append(
                    {
                        "sample_id": sid,
                        "rel": rel,
                        "window_start": t,
                        "items": win_items,
                        "wav": wav,
                    }
                )
            t += 20.0

    picked = sample_stride(cands, N)
    out = []
    for row in picked:
        ws = float(row["window_start"])
        utts = [
            (x["start"], x["end"], x["speaker"], x["text"]) for x in row["items"]
        ]
        utterances = clip_utterances(utts, ws, ws + WIN)
        if len({u["speaker"] for u in utterances}) < 2:
            continue
        clip = CLIPS / "aishell_5" / f"{row['sample_id']}.wav"
        ffmpeg_cut(row["wav"], clip, ws, WIN)
        out.append(
            {
                "sample_id": row["sample_id"],
                "dataset": "aishell_5",
                "subset": "eval1",
                "protocol": "msswift_window",
                "language": "zh",
                "window_start": ws,
                "WavPath": str(clip),
                "text": {"utterances": utterances},
                "n_speakers": len({u["speaker"] for u in utterances}),
                "n_utterances": len(utterances),
            }
        )
    if len(out) < N:
        raise RuntimeError(f"aishell_5 attr: only {len(out)}")
    return out[:N]


def build_cn_celeb() -> list[dict]:
    """SID diagnostic — not true speaker_attribution; gold is speaker_id."""
    data = json.loads(CN_CELEB_JSON.read_text(encoding="utf-8"))
    by_spk: dict[str, list[dict]] = defaultdict(list)
    for obj in data:
        spk = obj.get("speaker_id") or obj["messages"][1]["content"]
        rel = obj["audios"][0]
        wav = CN_CELEB_ROOT / rel
        if not wav.is_file():
            continue
        by_spk[spk].append(
            {
                "sample_id": obj["id"],
                "WavPath": str(wav),
                "text": spk,
                "speaker_id": spk,
                "dataset": "cn_celeb",
                "subset": "audiobench_sid",
                "protocol": "sid_mislabeled_as_attribution",
                "language": "zh",
            }
        )
    # one sample per speaker, stride across speakers
    speakers = sorted(by_spk)
    picked_spks = sample_stride(speakers, N)
    out = []
    for spk in picked_spks:
        out.append(by_spk[spk][0])
    if len(out) < N:
        raise RuntimeError(f"cn_celeb: only {len(out)}")
    return out[:N]


def main(only: list[str] | None = None) -> None:
    builders = {
        "aishell_4": build_aishell_4,
        "aishell_5": build_aishell_5,
        "alimeeting": build_alimeeting,
        "ami": build_ami,
        "chime_6": build_chime_6,
        "cn_celeb": build_cn_celeb,
    }
    names = only or list(builders)
    for name in names:
        if name not in builders:
            raise SystemExit(f"unknown dataset: {name}; choose from {list(builders)}")
    summary_path = ROOT / "summary_build.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() and only else {}
    for name in names:
        fn = builders[name]
        print(f"[attr] build {name}...", flush=True)
        rows = fn()
        write_jsonl(ROOT / name / "manifest.jsonl", rows)
        summary[name] = len(rows)
        print(f"  -> {len(rows)}", flush=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("[attr] done", summary)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)
