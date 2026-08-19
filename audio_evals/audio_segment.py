import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict


def _write_with_soundfile(source: Path, target: Path, start: float, end: float) -> None:
    import soundfile as sound_file

    open_kwargs = {}
    if source.suffix.lower() == ".pcm":
        open_kwargs = {
            "samplerate": 16000,
            "channels": 1,
            "format": "RAW",
            "subtype": "PCM_16",
        }
    with sound_file.SoundFile(str(source), mode="r", **open_kwargs) as handle:
        start_frame = int(start * handle.samplerate)
        frame_count = max(1, int((end - start) * handle.samplerate))
        handle.seek(start_frame)
        audio = handle.read(frame_count, dtype="float32", always_2d=True)
        if len(audio) == 0:
            raise RuntimeError(f"audio segment is outside source duration: {source}")
        if audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        sound_file.write(
            str(target),
            audio,
            handle.samplerate,
            subtype="PCM_16",
            format="WAV",
        )


def _write_with_ffmpeg(source: Path, target: Path, start: float, end: float) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("soundfile could not decode the source and ffmpeg is unavailable")
    command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y"]
    if source.suffix.lower() == ".pcm":
        command.extend(["-f", "s16le", "-ar", "16000", "-ac", "1"])
    command.extend(
        [
            "-ss",
            f"{start:.6f}",
            "-t",
            f"{end - start:.6f}",
            "-i",
            str(source),
            "-vn",
            "-acodec",
            "pcm_s16le",
            str(target),
        ]
    )
    subprocess.run(command, check=True, capture_output=True, text=True)


def materialize_audio_segment(value: Dict[str, Any]) -> str:
    source = Path(str(value.get("path") or value.get("audio_path") or ""))
    start = float(value["start_time"])
    end = float(value["end_time"])
    if not source.is_file():
        raise FileNotFoundError(f"audio segment source not found: {source}")
    if start < 0 or end <= start:
        raise ValueError(f"invalid audio segment: start={start}, end={end}")

    cache_root = Path(
        os.environ.get("MESH_AUDIO_SEGMENT_CACHE")
        or Path(tempfile.gettempdir()) / "voxmatrix_audio_segments"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(
        f"mono-v1\0{source.resolve()}\0{start:.6f}\0{end:.6f}".encode("utf-8")
    ).hexdigest()
    target = cache_root / fingerprint[:2] / f"{fingerprint}.wav"
    if target.is_file() and target.stat().st_size > 44:
        return str(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{fingerprint}.", suffix=".wav", dir=target.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        try:
            _write_with_soundfile(source, temporary_path, start, end)
        except (RuntimeError, OSError):
            _write_with_ffmpeg(source, temporary_path, start, end)
        if temporary_path.stat().st_size <= 44:
            raise RuntimeError(f"empty audio segment generated from {source}")
        os.replace(temporary_path, target)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        if target.is_file() and target.stat().st_size > 44:
            return str(target)
        raise
    return str(target)
