import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import soundfile as sf
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from audio_evals.config import get_environment
from audio_evals.dataset.dataset import Dataset as BaseDataset

logger = logging.getLogger(__name__)

CACHE_FORMAT_VERSION = "hf-audio-cache-v2"


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _safe_component(value: Any) -> str:
    text = str(value or "default").strip().replace("\\", "/")
    text = text.replace("/", "--")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._") or "default"


def _audio_columns(dataset: Dataset) -> List[str]:
    return [
        column
        for column in dataset.column_names
        if column == "audio"
        or (column.startswith("audio") and column[5:].isdigit())
    ]


def _wav_column(column: str) -> str:
    return "WavPath" if column == "audio" else f"WavPath{column[5:]}"


def _wav_path(save_path: str, column: str, index: int) -> str:
    suffix = "" if column == "audio" else f"_{column}"
    return os.path.abspath(os.path.join(save_path, f"{index}{suffix}.wav"))


def _valid_wav(path: str) -> bool:
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        return False
    try:
        info = sf.info(path)
    except (OSError, RuntimeError):
        return False
    return info.frames > 0 and info.samplerate > 0


@contextmanager
def _exclusive_lock(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")
    try:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:
            logger.warning("fcntl is unavailable; cache lock is process-local only: %s", path)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            pass
        handle.close()


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_wav(
    output_path: str,
    array: Any,
    sampling_rate: int,
    subtype: str,
) -> None:
    directory = os.path.dirname(output_path)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.", suffix=".tmp.wav", dir=directory
    )
    os.close(fd)
    try:
        sf.write(
            temporary,
            array,
            int(sampling_rate),
            format="WAV",
            subtype=subtype,
        )
        if not _valid_wav(temporary):
            raise RuntimeError(f"temporary WAV failed validation: {temporary}")
        os.replace(temporary, output_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _replace_column(dataset: Dataset, name: str, values: List[str]) -> Dataset:
    if name in dataset.column_names:
        dataset = dataset.remove_columns(name)
    return dataset.add_column(name, values)


def _range_marker_path(
    save_path: str,
    index_offset: int,
    count: int,
    audio_columns: List[str],
    subtype: str,
) -> str:
    marker_key = hashlib.sha256(
        _stable_json(
            {
                "offset": index_offset,
                "count": count,
                "audio_columns": audio_columns,
                "subtype": subtype,
            }
        ).encode("utf-8")
    ).hexdigest()[:16]
    return os.path.join(save_path, ".ranges", f"{marker_key}.json")


def save_audio_to_local(
    ds: Dataset,
    save_path: str,
    *,
    index_offset: int = 0,
    audio_subtype: str = "PCM_16",
    keep_audio_columns: bool = False,
    verify_existing: bool = True,
    cache_identity: Optional[Dict[str, Any]] = None,
) -> Dataset:
    """Materialize HF Audio columns into a deterministic, concurrent-safe cache.

    Existing valid WAV files are attached without reading ``example[column]``.
    Audio arrays are removed by default so converting the result to a Python list
    does not decode the source audio a second time.
    """
    if index_offset < 0:
        raise ValueError("index_offset must be non-negative")
    columns = _audio_columns(ds)
    if not columns:
        logger.warning("audio field not found; returning dataset without WAV paths")
        return ds

    save_path = os.path.abspath(save_path)
    os.makedirs(save_path, exist_ok=True)
    expected_paths = {
        column: [
            _wav_path(save_path, column, index_offset + index)
            for index in range(len(ds))
        ]
        for column in columns
    }

    def path_is_ready(path: str) -> bool:
        return _valid_wav(path) if verify_existing else os.path.isfile(path)

    complete = all(
        path_is_ready(path)
        for paths in expected_paths.values()
        for path in paths
    )

    if not complete:

        def save_audio(example, index):
            global_index = index_offset + index
            for column in columns:
                output_path = _wav_path(save_path, column, global_index)
                example[_wav_column(column)] = output_path
                if path_is_ready(output_path):
                    continue
                lock_path = f"{output_path}.lock"
                with _exclusive_lock(lock_path):
                    if path_is_ready(output_path):
                        continue
                    audio = example[column]
                    if not isinstance(audio, dict) or audio.get("array") is None:
                        raise ValueError(
                            f"invalid {column} field for row {global_index}: "
                            "expected decoded array and sampling_rate"
                        )
                    sampling_rate = audio.get("sampling_rate")
                    if not sampling_rate:
                        raise ValueError(
                            f"missing sampling_rate in {column} for row {global_index}"
                        )
                    _atomic_write_wav(
                        output_path,
                        audio["array"],
                        int(sampling_rate),
                        audio_subtype,
                    )
            return example

        ds = ds.map(save_audio, with_indices=True, load_from_cache_file=False)
    else:
        for column in columns:
            ds = _replace_column(ds, _wav_column(column), expected_paths[column])

    if not keep_audio_columns:
        removable = [column for column in columns if column in ds.column_names]
        if removable:
            ds = ds.remove_columns(removable)

    marker = _range_marker_path(
        save_path, index_offset, len(ds), columns, audio_subtype
    )
    marker_payload = {
        "format_version": CACHE_FORMAT_VERSION,
        "offset": index_offset,
        "count": len(ds),
        "audio_columns": columns,
        "audio_subtype": audio_subtype,
        "cache_identity": cache_identity or {},
    }
    _atomic_write_json(marker, marker_payload)
    return ds


def _rename_aliases(dataset: Dataset, col_aliases: Dict[str, str]) -> Dataset:
    for source, target in col_aliases.items():
        if source not in dataset.column_names:
            continue
        if target in dataset.column_names and source != target:
            raise ValueError(f"col_aliases conflict with existing column name: {target}")
        if source != target:
            dataset = dataset.rename_column(source, target)
    return dataset


def _cache_identity(
    *,
    name: str,
    subset: Optional[str],
    requested_split: str,
    actual_split: str,
    local_path: str,
    data_files: Any,
    revision: str,
    dataset: Dataset,
    col_aliases: Dict[str, str],
    audio_subtype: str,
) -> Dict[str, Any]:
    return {
        "format_version": CACHE_FORMAT_VERSION,
        "name": name,
        "subset": subset or "",
        "requested_split": requested_split or "",
        "actual_split": actual_split or "",
        "local_path": os.path.abspath(local_path) if local_path else "",
        "data_files": data_files,
        "revision": revision or "",
        "dataset_fingerprint": str(getattr(dataset, "_fingerprint", "")),
        "col_aliases": col_aliases,
        "audio_subtype": audio_subtype,
    }


def _cache_path(audio_cache_root: str, name: str, identity: Dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json(identity).encode("utf-8")).hexdigest()[:20]
    split = _safe_component(identity.get("actual_split") or "single")
    return os.path.join(
        os.path.abspath(audio_cache_root),
        _safe_component(name),
        digest,
        split,
    )


def _select_range(dataset: Dataset, offset: int, limit: int) -> Dataset:
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    if offset >= len(dataset):
        return dataset.select([])
    end = min(len(dataset), offset + limit) if limit else len(dataset)
    return dataset.select(range(offset, end))


def load_audio_hf_dataset(
    name,
    subset=None,
    split="",
    local_path="",
    col_aliases=None,
    limit=0,
    data_files=None,
    *,
    offset=0,
    revision="",
    cache_dir="",
    audio_cache_root="raw",
    audio_subtype="PCM_16",
    keep_audio_columns=False,
    verify_existing_audio=True,
    trust_remote_code=True,
    return_metadata=False,
):
    if col_aliases is None:
        col_aliases = {}
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    if local_path:
        ds = load_from_disk(local_path)
        if isinstance(ds, DatasetDict) and split and split not in {"all", "*"}:
            if split not in ds:
                raise KeyError(
                    f"split '{split}' not found in {local_path}; available={list(ds)}"
                )
            ds = ds[split]
    else:
        load_args = {"path": name}
        if subset:
            load_args["name"] = subset
        if split and split not in {"all", "*"}:
            load_args["split"] = split
        if data_files:
            load_args["data_files"] = data_files
        if revision:
            load_args["revision"] = revision
        if cache_dir:
            load_args["cache_dir"] = cache_dir
        try:
            ds = load_dataset(**load_args, trust_remote_code=trust_remote_code)
        except Exception as exc:
            logger.error("load args=%s; load dataset error: %s", load_args, exc)
            raise

    metadata: Dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "source": {
            "name": name,
            "subset": subset or "",
            "split": split or "",
            "local_path": os.path.abspath(local_path) if local_path else "",
            "data_files": data_files,
            "revision": revision or "",
        },
        "ranges": [],
    }

    def convert_range(
        dataset: Dataset,
        *,
        actual_split: str,
        local_offset: int,
        row_limit: int,
    ) -> List[Dict[str, Any]]:
        dataset = _rename_aliases(dataset, col_aliases)
        identity = _cache_identity(
            name=name,
            subset=subset,
            requested_split=split,
            actual_split=actual_split,
            local_path=local_path,
            data_files=data_files,
            revision=revision,
            dataset=dataset,
            col_aliases=col_aliases,
            audio_subtype=audio_subtype,
        )
        selected = _select_range(dataset, local_offset, row_limit)
        save_path = _cache_path(audio_cache_root, name, identity)
        converted = save_audio_to_local(
            selected,
            save_path,
            index_offset=local_offset,
            audio_subtype=audio_subtype,
            keep_audio_columns=keep_audio_columns,
            verify_existing=verify_existing_audio,
            cache_identity=identity,
        )
        metadata["ranges"].append(
            {
                "split": actual_split,
                "offset": local_offset,
                "count": len(converted),
                "cache_path": save_path,
                "cache_identity": identity,
            }
        )
        return list(converted)

    if isinstance(ds, DatasetDict):
        result: List[Dict[str, Any]] = []
        remaining_offset = offset
        remaining_limit = limit
        for key, dataset in ds.items():
            if remaining_offset >= len(dataset):
                remaining_offset -= len(dataset)
                continue
            local_limit = remaining_limit
            converted = convert_range(
                dataset,
                actual_split=str(key),
                local_offset=remaining_offset,
                row_limit=local_limit,
            )
            result.extend(converted)
            remaining_offset = 0
            if limit:
                remaining_limit -= len(converted)
                if remaining_limit <= 0:
                    break
    else:
        actual_split = split if split not in {"all", "*"} else ""
        result = convert_range(
            ds,
            actual_split=actual_split,
            local_offset=offset,
            row_limit=limit,
        )

    metadata["row_count"] = len(result)
    if return_metadata:
        return result, metadata
    return result


class Huggingface(BaseDataset):
    def __init__(
        self,
        name: str,
        default_task: str,
        ref_col: str,
        subset: Optional[str] = None,
        split: str = "",
        local_path: str = "",
        col_aliases: Dict[str, str] = None,
        data_files=None,
        revision: str = "",
        cache_dir: str = "",
        audio_cache_root: str = "",
        audio_subtype: str = "PCM_16",
        keep_audio_columns: bool = False,
        verify_existing_audio: bool = True,
        trust_remote_code: bool = True,
    ):
        super().__init__(default_task, ref_col, col_aliases)
        self.name = name
        self.subset = subset
        self.split = split
        self.local_path = local_path
        self.data_files = data_files
        self.revision = revision
        self.cache_dir = cache_dir
        self.audio_cache_root = audio_cache_root or get_environment(
            "VOXMATRIX_AUDIO_CACHE", "raw"
        )
        self.audio_subtype = audio_subtype
        self.keep_audio_columns = keep_audio_columns
        self.verify_existing_audio = verify_existing_audio
        self.trust_remote_code = trust_remote_code
        self.last_load_metadata: Dict[str, Any] = {}

    def load(self, limit=0) -> List[Dict[str, Any]]:
        return self.load_slice(offset=0, limit=limit)

    def load_slice(self, offset=0, limit=0) -> List[Dict[str, Any]]:
        logger.info(
            "loading Hugging Face dataset %s (offset=%s, limit=%s)",
            self.name,
            offset,
            limit,
        )
        result, metadata = load_audio_hf_dataset(
            self.name,
            self.subset,
            self.split,
            self.local_path,
            self.col_aliases,
            limit=limit,
            data_files=self.data_files,
            offset=offset,
            revision=self.revision,
            cache_dir=self.cache_dir,
            audio_cache_root=self.audio_cache_root,
            audio_subtype=self.audio_subtype,
            keep_audio_columns=self.keep_audio_columns,
            verify_existing_audio=self.verify_existing_audio,
            trust_remote_code=self.trust_remote_code,
            return_metadata=True,
        )
        self.last_load_metadata = metadata
        return result
