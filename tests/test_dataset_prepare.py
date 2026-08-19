import json
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import soundfile as sf
from datasets import Dataset as HFDataset

from audio_evals.dataset.huggingface import (
    _cache_identity,
    _cache_path,
    load_audio_hf_dataset,
    save_audio_to_local,
)
from audio_evals.dataset.prepared import PreparedAudioJsonl
from audio_evals.prepare_dataset import prepare_dataset
from audio_evals.registry import Registry
from audio_evals.main import preload_dataset
from mesh_eval.scripts import build_manifest_from_registry as manifest_builder


def _audio_row(value=0.1):
    return {
        "audio": {
            "array": [0.0, value, 0.0],
            "sampling_rate": 16000,
        },
        "text": "hello",
    }


def test_hf_slice_is_selected_before_audio_materialization(monkeypatch, tmp_path):
    import audio_evals.dataset.huggingface as hf

    source = HFDataset.from_dict({"row": list(range(100))})
    captured = {}

    monkeypatch.setattr(hf, "load_dataset", lambda **kwargs: source)

    def fake_save(dataset, save_path, **kwargs):
        captured["rows"] = list(dataset["row"])
        captured["index_offset"] = kwargs["index_offset"]
        return dataset

    monkeypatch.setattr(hf, "save_audio_to_local", fake_save)
    rows = load_audio_hf_dataset(
        "local/fake",
        offset=73,
        limit=3,
        audio_cache_root=str(tmp_path),
    )

    assert [row["row"] for row in rows] == [73, 74, 75]
    assert captured == {"rows": [73, 74, 75], "index_offset": 73}


def test_hf_cache_identity_tracks_revision_and_fingerprint(tmp_path):
    first = HFDataset.from_dict({"value": [1]})
    second = HFDataset.from_dict({"value": [2]})

    def identity(dataset, revision):
        return _cache_identity(
            name="org/data",
            subset=None,
            requested_split="test",
            actual_split="test",
            local_path="",
            data_files={"test": "*.parquet"},
            revision=revision,
            dataset=dataset,
            col_aliases={},
            audio_subtype="PCM_16",
        )

    base = _cache_path(str(tmp_path), "org/data", identity(first, "rev-a"))
    changed_revision = _cache_path(
        str(tmp_path), "org/data", identity(first, "rev-b")
    )
    changed_data = _cache_path(
        str(tmp_path), "org/data", identity(second, "rev-a")
    )

    assert base != changed_revision
    assert base != changed_data


def test_completed_wav_cache_skips_dataset_map(monkeypatch, tmp_path):
    source = HFDataset.from_list([_audio_row()])
    first = list(save_audio_to_local(source, str(tmp_path)))
    wav_path = first[0]["WavPath"]
    first_mtime = os.stat(wav_path).st_mtime_ns

    def fail_map(*args, **kwargs):
        raise AssertionError("a complete cache must not map/decode audio again")

    monkeypatch.setattr(HFDataset, "map", fail_map)
    second = list(save_audio_to_local(source, str(tmp_path)))

    assert second[0]["WavPath"] == wav_path
    assert "audio" not in second[0]
    assert os.stat(wav_path).st_mtime_ns == first_mtime


def test_concurrent_wav_materialization_is_valid(tmp_path):
    def materialize(_):
        source = HFDataset.from_list([_audio_row(0.2)])
        return list(save_audio_to_local(source, str(tmp_path)))[0]["WavPath"]

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(materialize, range(2)))

    assert paths[0] == paths[1]
    info = sf.info(paths[0])
    assert info.frames == 3
    assert info.samplerate == 16000


def test_prepare_dataset_roundtrip_multi_audio(monkeypatch, tmp_path):
    import audio_evals.prepare_dataset as prepare_module

    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    sf.write(first, [0.0, 0.1], 16000)
    sf.write(second, [0.1, 0.0], 16000)

    class FakeDataset:
        task_name = "asr-zh"
        ref_col = "text"
        col_aliases = {}
        last_load_metadata = {"source": {"revision": "fixed"}}

        def load_slice(self, offset=0, limit=0):
            rows = [
                {
                    "WavPath": str(first),
                    "WavPath2": str(second),
                    "text": "hello",
                }
            ]
            return rows[offset : offset + limit if limit else None]

    monkeypatch.setattr(
        prepare_module.registry, "get_dataset", lambda name: FakeDataset()
    )
    output = tmp_path / "prepared"
    result = prepare_dataset(
        dataset_name="fake", output_dir=str(output), checksum="sha256"
    )

    assert result["row_count"] == 1
    loaded = PreparedAudioJsonl(
        str(output / "manifest.jsonl"),
        default_task="asr-zh",
        ref_col="text",
        verify_asset_checksums=True,
    ).load()
    assert loaded[0]["WavPath"] == str(first)
    assert loaded[0]["WavPath2"] == str(second)
    assert loaded[0]["asset_schema_version"] == "prepared-audio/1.0"

    with open(output / "metadata.json", "a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        PreparedAudioJsonl(
            str(output / "manifest.jsonl"),
            default_task="asr-zh",
            ref_col="text",
        ).load()


def test_prepared_manifest_requires_success_and_valid_checksum(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
    loader = PreparedAudioJsonl(
        str(manifest), default_task="asr", ref_col="text"
    )
    with pytest.raises(ValueError, match="missing"):
        loader.load()


def test_registry_overrides_do_not_mutate_cached_spec():
    registry = Registry([])
    registry.__dict__["_model"] = {
        "fake": {"cls": "builtins.dict", "args": {"stable": 1}}
    }

    assert registry.get_model("fake", transient=2) == {"stable": 1, "transient": 2}
    assert registry.get_model("fake") == {"stable": 1}


def test_preload_dataset_applies_shard_before_model_construction():
    calls = []

    class SeekableDataset:
        task_name = "asr"
        ref_col = "text"
        col_aliases = {}

        def load_slice(self, offset=0, limit=0):
            calls.append((offset, limit))
            return [{"text": str(index)} for index in range(offset, offset + limit)]

    loaded = preload_dataset(SeekableDataset(), offset=10, limit=2)

    assert calls == [(10, 2)]
    assert loaded.load() == [{"text": "10"}, {"text": "11"}]


def test_diagnostic_hf_loader_supports_unbounded_and_forwards_identity(monkeypatch):
    captured = {}

    def fake_load_dataset(**kwargs):
        captured.update(kwargs)
        return HFDataset.from_list([{"value": 1}, {"value": 2}])

    monkeypatch.setattr(manifest_builder, "load_dataset", fake_load_dataset)
    spec = {
        "args": {
            "name": "org/data",
            "split": "test",
            "data_files": {"test": "test-*.parquet"},
            "revision": "fixed-revision",
            "cache_dir": "/cache",
        }
    }
    rows = list(manifest_builder.iter_hf_rows(spec, limit=0, no_streaming=True))

    assert [row["value"] for row in rows] == [1, 2]
    assert captured["revision"] == "fixed-revision"
    assert captured["data_files"] == {"test": "test-*.parquet"}
    assert captured["split"] == "test"


def test_diagnostic_builder_materializes_multiple_audio_columns(tmp_path):
    row = {
        "audio1": {"array": [0.0, 0.1], "sampling_rate": 16000},
        "audio2": {"array": [0.1, 0.0], "sampling_rate": 16000},
    }
    converted = manifest_builder.save_audio_field(
        row,
        dataset_name="multi",
        index=0,
        audio_root=str(tmp_path),
        allow_missing_audio=False,
    )

    assert "audio1" not in converted and "audio2" not in converted
    assert sf.info(converted["WavPath1"]).frames == 2
    assert sf.info(converted["WavPath2"]).frames == 2


def test_manifest_builder_exits_nonzero_on_missing_dataset(monkeypatch, tmp_path):
    args = SimpleNamespace(
        output=str(tmp_path / "manifest.jsonl"),
        datasets=["missing"],
        limit_per_dataset=1,
        registry_path="",
        use_bucket="diagnostic_evidence",
        split="mini",
        loader="registry",
        mode="diagnostic",
        prepared_root="",
        checksum="none",
        audio_root=str(tmp_path / "audio"),
        no_streaming=False,
        allow_missing_audio=False,
        allow_partial=False,
        evaluator_override=[],
    )
    monkeypatch.setattr(manifest_builder, "get_args", lambda: args)
    monkeypatch.setattr(manifest_builder, "get_dataset_spec", lambda name: {})

    with pytest.raises(SystemExit) as exc_info:
        manifest_builder.main()

    assert exc_info.value.code == 1
    assert (tmp_path / "manifest.jsonl").is_file()


def test_formal_manifest_builder_rejects_json_schema_invalid_output(
    monkeypatch, tmp_path, capsys
):
    output = tmp_path / "manifest.jsonl"
    args = SimpleNamespace(
        output=str(output),
        datasets=["aishell-1"],
        limit_per_dataset=1,
        registry_path="",
        use_bucket="diagnostic_evidence",
        split="not-a-valid-split",
        loader="prepared",
        mode="formal",
        prepared_root=str(tmp_path / "prepared"),
        checksum="none",
        audio_root=str(tmp_path / "audio"),
        no_streaming=False,
        allow_missing_audio=False,
        allow_partial=False,
        evaluator_override=[],
    )
    monkeypatch.setattr(manifest_builder, "get_args", lambda: args)
    monkeypatch.setattr(
        manifest_builder,
        "get_dataset_spec",
        lambda name: {"args": {"ref_col": "text", "default_task": "ASR-zh"}},
    )
    monkeypatch.setattr(
        manifest_builder,
        "load_docs_prepared",
        lambda *args, **kwargs: (
            [{"WavPath": str(tmp_path / "audio.wav"), "text": "hello"}],
            "text",
            "ASR-zh",
        ),
    )

    with pytest.raises(SystemExit) as exc_info:
        manifest_builder.main()

    assert exc_info.value.code == 1
    assert output.read_text(encoding="utf-8") == ""
    report = capsys.readouterr().out
    assert "json_schema" in report
    assert "not-a-valid-split" in report
