from pathlib import Path

from audio_evals.config import expand_environment, get_environment
from audio_evals.registry import Registry
from mesh_eval.core.benchmark_map import load_benchmark_map
from mesh_eval.dataset.unified import UnifiedSpeechJsonl


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_environment_templates_expand_recursively(monkeypatch):
    monkeypatch.setenv("VOXMATRIX_TEST_ROOT", "/portable/root")
    value = {
        "path": "${VOXMATRIX_TEST_ROOT:-fallback}/file.wav",
        "nested": ["${MISSING_TEST_ROOT:-relative}/asset"],
    }
    assert expand_environment(value) == {
        "path": "/portable/root/file.wav",
        "nested": ["relative/asset"],
    }


def test_legacy_environment_aliases_are_bidirectional(monkeypatch):
    monkeypatch.delenv("VOXMATRIX_RAW_ROOT", raising=False)
    monkeypatch.setenv("ULTRAEVAL_RAW_ROOT", "/legacy/raw")
    assert get_environment("VOXMATRIX_RAW_ROOT") == "/legacy/raw"
    assert expand_environment("${VOXMATRIX_RAW_ROOT:-raw}/one.wav") == (
        "/legacy/raw/one.wav"
    )

    monkeypatch.setenv("VOXMATRIX_RAW_ROOT", "/canonical/raw")
    assert get_environment("ULTRAEVAL_RAW_ROOT") == "/canonical/raw"
    assert expand_environment("${ULTRAEVAL_RAW_ROOT:-raw}/one.wav") == (
        "/canonical/raw/one.wav"
    )


def test_registry_expands_environment_paths_without_mutating_yaml(monkeypatch, tmp_path):
    registry_root = tmp_path / "registry"
    model_dir = registry_root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "fake.yaml").write_text(
        "fake:\n"
        "  class: audio_evals.models.model.Model\n"
        "  args:\n"
        "    path: '${VOXMATRIX_TEST_MODEL_ROOT:-models}/fake'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VOXMATRIX_TEST_MODEL_ROOT", "/model-store")

    spec = Registry([registry_root])._model["fake"]

    assert spec["args"]["path"] == "/model-store/fake"
    assert "${VOXMATRIX_TEST_MODEL_ROOT" in (
        model_dir / "fake.yaml"
    ).read_text(encoding="utf-8")


def test_benchmark_map_honors_data_root_overrides(monkeypatch):
    monkeypatch.setenv("VOXMATRIX_HF_DATA_ROOT", "/datasets/hf")
    monkeypatch.setenv("VOXMATRIX_RAW_ROOT", "/datasets/raw")
    load_benchmark_map.cache_clear()
    try:
        data = load_benchmark_map()
    finally:
        load_benchmark_map.cache_clear()

    profile = data["benchmarks"]["librispeech-test-clean"]["data"]
    assert profile["local_path"] == "/datasets/hf/librispeech_c"
    assert profile["raw_audio_path"].startswith("/datasets/raw/")
    assert "historical_events" not in profile


def test_unified_jsonl_expands_portable_audio_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("VOXMATRIX_RAW_ROOT", "/configured/raw")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "{"
        '"sample_id":"portable-1",'
        '"dataset":"portable",'
        '"task":"speech_understanding",'
        '"capability":"asr",'
        '"scenario":"phone",'
        '"condition":{"acoustic":"clean","spatial":"near_field",'
        '"speaker":"single_speaker","device":"phone_mic",'
        '"interaction":"single_turn"},'
        '"input":{"audio_path":"${VOXMATRIX_RAW_ROOT:-raw}/one.wav"},'
        '"reference":{"text":"hello"},'
        '"metrics":["wer"],'
        '"use_bucket":"diagnostic_evidence",'
        '"split":"test"'
        "}\n",
        encoding="utf-8",
    )

    row = UnifiedSpeechJsonl(
        str(manifest), default_task="ASR-en", strict=True
    ).load()[0]

    assert row["WavPath"] == "/configured/raw/one.wav"


def test_bundled_smoke_manifest_resolves_audio_relative_to_manifest():
    dataset = UnifiedSpeechJsonl(
        "mesh_eval/data/unified_speech_smoke.jsonl",
        default_task="mesh-unified-speech",
        strict=True,
        check_audio_exists=True,
    )

    row = dataset.load()[0]

    assert Path(row["WavPath"]).resolve() == (
        REPO_ROOT / "mesh_eval" / "data" / "default.wav"
    ).resolve()


def test_distribution_declares_mesh_annotation_and_registry_resources():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "voxmatrix"' in pyproject
    assert 'include = ["voxmatrix*", "audio_evals*", "mesh_eval*", "annotation*", "registry*"]' in pyproject
    assert 'voxmatrix = "audio_evals.main:main"' in pyproject
    assert 'ultraeval-audio = "audio_evals.main:main"' in pyproject
    assert 'mesh_eval = ["config/*.yaml"' in pyproject
    assert 'annotation = ["schema/*"' in pyproject
    assert 'registry = ["**/*.yaml"]' in pyproject
    assert '"**/*.wav"' in pyproject


def test_voxmatrix_public_facade_and_legacy_packages_import():
    import audio_evals
    import mesh_eval
    import voxmatrix

    assert voxmatrix.__version__ == "0.0.0"
    assert audio_evals is not None
    assert mesh_eval is not None


def test_active_runtime_files_have_no_personal_afs_paths():
    roots = [
        REPO_ROOT / "audio_evals",
        REPO_ROOT / "registry",
        REPO_ROOT / "mesh_eval" / "config",
        REPO_ROOT / "mesh_eval" / "core",
        REPO_ROOT / "mesh_eval" / "dataset",
        REPO_ROOT / "mesh_eval" / "scripts",
        REPO_ROOT / "mesh_eval" / "data",
        REPO_ROOT / "annotation" / "pipeline",
    ]
    first_party_lib_files = {
        REPO_ROOT / "audio_evals" / "lib" / "coco.py",
        REPO_ROOT / "audio_evals" / "lib" / "cpm_tts" / "processor.py",
        REPO_ROOT / "audio_evals" / "lib" / "cpm_tts" / "chattts.py",
    }
    offenders = []
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(REPO_ROOT)
            if "lib" in relative.parts and path not in first_party_lib_files:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if any(
                prefix in text
                for prefix in ("/mnt/afs/users/", "/DATA/disk1/home/", "/home/")
            ):
                offenders.append(str(relative))
    assert offenders == []
