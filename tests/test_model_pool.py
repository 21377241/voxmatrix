import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from audio_evals import isolate
from audio_evals.main import resolve_pool_config
from audio_evals.models.model_pool import (
    IsolatedModelPool,
    get_available_gpu_ids,
)
from audio_evals.models.model import OfflineModel


class FakeModel:
    _isolated_ready_protocol = "custom"

    def __init__(self, gpu_id, released=None):
        self.gpu_id = gpu_id
        self.released = released if released is not None else []

    def wait_until_ready(self):
        return None

    def inference(self, prompt, **kwargs):
        return f"{self.gpu_id}:{prompt}"

    def release(self):
        self.released.append(self.gpu_id)


def _pool_args(**overrides):
    values = {
        "gpus_per_replica": 1,
        "workers": None,
        "replicas": 0,
        "inference_workers": 0,
        "allow_gpu_sharing": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cuda_visible_devices_accepts_uuid_and_mig_without_host_fallback(
    monkeypatch,
):
    monkeypatch.setenv(
        "CUDA_VISIBLE_DEVICES", "GPU-aaaaaaaa,MIG-GPU-bbbbbbbb/1/0"
    )
    monkeypatch.setattr(
        "audio_evals.models.model_pool.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("nvidia-smi must not run")
        ),
    )

    assert get_available_gpu_ids() == [
        "GPU-aaaaaaaa",
        "MIG-GPU-bbbbbbbb/1/0",
    ]


@pytest.mark.parametrize("hidden", ["", "-1", "none"])
def test_explicit_hidden_gpu_environment_is_authoritative(monkeypatch, hidden):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", hidden)
    assert get_available_gpu_ids() == []


def test_invalid_visible_gpu_token_fails_without_host_fallback(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "not-a-gpu")
    with pytest.raises(ValueError, match="unsupported GPU token"):
        get_available_gpu_ids()


def test_no_gpu_pool_fails_fast():
    with pytest.raises(RuntimeError, match="no visible GPU"):
        IsolatedModelPool(lambda **kwargs: FakeModel(**kwargs), {}, gpu_ids=[])


def test_gpu_assignments_require_explicit_sharing():
    assert IsolatedModelPool._compute_gpu_assignments(
        ["0", "1", "2", "3"], 2, 2
    ) == [["0", "1"], ["2", "3"]]
    with pytest.raises(ValueError, match="allow_gpu_sharing"):
        IsolatedModelPool._compute_gpu_assignments(["0", "1"], 3)
    assert IsolatedModelPool._compute_gpu_assignments(
        ["0", "1"], 3, allow_gpu_sharing=True
    ) == [["0"], ["1"], ["0"]]


def test_model_replicas_start_concurrently_and_dispatch():
    barrier = threading.Barrier(2)
    launched = []
    released = []

    def factory(gpu_id):
        launched.append(gpu_id)
        barrier.wait(timeout=2)
        return FakeModel(gpu_id, released)

    pool = IsolatedModelPool(
        factory,
        {},
        gpu_ids=["0", "1"],
        replicas=2,
        ready_policy="required",
    )
    try:
        assert set(launched) == {"0", "1"}
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(pool.inference, ["a", "b"]))
        assert {item.split(":", 1)[0] for item in results} == {"0", "1"}
        assert pool.available_count == 2
    finally:
        pool.release()
    assert set(released) == {"0", "1"}


def test_partial_startup_failure_releases_successful_replicas():
    barrier = threading.Barrier(2)
    released = []

    def factory(gpu_id):
        barrier.wait(timeout=2)
        if gpu_id == "1":
            raise RuntimeError("replica failed")
        return FakeModel(gpu_id, released)

    with pytest.raises(RuntimeError, match="replica failed"):
        IsolatedModelPool(factory, {}, gpu_ids=["0", "1"], replicas=2)
    assert released == ["0"]


def test_required_ready_policy_rejects_legacy_launch_only_model():
    class LegacyModel(FakeModel):
        _isolated_ready_protocol = "launch"

    released = []
    with pytest.raises(RuntimeError, match="strict readiness"):
        IsolatedModelPool(
            lambda gpu_id: LegacyModel(gpu_id, released),
            {},
            gpu_ids=["0"],
            replicas=1,
            ready_policy="required",
        )
    assert released == ["0"]


def test_release_uses_process_group_fallback_and_is_idempotent():
    terminated = []

    class Process:
        def poll(self):
            return None

    class ProcessModel(FakeModel):
        process = Process()

        def release(self):
            raise RuntimeError("hook failed")

        def _terminate_isolated_process(self):
            terminated.append(self.gpu_id)

    pool = IsolatedModelPool(
        lambda gpu_id: ProcessModel(gpu_id), {}, gpu_ids=["0"], replicas=1
    )
    pool.release()
    pool.release()
    assert terminated == ["0"]
    with pytest.raises(RuntimeError, match="released"):
        pool.inference("after-release")


def test_isolated_environment_install_is_locked_and_reused(monkeypatch, tmp_path):
    env_path = tmp_path / "env"
    python_path = env_path / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("example==1\n", encoding="utf-8")
    installs = []

    def fake_run(command, **kwargs):
        installs.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(isolate.subprocess, "run", fake_run)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: isolate._prepare_isolated_environment(
                    str(env_path), str(requirements)
                ),
                range(2),
            )
        )

    assert results == [str(python_path), str(python_path)]
    assert len(installs) == 1
    assert (env_path / ".ultraeval-prepare.json").is_file()


def test_pool_config_separates_replicas_groups_and_inference_workers():
    assert resolve_pool_config(_pool_args(), ["0", "1", "2", "3"]) == (
        4,
        1,
        4,
    )
    assert resolve_pool_config(
        _pool_args(gpus_per_replica=2), ["0", "1", "2", "3"]
    ) == (2, 2, 2)
    assert resolve_pool_config(
        _pool_args(workers=2), ["0", "1", "2", "3"]
    ) == (2, 1, 2)
    assert resolve_pool_config(
        _pool_args(replicas=2, inference_workers=7), ["0", "1"]
    ) == (2, 1, 7)
    with pytest.raises(ValueError, match="allow-gpu-sharing"):
        resolve_pool_config(_pool_args(replicas=3), ["0", "1"])


def test_incomplete_model_directory_resumes_download_and_is_locked(
    monkeypatch, tmp_path
):
    import audio_evals.models.model as model_module

    model_root = tmp_path / "models"
    local_dir = model_root / "org" / "model"
    local_dir.mkdir(parents=True)
    (local_dir / "config.json").write_text("{}", encoding="utf-8")
    downloads = []

    class Sibling:
        rfilename = "model.bin"
        size = 4

    class Api:
        def repo_info(self, **kwargs):
            return SimpleNamespace(siblings=[Sibling()])

    def fake_download(**kwargs):
        downloads.append(kwargs["local_dir"])
        (local_dir / "model.bin").write_bytes(b"1234")
        return str(local_dir)

    monkeypatch.setattr(model_module, "DEFAULT_MODEL_PATH", str(model_root))
    monkeypatch.setattr(model_module, "HfApi", Api)
    monkeypatch.setattr(model_module, "snapshot_download", fake_download)

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = list(executor.map(lambda _: OfflineModel._download_model("org/model"), range(2)))

    assert paths == [str(local_dir), str(local_dir)]
    assert downloads == [str(local_dir)]
