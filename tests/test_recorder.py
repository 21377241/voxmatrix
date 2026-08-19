import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from audio_evals.recorder import Recorder


def _append_process(path, process_id, count):
    recorder = Recorder(path, overwrite=False)
    for index in range(count):
        recorder.add(
            {
                "type": "eval",
                "id": f"process-{process_id}-{index}",
                "data": {"payload": "并发事件" * 20},
            }
        )


def test_recorder_serializes_concurrent_threads_without_corruption(tmp_path):
    path = tmp_path / "thread-events.jsonl"
    recorder = Recorder(str(path))
    expected = {f"thread-{index}" for index in range(500)}

    def write(index):
        recorder.add(
            {
                "type": "eval",
                "id": f"thread-{index}",
                "data": {"payload": "语音评测" * 50},
            }
        )

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(write, range(500)))

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 500
    assert {row["id"] for row in rows} == expected


def test_recorder_coordinates_appenders_across_processes(tmp_path):
    path = tmp_path / "process-events.jsonl"
    Recorder(str(path))
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_append_process, args=(str(path), process_id, 50))
        for process_id in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 200
    assert len({row["id"] for row in rows}) == 200


def test_recorder_supports_filename_without_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recorder = Recorder("events.jsonl")
    recorder.add({"type": "eval", "id": 1, "data": {}})
    assert json.loads((tmp_path / "events.jsonl").read_text())["id"] == 1
