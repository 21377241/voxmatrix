import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from audio_evals.models.model_pool import IsolatedModelPool
from audio_evals.models.qwen3_omni_vllm_omni import Qwen3OmniVLLMOmni


class SlotModel:
    _isolated_ready_protocol = "custom"

    def __init__(self, gpu_id, max_inflight=1, barrier=None):
        self.gpu_id = gpu_id
        self.max_inflight = max_inflight
        self.barrier = barrier

    def wait_until_ready(self):
        return None

    def inference(self, prompt, **kwargs):
        if self.barrier is not None:
            self.barrier.wait()
        return f"{self.gpu_id}:{prompt}"

    def release(self):
        return None


def test_pool_queues_each_replica_once_per_inflight_slot():
    barrier = threading.Barrier(3, timeout=5)
    pool = IsolatedModelPool(
        lambda **kwargs: SlotModel(max_inflight=3, barrier=barrier, **kwargs),
        {},
        gpu_ids=["0"],
        replicas=1,
    )
    try:
        assert pool.total_slots == 3
        assert pool.replica_metadata[0]["slots"] == 3
        # All three calls must be in flight on the single replica at once.
        with ThreadPoolExecutor(max_workers=3) as executor:
            outputs = list(executor.map(pool.inference, ["a", "b", "c"]))
        assert outputs == ["0:a", "0:b", "0:c"]
    finally:
        pool.release()


def test_pool_without_inflight_attribute_keeps_one_slot_per_replica():
    pool = IsolatedModelPool(
        lambda **kwargs: SlotModel(**kwargs), {}, gpu_ids=["0", "1"], replicas=2
    )
    try:
        assert pool.total_slots == 2
        assert pool.available_count == 2
    finally:
        pool.release()


CHILD = """
import json
import sys

print('__QWEN3_OMNI_READY__ ready', flush=True)
requests = []
for _ in range(4):
    line = sys.stdin.readline()
    prefix, body = line.split('->', 1)
    requests.append((prefix + '->', json.loads(body)))
print('unrelated child log', flush=True)
print('Error: unattributed worker noise', flush=True)
for prefix, conversation in reversed(requests):
    tag = conversation[0]['content']
    if tag == 'bad':
        print(prefix + json.dumps({'error': 'ValueError: bad input'}), flush=True)
    else:
        print(prefix + json.dumps({'text': 'assistant echo ' + tag}), flush=True)
sys.stdin.read()
"""


def _concurrent_model(max_num_seqs=4):
    model = Qwen3OmniVLLMOmni.__new__(Qwen3OmniVLLMOmni)
    Qwen3OmniVLLMOmni.__init__.__wrapped__(
        model,
        path="/model",
        max_num_seqs=max_num_seqs,
        startup_timeout=5,
        request_timeout=10,
        request_log_interval=0.1,
    )
    model.process = subprocess.Popen(
        [sys.executable, "-u", "-c", CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return model


def test_concurrent_requests_receive_their_own_out_of_order_replies():
    model = _concurrent_model()
    assert model.max_inflight == 4

    def ask(tag):
        try:
            return model._request([{"role": "user", "content": tag}])["text"]
        except RuntimeError as exc:
            return f"raised: {exc}"

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = dict(
                zip(["a", "b", "c", "bad"], executor.map(ask, ["a", "b", "c", "bad"]))
            )
    finally:
        model.release()

    assert outputs["a"] == "echo a"
    assert outputs["b"] == "echo b"
    assert outputs["c"] == "echo c"
    assert outputs["bad"].startswith("raised: ") and "bad input" in outputs["bad"]


SLOW_CHILD = r"""
import json, sys, threading, time
print('__QWEN3_OMNI_READY__ ready', flush=True)
lock = threading.Lock()
def reply(prefix):
    time.sleep(0.5)
    with lock:
        print(prefix + json.dumps({'text': 'ok'}), flush=True)
for line in sys.stdin:
    threading.Thread(target=reply, args=(line.split('->', 1)[0] + '->',)).start()
"""


def test_pool_inference_overlaps_requests_on_one_concurrent_replica():
    def factory(gpu_id=None, **kwargs):
        model = Qwen3OmniVLLMOmni.__new__(Qwen3OmniVLLMOmni)
        Qwen3OmniVLLMOmni.__init__.__wrapped__(
            model, path="/model", max_num_seqs=4, startup_timeout=10
        )
        model.process = subprocess.Popen(
            [sys.executable, "-u", "-c", SLOW_CHILD],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        model._isolated_ready_protocol = "custom"
        model.wait_until_ready = model._wait_until_ready
        return model

    pool = IsolatedModelPool(factory, {}, gpu_ids=["0"], replicas=1)
    prompt = [{"role": "user", "contents": [{"type": "text", "value": "hi"}]}]
    try:
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=4) as executor:
            outputs = list(
                executor.map(lambda _: pool.inference_with_runtime(prompt), range(4))
            )
        elapsed = time.monotonic() - start
    finally:
        pool.release()
    assert [output for output, _ in outputs] == ["ok"] * 4
    # Four 0.5s replies must overlap instead of queueing behind one lock.
    assert elapsed < 1.5


def test_speech_model_stays_serial():
    model = Qwen3OmniVLLMOmni.__new__(Qwen3OmniVLLMOmni)
    Qwen3OmniVLLMOmni.__init__.__wrapped__(
        model, path="/model", speech=True, max_num_seqs=8
    )
    assert model.max_inflight == 1


def test_concurrent_requests_fail_when_worker_exits():
    model = Qwen3OmniVLLMOmni.__new__(Qwen3OmniVLLMOmni)
    Qwen3OmniVLLMOmni.__init__.__wrapped__(
        model, path="/model", max_num_seqs=2, startup_timeout=5, request_timeout=10
    )
    model.process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import sys; print('__QWEN3_OMNI_READY__ ready', flush=True); "
            "sys.stdin.readline(); sys.exit(3)",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with pytest.raises(RuntimeError, match="exited"):
            model._request([{"role": "user", "content": "x"}])
    finally:
        model.release()
