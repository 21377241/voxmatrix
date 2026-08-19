import json
import logging
import select
import time
import uuid
from typing import Any, Dict

from audio_evals.isolate import isolated
from audio_evals.models.model import OfflineModel


logger = logging.getLogger(__name__)


class _JsonMetricModel(OfflineModel):
    def __init__(self, command_args: Dict[str, Any], sample_params: Dict = None):
        self.command_args = command_args
        super().__init__(is_chat=False, sample_params=sample_params)

    def _inference(self, prompt, **kwargs):
        payload = dict(prompt) if isinstance(prompt, dict) else {"value": prompt}
        payload.update(kwargs)
        prefix = f"{uuid.uuid4()}->"

        _, writable, exceptional = select.select(
            [], [self.process.stdin], [self.process.stdin], 60
        )
        if exceptional or not writable:
            raise TimeoutError("Metric process stdin is not writable")
        self.process.stdin.write(f"{prefix}{json.dumps(payload, ensure_ascii=False)}\n")
        self.process.stdin.flush()

        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Metric process exited with code {self.process.returncode}"
                )
            readable, _, _ = select.select(
                [self.process.stdout, self.process.stderr], [], [], 1
            )
            for stream in readable:
                line = stream.readline().strip()
                if not line:
                    continue
                if stream is self.process.stderr:
                    logger.error("Metric process stderr: %s", line)
                    continue
                if line.startswith(prefix):
                    self.process.stdin.write(f"{prefix}close\n")
                    self.process.stdin.flush()
                    return json.loads(line[len(prefix) :])
                if line.startswith("Error:"):
                    raise RuntimeError(line)
                logger.info("Metric process stdout: %s", line)
        raise TimeoutError("Timed out waiting for metric process response")


@isolated("audio_evals/lib/text_metrics/comet_metric.py")
class CometMetricModel(_JsonMetricModel):
    def __init__(
        self,
        path: str,
        encoder_path: str,
        sample_params: Dict = None,
    ):
        super().__init__(
            {"path": path, "encoder_path": encoder_path},
            sample_params=sample_params,
        )


@isolated("audio_evals/lib/text_metrics/bert_score_metric.py")
class BertScoreMetricModel(_JsonMetricModel):
    def __init__(
        self,
        path: str,
        num_layers: int = 17,
        sample_params: Dict = None,
    ):
        super().__init__(
            {"path": path, "num_layers": num_layers},
            sample_params=sample_params,
        )


@isolated("audio_evals/lib/text_metrics/jer.py")
class JerMetricModel(_JsonMetricModel):
    def __init__(
        self,
        collar: float = 0.0,
        skip_overlap: bool = False,
        sample_params: Dict = None,
    ):
        super().__init__(
            {"collar": collar, "skip_overlap": str(skip_overlap).lower()},
            sample_params=sample_params,
        )
