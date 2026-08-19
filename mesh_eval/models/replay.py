from typing import Any, Dict

from audio_evals.base import PromptStruct
from audio_evals.models.model import Model


class ReplayModel(Model):
    """Sentinel model for manifests carrying recorded inference in eval_info."""

    def __init__(self, is_chat: bool = True, sample_params: Dict[str, Any] = None):
        super().__init__(is_chat=is_chat, sample_params=sample_params)

    def _inference(self, prompt: PromptStruct, **kwargs: Any) -> str:
        raise RuntimeError(
            "mesh-replay received a live inference call; the sample is missing "
            "eval_info.inference.content"
        )

