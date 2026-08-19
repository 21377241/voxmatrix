import os.path
import threading

from audio_evals.process.base import Process


class Speech2text(Process):

    def __init__(self, model_name: str = "whisper", prompt_name: str = "whisper-asr"):
        self.model_name = model_name
        self.prompt_name = prompt_name
        self.model = None
        self.prompt = None
        self._load_lock = threading.Lock()

    def _ensure_loaded(self):
        if self.model is not None:
            return
        with self._load_lock:
            if self.model is not None:
                return
            from audio_evals.registry import registry

            self.model = registry.get_model(self.model_name)
            self.prompt = registry.get_prompt(self.prompt_name)

    def __call__(self, answer: str) -> str:
        assert os.path.exists(answer), "must be a valid audio file, but got {}".format(
            answer
        )
        self._ensure_loaded()
        real_prompt = self.prompt.load(WavPath=answer)
        return self.model.inference(real_prompt)
