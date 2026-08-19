import os
from typing import Dict

from audio_evals.evaluator.base import Evaluator


class Simo(Evaluator):
    def __init__(
        self,
        model_name: str = "wavlm_large",
    ):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        reference_audio = str(kwargs.get("reference_audio_path") or label)
        if not os.path.isfile(reference_audio) and kwargs.get("WavPath"):
            reference_audio = str(kwargs["WavPath"])
        assert os.path.isfile(pred), f"Prediction file {pred} does not exist"
        assert os.path.isfile(
            reference_audio
        ), f"Reference audio file {reference_audio} does not exist"
        similarity = self.model.inference({"audios": [pred, reference_audio]})
        return {
            "sim": similarity,
            "simo": similarity,
            "pred": pred,
            "ref": reference_audio,
        }


class CV3SpeakerSim(Evaluator):
    def __init__(
        self,
        model_name: str = "speech_eres2net_sv_en_voxceleb_16k",
    ):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        pred = str(pred)
        return {
            "speaker_sim": self.model.inference({"audios": [pred, kwargs["WavPath"]]}),
            "pred": pred,
            "ref": kwargs["WavPath"],
        }
