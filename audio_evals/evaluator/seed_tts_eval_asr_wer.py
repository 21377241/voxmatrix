from typing import Dict
import os
from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.wer import (
    FillerOnlyReferenceSkipped,
    compute_wer,
    normalized_ref_token_count,
)
import zhconv
import logging

logger = logging.getLogger(__name__)

CHARACTER_ERROR_LANGUAGES = ("zh", "ja", "yue", "th", "ko")


class SeedTTSEvalASRWER(Evaluator):
    def __init__(self, model_name, prompt_name, lang):
        from audio_evals.registry import registry

        self.prompt = registry.get_prompt(prompt_name)
        self.model = registry.get_model(model_name)
        self.lang = lang

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.process.tts import ExtractOmniAudioPath

        pred = ExtractOmniAudioPath()(pred)
        pred = str(pred)
        label_text = kwargs["text"]
        assert os.path.exists(pred), "must be a valid audio file, but got {}".format(
            pred
        )

        real_prompt = self.prompt.load(WavPath=pred)

        # Pass language to model for non-Chinese languages or if specified
        # Whisper model expects language in generate_kwargs
        inf_kwargs = {}
        if self.lang != "zh":
            inf_kwargs["generate_kwargs"] = {
                "language": kwargs.get("language", self.lang)
            }

        transcription = self.model.inference(real_prompt, **inf_kwargs)

        if self.lang == "zh" or kwargs.get("language") == "chinese":
            transcription = zhconv.convert(transcription, "zh-cn")

        # Same normalizer/tokenizer as the corpus aggregation in mesh agg, so
        # per-sample and overall error rates are directly comparable.
        if normalized_ref_token_count(label_text, self.lang) == 0:
            raise FillerOnlyReferenceSkipped(
                label_text, self.lang, reason="empty_after_normalization"
            )
        measure_name = "cer" if self.lang in CHARACTER_ERROR_LANGUAGES else "wer"
        error_rate = compute_wer([label_text], [transcription], language=self.lang)
        res = {
            f"{measure_name}%": error_rate * 100,
            "content_acc": max(0.0, 1.0 - error_rate),
            "intelligibility": max(0.0, 1.0 - error_rate),
            "wer_language": self.lang,
        }
        res.update(
            {
                "transcription": transcription,
                "label_text": label_text,
                "pred": pred,
                "ref": label,
            }
        )
        return res
