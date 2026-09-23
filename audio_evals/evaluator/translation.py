from __future__ import annotations

import sacrebleu

from audio_evals.evaluator.base import Evaluator


class TranslationMetrics(Evaluator):
    """Per-sample BLEU/chrF details; corpus aggregation happens in the agg."""

    def _eval(self, pred, label, **kwargs):
        target = str(kwargs.get("target_lang") or "en").lower()
        tokenize = "zh" if target.startswith("zh") else "13a"
        ref = str(label)
        return {
            "bleu": sacrebleu.corpus_bleu([str(pred)], [[ref]], tokenize=tokenize).score,
            "chrf": sacrebleu.corpus_chrf([str(pred)], [[ref]]).score,
            "target_lang": target,
            "pred": pred,
            "ref": ref,
        }
