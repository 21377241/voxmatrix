from __future__ import annotations

import sacrebleu

from audio_evals.agg.base import AggPolicy


class TranslationMetrics(AggPolicy):
    def _agg(self, score_detail):
        pred = [str(x["pred"]) for x in score_detail if x.get("pred") is not None]
        ref = [str(x["ref"]) for x in score_detail if x.get("pred") is not None]
        target = str(score_detail[0].get("target_lang") or "en").lower() if score_detail else "en"
        tokenize = "zh" if target.startswith("zh") else "13a"
        return {
            "bleu": sacrebleu.corpus_bleu(pred, [ref], tokenize=tokenize).score if pred else 0.0,
            "chrf": sacrebleu.corpus_chrf(pred, [ref]).score if pred else 0.0,
        }
