"""FENSE evaluator for audio caption predictions."""

from __future__ import annotations

from audio_evals.evaluator.base import Evaluator


class FENSE(Evaluator):
    """Compute official FENSE sentence score against one or more references."""

    def __init__(
        self,
        sbert_model: str = "paraphrase-TinyBERT-L6-v2",
        echecker_model: str = "echecker_clotho_audiocaps_base",
        penalty: float = 0.9,
        error_threshold: float = 0.9,
        device: str | None = None,
    ):
        # FENSE v0.1 checkpoints contain the legacy BERT position_ids buffer;
        # newer Transformers omit it from the model state.  Relax only this
        # checkpoint load while keeping all metric behavior unchanged.
        from fense.model import BERTFlatClassifier

        original_load_state_dict = BERTFlatClassifier.load_state_dict

        def compatible_load_state_dict(model, state_dict, strict=True, assign=False):
            return original_load_state_dict(model, state_dict, strict=False, assign=assign)

        BERTFlatClassifier.load_state_dict = compatible_load_state_dict
        from fense.evaluator import Evaluator as FenseEvaluator

        self._metric = FenseEvaluator(
            device=device or ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
            sbert_model=sbert_model,
            echecker_model=echecker_model,
            penalty=penalty,
            error_threshold=error_threshold,
        )
        # FENSE v0.1 returns a 2-D probability array in its small-batch branch;
        # batch_size=1 routes through the branch that selects the error class.
        self._metric.batch_size = 1

    def _eval(self, pred, label, **kwargs):
        reference = kwargs.get("reference_obj") or {}
        refs = reference.get("captions") if isinstance(reference, dict) else None
        if not refs:
            refs = label if isinstance(label, list) else [label]
        refs = [str(item) for item in refs if item not in (None, "")]
        score = self._metric.sentence_score(str(pred), refs)
        return {"fense": float(score), "pred": pred, "ref": refs}
