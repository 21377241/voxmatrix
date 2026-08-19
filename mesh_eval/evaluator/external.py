from typing import Any, Dict

from audio_evals.evaluator.base import Evaluator
from mesh_eval.evaluator.utils import first_present, reference_object


class CometEvaluator(Evaluator):
    def __init__(self, model_name: str = "comet-wmt22"):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        source = first_present(
            kwargs.get("source_text"),
            kwargs.get("source"),
            kwargs.get("src"),
            kwargs.get("input_text"),
        )
        if source is None:
            raise ValueError("COMET requires source_text/source/src in the sample")
        result = self.model.inference(
            {"src": str(source), "mt": str(pred), "ref": str(label)}
        )
        return {"comet": float(result["score"])}


class BertScoreEvaluator(Evaluator):
    def __init__(self, model_name: str = "bertscore-roberta-large"):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.model.inference({"pred": str(pred), "ref": label})
        return {
            "bert_score": min(1.0, float(result["f1"])),
            "bert_score_precision": min(1.0, float(result["precision"])),
            "bert_score_recall": min(1.0, float(result["recall"])),
        }


class JerEvaluator(Evaluator):
    def __init__(self, model_name: str = "jer-pyannote"):
        from audio_evals.registry import registry

        self.model = registry.get_model(model_name)

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        reference_segments = first_present(reference.get("segments"), label)
        result = self.model.inference(
            {"hypothesis": pred, "reference": reference_segments}
        )
        return {"jer": float(result["jer"]), "jer_valid": 1}
