from collections import defaultdict
from typing import Any, Dict, Iterable, List

import sacrebleu

from audio_evals.agg.base import AggPolicy
from audio_evals.lib.wer import compute_wer

CONDITION_FIELDS = [
    "condition__acoustic",
    "condition__spatial",
    "condition__speaker",
    "condition__device",
    "condition__interaction",
]


def _v2_slice_fields() -> List[str]:
    from mesh_eval.core.schema_v2 import load_taxonomy_v2

    metadata = (load_taxonomy_v2().get("metadata") or {}).get("slice_fields") or []
    return [
        "scenario__primary",
        "scenario__secondary",
        *[f"metadata__{field}" for field in metadata],
    ]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


class MeshAgg(AggPolicy):
    def __init__(
        self,
        metric_fields: List[str] = None,
        group_by: List[Any] = None,
        min_slice_size: int = 5,
    ):
        super().__init__(need_score_col=[])
        self.metric_fields = metric_fields or []
        self.min_slice_size = int(min_slice_size)
        if self.min_slice_size < 1:
            raise ValueError("min_slice_size must be >= 1")
        self.group_by = group_by or [
            "task",
            "capability",
            ["task", "capability"],
            "scenario",
            ["scenario", "task", "capability"],
            "use_bucket",
            "split",
            "dataset_id",
            "benchmark_id",
            "language",
            ["task", "capability", "language"],
            ["task", "capability", "use_bucket"],
            *CONDITION_FIELDS,
            *[["scenario", field, "capability"] for field in CONDITION_FIELDS],
            *_v2_slice_fields(),
            *[
                ["scenario__primary", field, "capability"]
                for field in _v2_slice_fields()
                if field.startswith("metadata__")
            ],
        ]

    def __call__(self, score_detail: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self._agg(score_detail)

    def _agg(self, score_detail: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not score_detail:
            return {"sample_count": 0}

        self._corpus_cache = {}
        metric_fields = self.metric_fields or self._detect_metrics(score_detail)
        result = {"sample_count": len(score_detail)}
        self._write_v2_coverage(result, score_detail)
        self._write_metrics(
            result, "overall", score_detail, metric_fields, include_caption=True
        )

        formal_rows = [
            item for item in score_detail if item.get("use_bucket") == "formal_subscores"
        ]
        if formal_rows:
            result["formal/sample_count"] = len(formal_rows)
            self._write_metrics(
                result,
                "formal/overall",
                formal_rows,
                metric_fields,
                include_caption=True,
            )

        for group_spec in self.group_by:
            keys = group_spec if isinstance(group_spec, list) else [group_spec]
            groups = defaultdict(list)
            for item in score_detail:
                group_key = tuple(str(item.get(key, "")) for key in keys)
                if any(group_key):
                    groups[group_key].append(item)

            group_name = "+".join(keys)
            for group_key, rows in groups.items():
                group_label = "|".join(group_key)
                result[f"{group_name}/{group_label}/sample_count"] = len(rows)
                if len(rows) < self.min_slice_size:
                    result[f"{group_name}/{group_label}/warning"] = "small_sample"
                self._write_metrics(
                    result,
                    f"{group_name}/{group_label}",
                    rows,
                    metric_fields,
                    include_caption=group_name in {"dataset_id", "benchmark_id"},
                )

        return result

    @staticmethod
    def _write_v2_coverage(
        result: Dict[str, Any], rows: List[Dict[str, Any]]
    ) -> None:
        v2_rows = [
            row
            for row in rows
            if row.get("sample_schema_version") == "runtime-sample/2.0"
        ]
        if not v2_rows:
            return
        result["coverage/v2_sample_count"] = len(v2_rows)
        for field in _v2_slice_fields():
            known = [
                row
                for row in v2_rows
                if row.get(field) not in (None, "", "unknown", [])
            ]
            prefix = f"coverage/{field}"
            result[f"{prefix}/known_count"] = len(known)
            result[f"{prefix}/missing_or_unknown_count"] = len(v2_rows) - len(known)
            result[f"{prefix}/known_rate"] = len(known) / len(v2_rows)
            provenance_field = (
                "scenario"
                if field.startswith("scenario__")
                else field.removeprefix("metadata__")
            )
            sources = defaultdict(int)
            confidences = []
            for row in known:
                provenance = row.get("_slice_provenance") or {}
                if provenance_field == "scenario":
                    item = provenance.get("scenario") or {}
                else:
                    item = (provenance.get("metadata") or {}).get(
                        provenance_field, {}
                    )
                source = str(item.get("source") or "unknown")
                sources[source] += 1
                if _is_number(item.get("confidence")):
                    confidences.append(float(item["confidence"]))
            for source, count in sorted(sources.items()):
                result[f"{prefix}/source/{source}/count"] = count
            if confidences:
                result[f"{prefix}/confidence_mean"] = _mean(confidences)

    def _write_metrics(
        self,
        result: Dict[str, Any],
        prefix: str,
        rows: List[Dict[str, Any]],
        metric_fields: List[str],
        include_caption: bool = False,
    ) -> None:
        for metric in metric_fields:
            values = [row[metric] for row in rows if _is_number(row.get(metric))]
            if values:
                result[f"{prefix}/{metric}"] = _mean(values)
                result[f"{prefix}/{metric}/sample_count"] = len(values)
        self._write_corpus_text_metrics(result, prefix, rows)
        if include_caption:
            self._write_caption_metrics(result, prefix, rows)
        self._write_classification_metrics(result, prefix, rows)
        self._write_verification_metrics(result, prefix, rows)
        self._write_runtime_metrics(result, prefix, rows)

    def _write_corpus_text_metrics(
        self, result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        self._write_error_rate(result, prefix, rows, "wer%", default_language="en")
        self._write_error_rate(result, prefix, rows, "cer%", default_language="zh")
        self._write_bleu(result, prefix, rows)
        self._write_chrf(result, prefix, rows)

    def _write_error_rate(
        self,
        result: Dict[str, Any],
        prefix: str,
        rows: List[Dict[str, Any]],
        metric: str,
        default_language: str,
    ) -> None:
        metric_rows = [row for row in rows if _is_number(row.get(metric))]
        pairs = []
        for row in metric_rows:
            pred = row.get("transcription", row.get("pred"))
            ref = row.get("label_text", row.get("ref"))
            if pred not in (None, "") and ref not in (None, ""):
                pairs.append((row, str(pred), str(ref)))
        if not pairs:
            return

        languages = {
            str(row.get("language") or default_language).lower()
            for row, _pred, _ref in pairs
        }
        if len(languages) != 1:
            result.pop(f"{prefix}/{metric}", None)
            result.pop(f"{prefix}/{metric}/sample_count", None)
            return
        language = next(iter(languages))
        cache_key = (metric, language, tuple(id(row) for row, _pred, _ref in pairs))
        score = self._corpus_cache.get(cache_key)
        if score is None:
            predictions = [pred for _row, pred, _ref in pairs]
            references = [ref for _row, _pred, ref in pairs]
            score = compute_wer(references, predictions, language=language) * 100
            self._corpus_cache[cache_key] = score
        result[f"{prefix}/{metric}"] = score
        result[f"{prefix}/{metric}/sample_count"] = len(pairs)
        result[f"{prefix}/{metric}/aggregation"] = "corpus"
        if any(row.get("transcription") is not None for row, _pred, _ref in pairs):
            for derived_metric in ("content_acc", "intelligibility"):
                if any(
                    _is_number(row.get(derived_metric))
                    for row, _pred, _ref in pairs
                ):
                    result[f"{prefix}/{derived_metric}"] = max(
                        0.0, 1.0 - score / 100.0
                    )
                    result[f"{prefix}/{derived_metric}/sample_count"] = len(pairs)
                    result[
                        f"{prefix}/{derived_metric}/aggregation"
                    ] = "corpus_error_rate"

    def _write_bleu(
        self, result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        pairs = [
            (row, str(row["pred"]), str(row["ref"]))
            for row in rows
            if _is_number(row.get("bleu"))
            and row.get("pred") not in (None, "")
            and row.get("ref") not in (None, "")
        ]
        if not pairs:
            return
        tokenizers = {self._bleu_tokenizer(row) for row, _pred, _ref in pairs}
        if len(tokenizers) != 1:
            result.pop(f"{prefix}/bleu", None)
            result.pop(f"{prefix}/bleu/sample_count", None)
            return
        tokenizer = next(iter(tokenizers))
        cache_key = ("bleu", tokenizer, tuple(id(row) for row, _pred, _ref in pairs))
        score = self._corpus_cache.get(cache_key)
        if score is None:
            predictions = [pred for _row, pred, _ref in pairs]
            references = [ref for _row, _pred, ref in pairs]
            score = sacrebleu.corpus_bleu(
                predictions, [references], tokenize=tokenizer
            ).score
            self._corpus_cache[cache_key] = score
        result[f"{prefix}/bleu"] = score
        result[f"{prefix}/bleu/sample_count"] = len(pairs)
        result[f"{prefix}/bleu/aggregation"] = "corpus"

    def _write_chrf(
        self, result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        pairs = [
            (row, str(row["pred"]), str(row["ref"]))
            for row in rows
            if _is_number(row.get("chrf"))
            and row.get("pred") not in (None, "")
            and row.get("ref") not in (None, "")
        ]
        if not pairs:
            return
        cache_key = ("chrf", tuple(id(row) for row, _pred, _ref in pairs))
        score = self._corpus_cache.get(cache_key)
        if score is None:
            predictions = [pred for _row, pred, _ref in pairs]
            references = [ref for _row, _pred, ref in pairs]
            score = sacrebleu.corpus_chrf(predictions, [references]).score
            self._corpus_cache[cache_key] = score
        result[f"{prefix}/chrf"] = score
        result[f"{prefix}/chrf/sample_count"] = len(pairs)
        result[f"{prefix}/chrf/aggregation"] = "corpus"

    def _write_caption_metrics(
        self, result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        caption_rows = [
            row
            for row in rows
            if row.get("caption_prediction") not in (None, "")
            and row.get("caption_references") not in (None, "", [])
        ]
        if not caption_rows:
            return
        cache_key = ("coco", tuple(id(row) for row in caption_rows))
        scores = self._corpus_cache.get(cache_key)
        if scores is None:
            from audio_evals.lib.coco import compute_caption

            predictions = [str(row["caption_prediction"]) for row in caption_rows]
            references = [
                [str(item) for item in row["caption_references"]]
                for row in caption_rows
            ]
            scores = compute_caption(references, predictions)
            self._corpus_cache[cache_key] = scores

        mapped_scores = {
            "cider": scores.get("CIDEr", scores.get("cider")),
            "spice": scores.get("SPICE", scores.get("spice")),
        }
        requested = set()
        for row in caption_rows:
            requested.update(str(row.get("metric_names") or "").split("|"))
        requested &= set(mapped_scores)
        if not requested:
            requested = set(mapped_scores)
        missing = sorted(metric for metric in requested if mapped_scores[metric] is None)
        if missing:
            raise RuntimeError(
                "COCO corpus scoring did not produce required metrics: "
                + ", ".join(missing)
            )
        for metric in sorted(requested):
            result[f"{prefix}/{metric}"] = mapped_scores[metric]
            result[f"{prefix}/{metric}/sample_count"] = len(caption_rows)
            result[f"{prefix}/{metric}/aggregation"] = "corpus"
        result[f"{prefix}/caption_normalization"] = "first_sentence_max_64_words"

    @staticmethod
    def _bleu_tokenizer(row: Dict[str, Any]) -> str:
        metric_names = set(str(row.get("metric_names") or "").split("|"))
        evaluators = str(row.get("mesh_evaluator") or "").split("|")
        if "bleu_zh" in metric_names or "bleu-zh" in evaluators:
            return "zh"
        return "13a"

    @staticmethod
    def _write_classification_metrics(
        result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        pairs = [
            (str(row["classification_pred"]), str(row["classification_ref"]))
            for row in rows
            if row.get("classification_pred") not in (None, "")
            and row.get("classification_ref") not in (None, "")
        ]
        if not pairs:
            return
        labels = sorted({value for pair in pairs for value in pair})
        f1_values = []
        for label in labels:
            tp = sum(pred == label and ref == label for pred, ref in pairs)
            fp = sum(pred == label and ref != label for pred, ref in pairs)
            fn = sum(pred != label and ref == label for pred, ref in pairs)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1_values.append(
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        result[f"{prefix}/macro_f1"] = _mean(f1_values)
        result[f"{prefix}/macro_f1/sample_count"] = len(pairs)

    @staticmethod
    def _write_verification_metrics(
        result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        pairs = [
            (float(row["verification_score"]), int(row["verification_label"]))
            for row in rows
            if _is_number(row.get("verification_score"))
            and row.get("verification_label") in (0, 1)
        ]
        positives = [score for score, label in pairs if label == 1]
        negatives = [score for score, label in pairs if label == 0]
        if not positives or not negatives:
            return
        thresholds = [float("inf"), *sorted({score for score, _ in pairs}, reverse=True), float("-inf")]
        best_eer = 1.0
        best_gap = float("inf")
        for threshold in thresholds:
            false_accept = sum(score >= threshold for score in negatives) / len(negatives)
            false_reject = sum(score < threshold for score in positives) / len(positives)
            gap = abs(false_accept - false_reject)
            if gap < best_gap:
                best_gap = gap
                best_eer = (false_accept + false_reject) / 2
        wins = 0.0
        for positive in positives:
            for negative in negatives:
                wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
        result[f"{prefix}/eer"] = best_eer
        result[f"{prefix}/auc"] = wins / (len(positives) * len(negatives))
        result[f"{prefix}/eer/sample_count"] = len(pairs)
        result[f"{prefix}/auc/sample_count"] = len(pairs)

    @staticmethod
    def _write_runtime_metrics(
        result: Dict[str, Any], prefix: str, rows: List[Dict[str, Any]]
    ) -> None:
        for source, target in (
            ("latency_ms", "latency"),
            ("first_token_latency_ms", "first_token_latency"),
            ("first_audio_chunk_latency_ms", "first_audio_chunk_latency"),
            ("memory_peak_mb", "memory_peak"),
        ):
            values = [float(row[source]) for row in rows if _is_number(row.get(source))]
            if not values:
                continue
            for suffix, fraction in (("p50", 0.50), ("p90", 0.90), ("p95", 0.95)):
                result[f"{prefix}/{target}_{suffix}"] = _percentile(values, fraction)
            result[f"{prefix}/{target}_p50/sample_count"] = len(values)
        for source, target in (
            ("rtf", "rtf"),
            ("cpu_usage", "cpu_usage"),
            ("npu_usage", "npu_usage"),
            ("power_watts", "power_watts"),
        ):
            values = [float(row[source]) for row in rows if _is_number(row.get(source))]
            if values:
                result[f"{prefix}/{target}"] = _mean(values)
                result[f"{prefix}/{target}/sample_count"] = len(values)
        for source, target in (("timeout", "timeout_rate"), ("failure", "failure_rate")):
            values = [float(row[source]) for row in rows if _is_number(row.get(source))]
            if values:
                result[f"{prefix}/{target}"] = _mean(values)
                result[f"{prefix}/{target}/sample_count"] = len(values)
        for taxonomy in ("failure_stage", "failure_type"):
            counts = defaultdict(int)
            for row in rows:
                if row.get(taxonomy) not in (None, ""):
                    counts[str(row[taxonomy])] += 1
            for value, count in sorted(counts.items()):
                result[f"{prefix}/{taxonomy}/{value}/count"] = count

    @staticmethod
    def _detect_metrics(score_detail: List[Dict[str, Any]]) -> List[str]:
        ignored = {
            "id",
            "extract_fail",
            # These are evaluator payload/support fields, not reportable metrics.
            "pred",
            "ref",
            "classification_pred",
            "classification_ref",
            "verification_score",
            "verification_label",
            "audio_duration_seconds",
            "latency_ms",
            "first_token_latency_ms",
            "first_audio_chunk_latency_ms",
            "memory_peak_mb",
            "rtf",
            "cpu_usage",
            "npu_usage",
            "power_watts",
            "timeout",
            "failure",
        }
        ignored_prefixes = (
            "sample_count",
        )
        metrics = []
        for item in score_detail:
            for key, value in item.items():
                if key in ignored or key.startswith(ignored_prefixes):
                    continue
                if _is_number(value) and key not in metrics:
                    metrics.append(key)
        return metrics
