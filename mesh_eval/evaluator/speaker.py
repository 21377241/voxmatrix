import itertools
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.wer import compute_wer
from mesh_eval.evaluator.utils import canonical_scalar, coerce_bool, extract_json, first_present, reference_object


def _segments(value: Any) -> List[Tuple[float, float, str]]:
    if isinstance(value, str):
        try:
            value = extract_json(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, dict):
        value = value.get("segments", [])
    result = []
    for item in value or []:
        if not isinstance(item, dict):
            continue
        start = first_present(item.get("start"), item.get("start_time"), item.get("begin"))
        end = first_present(item.get("end"), item.get("end_time"))
        if end is None and start is not None and item.get("duration") is not None:
            end = float(start) + float(item["duration"])
        speaker = first_present(item.get("speaker"), item.get("speaker_id"), item.get("label"))
        if start is not None and end is not None and speaker is not None and float(end) > float(start):
            result.append((float(start), float(end), canonical_scalar(speaker)))
    return result


def _speaker_at(segments: List[Tuple[float, float, str]], point: float) -> str:
    active = sorted(speaker for start, end, speaker in segments if start <= point < end)
    return "|".join(active)


def _best_mapping(pred_speakers: List[str], ref_speakers: List[str], overlaps: Dict[Tuple[str, str], float]) -> Dict[str, str]:
    if not pred_speakers or not ref_speakers:
        return {}
    if len(pred_speakers) > 8 or len(ref_speakers) > 8:
        return {
            pred: max(ref_speakers, key=lambda ref: overlaps.get((pred, ref), 0.0))
            for pred in pred_speakers
        }
    padded = ref_speakers + [""] * max(0, len(pred_speakers) - len(ref_speakers))
    best_score = -1.0
    best: Dict[str, str] = {}
    for permutation in itertools.permutations(padded, len(pred_speakers)):
        score = sum(overlaps.get((pred, ref), 0.0) for pred, ref in zip(pred_speakers, permutation))
        if score > best_score:
            best_score = score
            best = dict(zip(pred_speakers, permutation))
    return best


class DiarizationEvaluator(Evaluator):
    """Dependency-free DER approximation with optimal speaker permutation."""

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        ref_segments = _segments(first_present(reference.get("segments"), label))
        pred_segments = _segments(pred)
        boundaries = sorted({point for start, end, _ in ref_segments + pred_segments for point in (start, end)})
        if len(boundaries) < 2:
            return {"der": 1.0, "diarization_valid": 0}
        overlaps: Dict[Tuple[str, str], float] = {}
        intervals = []
        for left, right in zip(boundaries, boundaries[1:]):
            if right <= left:
                continue
            midpoint = (left + right) / 2
            ref_speaker = _speaker_at(ref_segments, midpoint)
            pred_speaker = _speaker_at(pred_segments, midpoint)
            duration = right - left
            intervals.append((duration, ref_speaker, pred_speaker))
            if ref_speaker and pred_speaker and "|" not in ref_speaker and "|" not in pred_speaker:
                overlaps[(pred_speaker, ref_speaker)] = overlaps.get((pred_speaker, ref_speaker), 0.0) + duration
        pred_speakers = sorted({speaker for _, _, speaker in pred_segments})
        ref_speakers = sorted({speaker for _, _, speaker in ref_segments})
        mapping = _best_mapping(pred_speakers, ref_speakers, overlaps)
        reference_time = sum(duration for duration, ref, _ in intervals if ref)
        error_time = 0.0
        for duration, ref, predicted in intervals:
            mapped = "|".join(sorted(mapping.get(item, item) for item in predicted.split("|") if item))
            if mapped != ref:
                error_time += duration
        der = error_time / reference_time if reference_time else 1.0
        return {
            "der": der,
            "der_error_duration": error_time,
            "der_reference_duration": reference_time,
            "diarization_valid": 1,
        }


class SpeakerCountEvaluator(Evaluator):
    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = first_present(reference.get("count"), label)
        parsed = pred
        try:
            value = extract_json(pred)
            if isinstance(value, dict):
                parsed = first_present(value.get("count"), value.get("speaker_count"), pred)
        except (TypeError, ValueError):
            pass
        try:
            pred_count = int(parsed)
            ref_count = int(expected)
            valid = 1
        except (TypeError, ValueError):
            pred_count = -1
            ref_count = int(expected) if str(expected).isdigit() else -2
            valid = 0
        return {"speaker_count_acc": int(valid and pred_count == ref_count), "speaker_count_valid": valid}


class SpeakerVerificationEvaluator(Evaluator):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        expected = coerce_bool(first_present(reference.get("same_speaker"), label))
        value = pred
        try:
            parsed = extract_json(pred)
            if isinstance(parsed, dict):
                value = first_present(parsed.get("score"), parsed.get("similarity"), parsed.get("same_speaker"), pred)
        except (TypeError, ValueError):
            pass
        try:
            score = float(value)
            predicted = score >= self.threshold
        except (TypeError, ValueError):
            predicted = coerce_bool(value)
            score = 1.0 if predicted else 0.0
        valid = int(expected is not None and predicted is not None)
        return {
            "speaker_verification_acc": int(valid and predicted == expected),
            "verification_score": score,
            "verification_label": int(bool(expected)),
            "verification_valid": valid,
        }


def _utterances(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, str):
        try:
            value = extract_json(value)
        except (TypeError, ValueError):
            return []
    if isinstance(value, dict):
        value = value.get("utterances", [])
    return [item for item in (value or []) if isinstance(item, dict)]


class SpeakerAttributionEvaluator(Evaluator):
    """Speaker assignment accuracy and permutation-invariant concatenated error rate."""

    def _eval(self, pred: Any, label: Any, **kwargs: Any) -> Dict[str, Any]:
        reference = reference_object(label, kwargs)
        refs = _utterances(first_present(reference.get("utterances"), label))
        preds = _utterances(pred)
        if not refs or not preds:
            return {"attribution_acc": 0.0, "cpcer%": 100.0, "attribution_valid": 0}

        ref_by_speaker: Dict[str, List[str]] = defaultdict(list)
        pred_by_speaker: Dict[str, List[str]] = defaultdict(list)
        for item in refs:
            speaker = canonical_scalar(first_present(item.get("speaker"), item.get("speaker_id")))
            ref_by_speaker[speaker].append(str(first_present(item.get("text"), item.get("transcript"), "")))
        for item in preds:
            speaker = canonical_scalar(first_present(item.get("speaker"), item.get("speaker_id")))
            pred_by_speaker[speaker].append(str(first_present(item.get("text"), item.get("transcript"), "")))

        ref_speakers = sorted(ref_by_speaker)
        pred_speakers = sorted(pred_by_speaker)
        language = str(kwargs.get("language") or "zh")
        language = "zh" if language.startswith("zh") else "en"
        speaker_count = max(len(pred_speakers), len(ref_speakers))
        padded_preds = pred_speakers + [
            f"__missing_pred_{index}"
            for index in range(speaker_count - len(pred_speakers))
        ]
        padded_refs = ref_speakers + [""] * (speaker_count - len(ref_speakers))
        best_cost = float("inf")
        best_mapping: Dict[str, str] = {}
        permutations = itertools.permutations(padded_refs, speaker_count)
        if speaker_count > 8:
            permutations = [tuple(padded_refs)]
        for assignment in permutations:
            costs = []
            for pred_speaker, ref_speaker in zip(padded_preds, assignment):
                pred_text = " ".join(pred_by_speaker.get(pred_speaker, []))
                ref_text = " ".join(ref_by_speaker.get(ref_speaker, []))
                if not ref_text:
                    costs.append(float(bool(pred_text)))
                elif not pred_text:
                    costs.append(1.0)
                else:
                    costs.append(compute_wer([ref_text], [pred_text], language=language))
            cost = sum(costs) / len(costs) if costs else 1.0
            if cost < best_cost:
                best_cost = cost
                best_mapping = dict(zip(padded_preds, assignment))

        ref_by_id = {
            str(first_present(item.get("utterance_id"), item.get("id"), index)): canonical_scalar(
                first_present(item.get("speaker"), item.get("speaker_id"))
            )
            for index, item in enumerate(refs)
        }
        correct = 0
        compared = 0
        for index, item in enumerate(preds):
            utterance_id = str(first_present(item.get("utterance_id"), item.get("id"), index))
            if utterance_id not in ref_by_id:
                continue
            predicted_speaker = canonical_scalar(first_present(item.get("speaker"), item.get("speaker_id")))
            correct += int(best_mapping.get(predicted_speaker, predicted_speaker) == ref_by_id[utterance_id])
            compared += 1
        return {
            "attribution_acc": correct / compared if compared else 0.0,
            "cpcer%": best_cost * 100,
            "attribution_valid": 1,
        }
