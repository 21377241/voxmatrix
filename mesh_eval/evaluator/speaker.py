import itertools
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from audio_evals.evaluator.base import Evaluator
from audio_evals.lib.wer import compute_wer
from mesh_eval.evaluator.utils import canonical_scalar, coerce_bool, extract_json, first_present, reference_object

_SEGMENT_OBJECT_RE = re.compile(
    r"\{[^{}]*?"
    r"\"(?:start|start_time|begin)\"\s*:\s*(?P<start>[-+0-9.eE]+)"
    r"[^{}]*?"
    r"\"(?:end|end_time)\"\s*:\s*(?P<end>[-+0-9.eE]+)"
    r"[^{}]*?"
    r"\"(?:speaker|speaker_id|label)\"\s*:\s*\"(?P<speaker>[^\"]+)\""
    r"[^{}]*?\}",
    flags=re.DOTALL,
)


def _partial_segment_dicts(text: str) -> List[Dict[str, Any]]:
    """Recover complete segment objects from truncated / messy JSON output."""
    return [
        {
            "start": float(match.group("start")),
            "end": float(match.group("end")),
            "speaker": match.group("speaker"),
        }
        for match in _SEGMENT_OBJECT_RE.finditer(text or "")
    ]


def _audio_duration_seconds(kwargs: Dict[str, Any]) -> Optional[float]:
    runtime = kwargs.get("runtime") if isinstance(kwargs.get("runtime"), dict) else {}
    for source in (kwargs, runtime):
        if not isinstance(source, dict):
            continue
        for key in ("audio_duration_seconds", "audio_duration"):
            value = source.get(key)
            if value is None:
                continue
            try:
                duration = float(value)
            except (TypeError, ValueError):
                continue
            if duration > 0:
                return duration
    return None


def _clip_segments(
    segments: List[Tuple[float, float, str]], max_time: Optional[float]
) -> List[Tuple[float, float, str]]:
    if max_time is None:
        return segments
    clipped: List[Tuple[float, float, str]] = []
    for start, end, speaker in segments:
        if start >= max_time:
            continue
        end = min(end, max_time)
        if end > start:
            clipped.append((max(0.0, start), end, speaker))
    return clipped


def _merge_adjacent_segments(
    segments: List[Tuple[float, float, str]], gap: float = 1e-3
) -> List[Tuple[float, float, str]]:
    if not segments:
        return []
    ordered = sorted(segments, key=lambda item: (item[0], item[1], item[2]))
    merged = [ordered[0]]
    for start, end, speaker in ordered[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        if speaker == prev_speaker and start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end), prev_speaker)
        else:
            merged.append((start, end, speaker))
    return merged


def _items_to_segments(value: Any) -> List[Tuple[float, float, str]]:
    if isinstance(value, dict):
        if "segments" in value:
            value = value.get("segments", [])
        elif any(key in value for key in ("start", "start_time", "begin")) and any(
            key in value for key in ("end", "end_time", "duration")
        ):
            # extract_json may return a single segment object when the array is truncated.
            value = [value]
        else:
            value = value.get("segments", [])
    result: List[Tuple[float, float, str]] = []
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


def _json_parseable(value: str) -> bool:
    try:
        extract_json(value)
        return True
    except (TypeError, ValueError):
        return False


def _looks_like_truncated_diar_json(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped:
        return False
    if stripped.count("{") > stripped.count("}") or stripped.count("[") > stripped.count("]"):
        return True
    if stripped.startswith("```") and not stripped.rstrip().endswith("```"):
        return True
    return False


def normalize_diarization_prediction(
    value: Any, max_time: Optional[float] = None
) -> Dict[str, Any]:
    """Parse / salvage / clip diarization model output into canonical segments."""
    salvaged = 0
    truncated = 0
    parsed: Any = value
    if isinstance(value, str):
        truncated = int(_looks_like_truncated_diar_json(value))
        try:
            parsed = extract_json(value)
            # Truncated arrays often parse as the first complete object only — prefer regex salvage.
            if truncated and isinstance(parsed, dict) and "segments" not in parsed:
                recovered = _partial_segment_dicts(value)
                if len(recovered) >= 1:
                    parsed = recovered
                    salvaged = 1
        except (TypeError, ValueError):
            parsed = _partial_segment_dicts(value)
            salvaged = int(bool(parsed))
            truncated = 1
    segments = _merge_adjacent_segments(_clip_segments(_items_to_segments(parsed), max_time))
    if isinstance(value, str) and not segments and not _json_parseable(value):
        truncated = 1
    payload = {
        "segments": [
            {"start": round(start, 3), "end": round(end, 3), "speaker": speaker}
            for start, end, speaker in segments
        ]
    }
    return {
        "segments": segments,
        "payload": payload,
        "content": json.dumps(payload, ensure_ascii=False),
        "diarization_salvaged": salvaged,
        "diarization_truncated": truncated,
    }


def _segments(value: Any, max_time: Optional[float] = None) -> List[Tuple[float, float, str]]:
    return normalize_diarization_prediction(value, max_time=max_time)["segments"]


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
        max_time = _audio_duration_seconds(kwargs)
        ref_segments = _segments(first_present(reference.get("segments"), label), max_time=max_time)
        normalized = normalize_diarization_prediction(pred, max_time=max_time)
        pred_segments = normalized["segments"]
        empty_pred = not pred_segments
        empty_ref = not ref_segments
        # Empty prediction against non-empty reference is an engineering/parse failure,
        # not a silent DER=1 with failure_rate=0.
        parse_failure = int(empty_pred and not empty_ref)

        boundaries = sorted({point for start, end, _ in ref_segments + pred_segments for point in (start, end)})
        if len(boundaries) < 2:
            return {
                "der": 1.0,
                "der_error_duration": sum(end - start for start, end, _ in ref_segments) if ref_segments else 0.0,
                "der_reference_duration": sum(end - start for start, end, _ in ref_segments) if ref_segments else 0.0,
                "diarization_valid": 0,
                "diarization_salvaged": normalized["diarization_salvaged"],
                "diarization_truncated": normalized["diarization_truncated"],
                "failure": parse_failure,
                "failure_stage": "evaluation" if parse_failure else "",
                "failure_type": "diarization_empty_pred" if parse_failure else "",
            }
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
            "diarization_valid": 0 if parse_failure else 1,
            "diarization_salvaged": normalized["diarization_salvaged"],
            "diarization_truncated": normalized["diarization_truncated"],
            "failure": parse_failure,
            "failure_stage": "evaluation" if parse_failure else "",
            "failure_type": "diarization_empty_pred" if parse_failure else "",
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
