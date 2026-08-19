import argparse
import json

from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import JaccardErrorRate

from protocol import serve


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def segments(value):
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        value = value.get("segments", [])
    return value or []


def annotation(value):
    result = Annotation()
    for index, item in enumerate(segments(value)):
        start = item.get("start", item.get("start_time", item.get("begin")))
        end = item.get("end", item.get("end_time"))
        if end is None and start is not None and item.get("duration") is not None:
            end = float(start) + float(item["duration"])
        speaker = item.get("speaker", item.get("speaker_id", item.get("label")))
        if start is None or end is None or speaker is None or float(end) <= float(start):
            continue
        result[Segment(float(start), float(end)), str(index)] = str(speaker)
    return result


parser = argparse.ArgumentParser()
parser.add_argument("--collar", type=float, default=0.0)
parser.add_argument("--skip_overlap", default="false")
args = parser.parse_args()

metric = JaccardErrorRate(
    collar=args.collar,
    skip_overlap=parse_bool(args.skip_overlap),
)


def score(payload):
    reference = annotation(payload["reference"])
    hypothesis = annotation(payload["hypothesis"])
    return {"jer": float(metric(reference, hypothesis))}


serve(score)
