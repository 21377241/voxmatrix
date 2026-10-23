"""Load registry models one at a time and run a few fixed requests.

    python scripts/smoke_models.py --models step-audio-2-mini,step-audio-2-mini-speech --out DIR

A model passes when every case returns non-empty text and speech-output cases
produce a readable wav >= 0.5 s.  Expected keywords only flag answers worth
reading (``keyword_miss``); they do not fail the run.
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ASSETS = ROOT / "audio_evals" / "lib" / "StepAudio2" / "assets"
EN_WAV = str(ASSETS / "give_me_a_brief_introduction_to_the_great_wall.wav")
ZH_WAV = str(ASSETS / "multi_turn_round1_zh.wav")


def user(*items):
    contents = []
    for item in items:
        if item.endswith(".wav"):
            contents.append({"type": "audio", "value": item})
        else:
            contents.append({"type": "text", "value": item})
    return [{"role": "user", "contents": contents}]


TEXT_CASES = [
    ("text_qa", user("What is the capital of France? Answer with one word."), ["paris"]),
    ("asr_en", user("Transcribe the speech into text. Output only the transcript.", EN_WAV), ["great wall"]),
    ("asr_zh", user("请把这段语音逐字转写成文字，只输出转写结果。", ZH_WAV), ["古镇"]),
    (
        "asr_zh_system",
        [{"role": "system", "contents": [{"type": "text", "value": "请记录下你所听到的语音内容。"}]}]
        + user(ZH_WAV),
        ["古镇"],
    ),
    ("audio_qa", user(EN_WAV), []),
    (
        "system_prompt",
        [{"role": "system", "contents": [{"type": "text", "value": "You are a helpful assistant. Reply in one sentence."}]}]
        + user("What did the speaker ask for?", EN_WAV),
        [],
    ),
]
SPEECH_CASES = [
    ("tts", user("Please read the following text aloud without adding any other words: Hello, this is a speech synthesis test."), []),
    ("speech_qa", user(EN_WAV), []),
]


def check_audio(path):
    import soundfile as sf

    info = sf.info(path)
    if info.duration < 0.5:
        raise AssertionError(f"generated audio too short: {info.duration:.2f}s")
    return round(info.duration, 2)


def run_model(name, speech):
    from audio_evals.registry import registry

    record = {"model": name, "cases": [], "ok": False}
    started = time.time()
    model = registry.get_model(name)
    if model is None:
        raise RuntimeError(f"model {name!r} not found in registry")
    if hasattr(model, "wait_until_ready"):
        model.wait_until_ready()
    if hasattr(model, "_wait_until_ready"):
        model._wait_until_ready()
    record["load_seconds"] = round(time.time() - started, 1)
    print(f"[smoke] {name} loaded in {record['load_seconds']}s", flush=True)
    try:
        for case, prompt, keywords in SPEECH_CASES if speech else TEXT_CASES:
            entry = {"case": case}
            t0 = time.time()
            try:
                raw = model.inference(prompt)
                entry["seconds"] = round(time.time() - t0, 1)
                if speech:
                    out = json.loads(raw)
                    entry["text"] = out.get("text", "")
                    entry["audio"] = out.get("audio", "")
                    entry["audio_seconds"] = check_audio(entry["audio"])
                else:
                    entry["text"] = str(raw)
                if not entry["text"].strip():
                    raise AssertionError("empty text output")
                missing = [k for k in keywords if k not in entry["text"].lower()]
                if missing:
                    entry["keyword_miss"] = missing
                entry["ok"] = True
            except Exception as exc:
                entry["ok"] = False
                entry["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
            print(f"[smoke] {name} {case}: {json.dumps(entry, ensure_ascii=False)}", flush=True)
            record["cases"].append(entry)
    finally:
        for attr in ("release", "_terminate_isolated_process"):
            fn = getattr(model, attr, None)
            if callable(fn):
                fn()
                break
    record["ok"] = all(c["ok"] for c in record["cases"])
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", required=True, help="comma-separated registry names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        speech = "speech" in name.lower()
        try:
            record = run_model(name, speech)
        except Exception as exc:
            traceback.print_exc()
            record = {"model": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(record)
        (out_dir / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    print("[smoke] summary:", flush=True)
    for r in results:
        print(f"[smoke]   {r['model']}: {'PASS' if r['ok'] else 'FAIL'} {r.get('error', '')}", flush=True)
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
