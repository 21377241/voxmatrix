#!/usr/bin/env python3
"""Smoke semantic LLM judge against ChatAnywhere / OpenAI-compatible endpoint.

Requires:
  OPENAI_API_KEY
  OPENAI_BASE_URL  (e.g. https://api.chatanywhere.tech/v1)

Does not print secrets. Writes a small JSON summary next to this script's CWD.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


CASES = [
    {
        "name": "qa_open_good",
        "capability": "qa",
        "rubric": "open",
        "question": "How do I wrap a present neatly?",
        "pred": "Measure the gift, cut enough paper, fold edges cleanly, and tape the seams.",
        "ref": "",
    },
    {
        "name": "qa_open_bad",
        "capability": "qa",
        "rubric": "open",
        "question": "How do I wrap a present neatly?",
        "pred": "The capital of France is Paris.",
        "ref": "",
    },
    {
        "name": "qa_binary_yes",
        "capability": "qa",
        "rubric": "binary",
        "question": "What is 2+2?",
        "pred": "The answer is four.",
        "ref": "4",
    },
    {
        "name": "ar_grounded",
        "capability": "audio_reasoning",
        "rubric": "semi-open",
        "question": "How many knocks were there?",
        "audio_transcript": "Someone knocked on the door three times, then paused.",
        "pred": "There were three knocks.",
        "ref": "3",
    },
    {
        "name": "ar_ungrounded",
        "capability": "audio_reasoning",
        "rubric": "open",
        "question": "How many knocks were there?",
        "audio_transcript": "Someone knocked on the door three times, then paused.",
        "pred": "Knocking is a common social custom around the world.",
        "ref": "",
    },
    {
        "name": "st_good",
        "capability": "speech_translation",
        "rubric": "translation",
        "direction": "Translate into English",
        "audio_transcript": "今天天气很好，我们去公园吧。",
        "pred": "The weather is nice today; let's go to the park.",
        "ref": "The weather is great today. Let's go to the park.",
    },
    {
        "name": "st_bad",
        "capability": "speech_translation",
        "rubric": "translation",
        "direction": "Translate into English",
        "audio_transcript": "今天天气很好，我们去公园吧。",
        "pred": "I like pizza and basketball.",
        "ref": "The weather is great today. Let's go to the park.",
    },
    {
        "name": "cs_yes",
        "capability": "code_switch",
        "rubric": "binary",
        "question": "这个 cup 多少钱?",
        "audio_transcript": "这个 cup 多少钱",
        "pred": "二十块",
        "ref": "20 yuan",
    },
    {
        "name": "cs_no",
        "capability": "code_switch",
        "rubric": "binary",
        "question": "这个 cup 多少钱?",
        "audio_transcript": "这个 cup 多少钱",
        "pred": "It is made of glass.",
        "ref": "20 yuan",
    },
]


def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        print("FAIL: OPENAI_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("OPENAI_BASE_URL"):
        print(
            "WARN: OPENAI_BASE_URL unset; default OpenAI host will be used",
            file=sys.stderr,
        )

    from audio_evals.evaluator.semantic_llm_judge import SemanticLLMJudgeEvaluator

    judge = SemanticLLMJudgeEvaluator(n_samples=1)
    rows = []
    for case in CASES:
        out = judge._eval(
            case["pred"],
            case.get("ref", ""),
            capability=case["capability"],
            rubric=case.get("rubric"),
            question=case.get("question", ""),
            audio_transcript=case.get("audio_transcript", ""),
            direction=case.get("direction", ""),
            judge_template="text",
        )
        row = {
            "name": case["name"],
            "capability": case["capability"],
            "rubric": case.get("rubric"),
            "prompt_id": out.get("prompt_id"),
            "gpt_score": out.get("gpt_score"),
            "score_0_100": out.get("score_0_100"),
            "match": out.get("match"),
            "skipped": out.get("skipped"),
            "skip_reason": out.get("skip_reason"),
            "raw": out.get("raw_judge_output"),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    # Sanity: good cases should outscore bad counterparts when both scored.
    def score_of(name: str):
        for r in rows:
            if r["name"] == name and not r.get("skipped"):
                return r.get("score_0_100")
        return None

    checks = []
    for good, bad in (
        ("qa_open_good", "qa_open_bad"),
        ("ar_grounded", "ar_ungrounded"),
        ("st_good", "st_bad"),
    ):
        g, b = score_of(good), score_of(bad)
        ok = g is not None and b is not None and g > b
        checks.append({"pair": [good, bad], "good": g, "bad": b, "ok": ok})

    cs_yes = next(r for r in rows if r["name"] == "cs_yes")
    cs_no = next(r for r in rows if r["name"] == "cs_no")
    checks.append(
        {
            "pair": ["cs_yes", "cs_no"],
            "good": cs_yes.get("score_0_100"),
            "bad": cs_no.get("score_0_100"),
            "ok": (not cs_yes.get("skipped") and not cs_no.get("skipped")
                  and cs_yes.get("match") == 1 and cs_no.get("match") == 0),
        }
    )

    summary = {
        "n": len(rows),
        "n_skipped": sum(1 for r in rows if r.get("skipped")),
        "checks": checks,
        "all_checks_ok": all(c["ok"] for c in checks),
        "rows": rows,
    }
    out_path = Path("/tmp/semantic_llm_judge_smoke.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps({k: summary[k] for k in ("n", "n_skipped", "all_checks_ok", "checks")}, ensure_ascii=False))
    print("WROTE", out_path)
    return 0 if summary["all_checks_ok"] and summary["n_skipped"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
