# From https://github.com/MatthewCYM/VoiceBench/blob/main/api_judge.py

import json
import os
import re
from typing import Dict

import numpy as np

from audio_evals.evaluator.base import Evaluator


DEFAULT_OPENAI_JUDGE_MODEL = "gpt4o-mini"
DEFAULT_LLMCENTER_JUDGE_MODEL = "mb-gpt4o-mini"


def resolve_judge_model_name(configured_name: str) -> str:
    override = os.environ.get("MESH_JUDGE_MODEL", "").strip()
    if override:
        return override
    if (
        configured_name == DEFAULT_OPENAI_JUDGE_MODEL
        and not os.environ.get("OPENAI_API_KEY")
        and os.environ.get("LLMCenterUserToken")
    ):
        return DEFAULT_LLMCENTER_JUDGE_MODEL
    return configured_name


def get_judge_model(configured_name: str):
    from audio_evals.registry import registry

    model_name = resolve_judge_model_name(configured_name)
    model = registry.get_model(model_name)
    if model is None:
        raise KeyError(f"Judge model is not registered: {model_name}")
    return model


meta_prompt_qa = """
### Question
{question}

### Reference answer
{reference}

### Candidate answer
{response}

Is the candidate answer correct based on the question and reference answer?
Please only output a single "Yes" or "No". Do not output anything else.
"""


def majority_vote(scores):
    scores = [item.lower() for item in scores]
    final_answer = max(set(scores), key=scores.count)

    # Convert the final answer to True for 'Yes' and False for 'No'
    return 1 if final_answer == "yes" else 0


class QaReferenceEvaluator(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = get_judge_model(self.model_name)
        p = meta_prompt_qa.format(
            question=kwargs["question"],
            reference=label,
            response=pred,
        )
        prompt = registry.get_prompt("yes_no_judge")

        res = [
            model.inference(
                prompt.load(real_prompt=p),
                max_tokens=1024,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=0.5,
                top_p=0.95,
            )
            for _ in range(3)
        ]

        return {
            "gpt_score": majority_vote(res),
            "pred": pred,
            "ref": label,
        }


class QA_YES_NO_PEDANT(Evaluator):
    def __init__(self):
        self.pedant = PEDANT()

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        panda_results = self.pedant.evaluate(
            [label.lower()], pred.lower(), kwargs["prompt"].lower()
        )
        return {
            "panda_score": panda_results,
            "pred": pred,
            "ref": label,
        }


meta_prompt_open = """
I need your help to evaluate the performance of several models in the speech interaction scenario. The models will receive a speech input from the user, which they need to understand and respond to with a speech output.
Your task is to rate the model’s responses based on the provided user input transcription [Instruction] and the model’s output transcription [Response].

Please evaluate the response on a scale of 1 to 5:
1 point: The response is largely irrelevant, incorrect, or fails to address the user’s query. It may be off-topic or provide incorrect information.
2 points: The response is somewhat relevant but lacks accuracy or completeness. It may only partially answer the user’s question or include extraneous information.
3 points: The response is relevant and mostly accurate, but it may lack conciseness or include unnecessary details that don’t contribute to the main point.
4 points: The response is relevant, accurate, and concise, providing a clear answer to the user’s question without unnecessary elaboration.
5 points: The response is exceptionally relevant, accurate, and to the point. It directly addresses the user’s query in a highly effective and efficient manner, providing exactly the information needed.

Below are the transcription of user’s instruction and models’ response:
### [Instruction]: {question}
### [Response]: {response}

After evaluating, please output the score only without anything else.
You don’t need to provide any explanations.
"""


def extract_rating(llm_output):
    """
    Extracts the rating in the format [[number]] from the LLM output.

    Args:
    - llm_output (str): The response from the LLM containing the evaluation and rating.

    Returns:
    - int: The extracted rating, or None if the rating is not found.
    """
    # Define the regular expression pattern to match the rating in the format [[number]]
    pattern = r"\[\[(\d+)\]\]"

    # Search for the pattern in the LLM output
    match = re.search(pattern, llm_output)

    if match:
        # Convert the matched rating to an integer and return it
        return int(match.group(1))
    else:
        # Return None if no rating is found
        # return None
        return None


class VoiceBenchQaOpenEvaluator(Evaluator):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        model = get_judge_model(self.model_name)
        p = meta_prompt_open.format(
            question=kwargs["question"],
            response=pred,
        )
        prompt = registry.get_prompt("yes_no_judge")

        res = [
            model.inference(
                prompt.load(real_prompt=p),
                max_tokens=1024,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                temperature=0.5,
                top_p=0.95,
            )
            for _ in range(3)
        ]

        scores = []
        for item in res:
            try:
                score = float(item)
            except (TypeError, ValueError):
                score = extract_rating(str(item))
            if score is None:
                raise ValueError(f"VoiceBench judge returned an invalid rating: {item!r}")
            scores.append(score)

        return {
            "gpt_score": sum(scores) / len(scores),
            "pred": pred,
            "ref": label,
        }


class VoiceBenchMultiTurnEvaluator(Evaluator):
    """Judge every assistant turn with the preceding conversation as context."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @staticmethod
    def _responses(pred):
        if isinstance(pred, dict):
            value = pred
        else:
            try:
                value = json.loads(str(pred))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError("multi-turn prediction must be JSON") from exc
        responses = value.get("responses") if isinstance(value, dict) else None
        if not isinstance(responses, list) or not responses:
            raise ValueError("multi-turn prediction has no responses")
        return [str(item) for item in responses]

    def _eval(self, pred, label, **kwargs) -> Dict[str, any]:
        from audio_evals.registry import registry

        turns = kwargs.get("turns") or []
        responses = self._responses(pred)
        if len(turns) != len(responses):
            raise ValueError(
                f"turn/response count mismatch: {len(turns)} != {len(responses)}"
            )

        model = get_judge_model(self.model_name)
        judge_prompt = registry.get_prompt("chatbot-eval")
        history = []
        scores = []
        for index, (instruction, response) in enumerate(
            zip(turns, responses), start=1
        ):
            history.append(f"User turn {index}: {instruction}")
            contextual_instruction = "\n".join(history)
            result = model.inference(
                judge_prompt.load(
                    instruction=contextual_instruction,
                    response=response,
                ),
                temperature=0,
                max_tokens=512,
            )
            match = re.search(r"\[\[(\d+)\]\]", str(result))
            if match is None:
                raise ValueError(f"MT-Bench judge returned an invalid score: {result!r}")
            scores.append(int(match.group(1)))
            history.append(f"Assistant turn {index}: {response}")

        return {
            "gpt_score": sum(scores) / len(scores),
            "turn_scores": "|".join(str(score) for score in scores),
            "pred": pred,
            "ref": label,
        }
