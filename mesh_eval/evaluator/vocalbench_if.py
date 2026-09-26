"""Official VocalBench / VocalBench-zh instruction-following judgment criteria.

Source: SJTU-OmniAgent/VocalBench ``utils/instruction_semantic.py`` (and zh twin).
Official metric is Following Rate (FR): semantic Correct, then acoustic gates for
the first 200 paralinguistic rows (see ``instruction_full.py``).
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# Sub-category / Category → English criteria (VocalBench).
EN_CRITERIA_BY_KEY = {
    "Plain repeat": (
        "The model's response must include the complete content that the user "
        "requests to be repeated. Minor word errors and additional explanations are allowed."
    ),
    "With emotion": (
        "The model's response must include the complete content that the user "
        "requests to be repeated. Minor word errors and additional explanations are allowed."
    ),
    "With speed": (
        "The model's response must include the complete content that the user "
        "requests to be repeated. Minor word errors and additional explanations are allowed."
    ),
    "With speed & emotion": (
        "The model's response must include the complete content that the user "
        "requests to be repeated. Minor word errors and additional explanations are allowed."
    ),
    "Num": (
        "The model must strictly list the exact number of items specified by the user. "
        "Providing more or fewer items is considered non-compliant with the instruction."
    ),
    "Constrain": (
        "The model must include the specified content at the designated position and ensure "
        "the required terms appear the mandated number of times. Failure to meet any of these "
        "requirements is considered non-compliant with the instruction."
    ),
    "Keywords": (
        "The model must include all specified keywords and must not include any excluded "
        "keywords. Any deviation is considered non-compliant with the instruction."
    ),
    "Progressive": (
        "The model must answer each question strictly in the order presented. Answering in "
        "the wrong sequence or omitting any question is considered non-compliant with the instruction."
    ),
    "Conditional": (
        "The model must select the appropriate follow-up question based on the judgment from "
        "the preliminary question. Answering the wrong follow-up question or answering both "
        "follow-ups simultaneously is considered non-compliant; however, if the preliminary "
        "judgment itself is incorrect but leads to a logically consistent choice of follow-up, "
        "it is still considered compliant."
    ),
    "Topic change": (
        "The model must follow only the user’s final corrected instruction. Responding to "
        "earlier instructions or addressing both pre- and post-correction instructions is "
        "considered non-compliant."
    ),
    "Spoken format": (
        "The model must respond using purely colloquial language. Inclusion of LaTeX code, "
        "formulas, or other structured text is considered non-compliant with the instruction."
    ),
    "Speaker tune": (
        "The model must use an appropriate and reasonable tone and language style tailored to "
        "the specified audience or its own assigned identity. Inappropriate, untimely, or overly "
        "direct language is considered non-compliant with the instruction."
    ),
    "Speaking style": (
        "The model must strictly mimic the specified character’s style in its reply. Deviating "
        "from the required style or using a flat, plain narrative is considered non-compliant "
        "with the instruction."
    ),
    "Instruction": (
        "The model must strictly adhere to the user-specified wording, structure, and content "
        "sequence. Any deviation or omission is considered non-compliant with the instruction."
    ),
    "Situation": (
        "The model must fully consider and address all specific circumstances described by the "
        "user. Omission or clear contradiction is considered non-compliant with the instruction."
    ),
    "Format": (
        "The model must strictly follow the format specified by the user. Any format deviation "
        "is considered non-compliant with the instruction."
    ),
}

# VocalBench-zh criteria (official Chinese strings).
ZH_CRITERIA_BY_KEY = {
    "Plain repeat": "模型回复必须包含完整用户要求重复的内容。允许有极个别字词错误，和额外的解释。",
    "With emotion": "模型回复必须包含完整用户要求重复的内容。允许有极个别字词错误，和额外的解释。",
    "With speed": "模型回复必须包含完整用户要求重复的内容。允许有极个别字词错误，和额外的解释。",
    "With speed & emotion": "模型回复必须包含完整用户要求重复的内容。允许有极个别字词错误，和额外的解释。",
    "Num": "模型必须严格按用户指定数量进行列举。多于或少于指定数量均视为未遵循指令。",
    "Constrain": "模型在指定位置包含指定内容，且确保要求词汇出现次数达标。任一要求未满足即视为未遵循指令。",
    "Keywords": "模型必须包含所有指定关键词，且不出现任何排除关键词。任一不符即视为未遵循指令。",
    "Progressive": "模型必须严格按问题顺序逐项回答。顺序错误或遗漏任一问题均视为未遵循指令。",
    "Conditional": (
        "要求模型根据先决问题的判断结果选择对应问题作答。选错问题或同时回答两个后续问题视为未遵循指令；"
        "先决问题错误所导致的后续问题选择错误，视为遵循指令。"
    ),
    "Topic change": "模型必须仅遵循用户最终更正后的指令。回应先前指令或回答更正前、后的指令均视为未遵循指令。",
    "Spoken format": "模型必须使用纯口语化文字回答。包含Latex代码、公式等结构化文本，均视为未遵循指令。",
    "Speaker tune": "模型必须根据指定受众身份或自身身份，使用恰当、合理的语言风格和口吻回答。语言不当、不合时宜或直叙均视为未遵循指令。",
    "Speaking style": "模型必须严格模仿指定人物的风格进行回复。风格不符或平铺直叙均视为未遵循指令。",
    "Instruction": "模型必须严格遵循用户指定的用词、结构及内容顺序。任何偏离或忽视均视为未遵循指令。",
    "Situation": "模型必须在回答中充分考虑并回应用户所述的所有具体情况，遗漏或明显冲突视为未遵循指令。",
    "Format": "模型必须严格遵循用户指定的格式，任何格式不符均视为未遵循指令。",
    "Mandarin specific": "模型必须严格使用指定的中文表达方式、文学体裁和参考中国的文化习俗，任何偏离均视为未遵循指令。",
}


def resolve_vocalbench_criteria(
    *,
    category: Any = None,
    subcategory: Any = None,
    language: str = "en",
) -> Optional[str]:
    table = ZH_CRITERIA_BY_KEY if str(language).lower().startswith("zh") else EN_CRITERIA_BY_KEY
    for key in (subcategory, category):
        if key in (None, ""):
            continue
        text = table.get(str(key))
        if text:
            return text
    return None


def vocalbench_official_judge_prompt(
    *,
    question: str,
    response: str,
    criteria: str,
    language: str = "en",
) -> str:
    """Mirror official ``instruction_semantic.py`` judge prompt."""
    if str(language).lower().startswith("zh"):
        return (
            "你是一个客观、严谨的指令遵循评估助手。接下来，我将提供以下三项内容：\n\n"
            "问题：一个中文问题，涉及用户指令。\n"
            "模型回答：模型针对该问题实际生成的回答。\n"
            "判断标准：判断模型是否正确遵循指令的标准。\n\n"
            "返回模型是否正确遵循了用户指令。注意，仅依据判断标准评估模型是否遵循指令，"
            "不因事实错误、回复质量等其他因素影响评估。\n\n"
            "输出格式\n"
            "输出格式：正确/错误 [简单解释]\n"
            "示例：正确 模型正确依据指令重复了用户的语言。\n\n"
            f"问题：{question}\n"
            f"模型回答：{response}\n"
            f"判断标准：{criteria}\n"
        )
    return (
        "You are an objective instruction compliance assessment assistant. I will provide "
        "the following three components:\n\n"
        "Instruction: A user instruction involving explicit requirements.\n"
        "Response: The response generated by a model for this instruction.\n"
        "Judgment Criteria: The criteria used to judge whether the model correctly follows "
        "the instruction.\n\n"
        "Returns whether the model correctly follows the user's instruction. Note that the "
        "assessment is based solely on the judgment criteria and is not influenced by factual "
        "errors, response quality, or other factors.\n\n"
        "Output Format\n"
        "Correct/Incorrect [Simple Explanation]\n"
        "Example: Correct. The model correctly repeated the user's words according to the "
        "instruction.\n\n"
        f"Question: {question}\n"
        f"Model Response: {response}\n"
        f"Judgment Criteria: {criteria}\n"
    )


def parse_vocalbench_correct_incorrect(raw: Any) -> Optional[float]:
    """Map official Correct/Incorrect (or 正确/错误) to 0/1 score."""
    text = str(raw or "").strip()
    if not text:
        return None
    # Prefer an explicit label line; Thinking models may place it after reasoning.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for head in reversed(lines or [text]):
        lower = head.lower()
        if head.startswith("正确") or lower.startswith("correct"):
            return 1.0
        if head.startswith("错误") or lower.startswith("incorrect"):
            return 0.0
    # Fallback: last whole-word match in the full text.
    matches = list(
        re.finditer(r"(?<![A-Za-z])(incorrect|correct)|正确|错误", text, flags=re.I)
    )
    if not matches:
        return None
    token = matches[-1].group(0).lower()
    if token in {"correct", "正确"}:
        return 1.0
    return 0.0


def rubric_mapping_from_item(item: Mapping[str, Any], language: str) -> dict:
    category = item.get("Category")
    subcategory = item.get("Sub-category")
    criteria = resolve_vocalbench_criteria(
        category=category, subcategory=subcategory, language=language
    )
    return {
        "protocol": "vocalbench_instruction_official",
        "instruction": item.get("Question"),
        "category": category,
        "subcategory": subcategory,
        "judgment_criteria": criteria,
        "text_emo": item.get("Text_emo"),
        "metric": "following_rate",
        "notes": (
            "Semantic stage uses official Correct/Incorrect criteria. "
            "Paralinguistic rows (index<200 / Plain repeat…speed&emotion) also need "
            "official acoustic gates (ASR WER, emotion2vec, speaking-speed duration) "
            "for Following Rate; see VocalBench utils/instruction_full.py."
        ),
    }
