from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
LAST_NUMBER_RE = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
STRIP_WRAPPERS_RE = re.compile(r"^\$+|\$+$")


def build_prompt(question: str) -> str:
    return (
        "You are solving an AIME 2025 mathematics problem.\n"
        "Reason carefully, then end with a single final answer in the format "
        "\\boxed{answer}.\n\n"
        f"Problem:\n{question}\n"
    )


def extract_final_answer(text: str) -> str:
    matches = BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()

    numbers = LAST_NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()

    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def normalize_answer(value: str) -> str:
    text = STRIP_WRAPPERS_RE.sub("", value.strip())
    text = text.replace("\\!", "").replace(" ", "")
    text = text.replace(",", "")
    if text.startswith("\\boxed{") and text.endswith("}"):
        text = text[7:-1]
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return text.lower()
    normalized = format(decimal_value.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def compute_binary_reward(prediction: str, answer: str) -> int:
    pred_final = normalize_answer(extract_final_answer(prediction))
    gold_final = normalize_answer(answer)
    return int(pred_final == gold_final)
