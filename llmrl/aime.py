from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation


BOXED_START_RE = re.compile(r"\\boxed\{")
FINAL_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|=|:)\s*(.+)$",
    flags=re.IGNORECASE | re.MULTILINE,
)
LAST_NUMBER_RE = re.compile(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)")
STRIP_WRAPPERS_RE = re.compile(r"^\$+|\$+$")
FRACTION_RE = re.compile(r"\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}")
LATEX_NOISE_RE = re.compile(r"\\(?:left|right|!|,|;)")
TRAILING_PUNCT_RE = re.compile(r"[\s\.\,\:\;\!\?]+$")


def build_prompt(question: str) -> str:
    return (
        "Solve the following AIME 2025 mathematics problem.\n"
        "You may think step by step, but the final line must be exactly:\n"
        "Final Answer: \\boxed{answer}\n"
        "Use only one boxed final answer.\n\n"
        f"Problem:\n{question}\n"
    )


def _extract_last_boxed(text: str) -> str:
    matches: list[str] = []
    for match in BOXED_START_RE.finditer(text):
        idx = match.end()
        depth = 1
        start = idx
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            matches.append(text[start : idx - 1].strip())
    return matches[-1] if matches else ""


def extract_final_answer(text: str) -> str:
    boxed = _extract_last_boxed(text)
    if boxed:
        return boxed

    final_matches = FINAL_ANSWER_RE.findall(text)
    if final_matches:
        candidate = final_matches[-1].strip()
        boxed_candidate = _extract_last_boxed(candidate)
        return boxed_candidate or candidate

    numbers = LAST_NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].strip()

    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def normalize_answer(value: str) -> str:
    text = STRIP_WRAPPERS_RE.sub("", value.strip())
    text = LATEX_NOISE_RE.sub("", text)
    text = text.replace(" ", "")
    boxed = _extract_last_boxed(text)
    if boxed:
        text = boxed
    final_matches = FINAL_ANSWER_RE.findall(text)
    if final_matches:
        text = final_matches[-1]
    text = text.replace(",", "")
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    text = TRAILING_PUNCT_RE.sub("", text)
    fraction_match = FRACTION_RE.fullmatch(text)
    if fraction_match:
        numerator = normalize_answer(fraction_match.group(1))
        denominator = normalize_answer(fraction_match.group(2))
        try:
            decimal_value = Decimal(numerator) / Decimal(denominator)
        except (InvalidOperation, ZeroDivisionError):
            return f"{numerator}/{denominator}".lower()
        normalized = format(decimal_value.normalize(), "f")
        return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized
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
