"""Pure scoring for trading-risk-audit, independent of the verifiers runtime.

Anti-gaming rules:
  - Exactly one <answer> tag must be present; zero or several tags score 0,
    so a model cannot hedge by listing several candidate answers.
  - The tag must contain a single plain number (commas, a leading $ and a
    trailing % are tolerated); anything else scores 0.
  - Integer answers must match exactly; 2-decimal answers must be within 0.01.
"""

from __future__ import annotations

import re

_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_answer(reply: str | None) -> str | None:
    matches = _TAG.findall(reply or "")
    if len(matches) != 1:
        return None
    return matches[0].strip()


def parse_number(text: str) -> float | None:
    cleaned = text.strip().replace(",", "")
    if cleaned.startswith("$"):
        cleaned = cleaned[1:]
    elif cleaned.startswith("-$"):
        cleaned = "-" + cleaned[2:]
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    cleaned = cleaned.strip()
    if not _NUMBER.fullmatch(cleaned):
        return None
    return float(cleaned)


def score(reply: str | None, gold: str, kind: str) -> float:
    raw = extract_answer(reply)
    if raw is None:
        return 0.0
    value = parse_number(raw)
    if value is None:
        return 0.0
    target = float(gold)
    if kind == "int":
        return 1.0 if value == target else 0.0
    if kind == "float2":
        return 1.0 if abs(value - target) <= 0.01 + 1e-9 else 0.0
    raise ValueError(f"unknown answer kind: {kind}")
