"""Tests for trading-risk-audit: gold answers, independent recomputation, anti-gaming."""

import asyncio
import math
import re

import pytest

from trading_risk_audit.generator import FAMILIES, generate
from trading_risk_audit.grader import score
from trading_risk_audit.taskset import TradingRiskConfig, TradingRiskTaskset

TASKS = generate(per_family=200, seed=0)


def test_deterministic():
    a = [(t.id, t.prompt, t.answer) for t in generate(per_family=20, seed=7)]
    b = [(t.id, t.prompt, t.answer) for t in generate(per_family=20, seed=7)]
    assert a == b
    c = [(t.id, t.prompt, t.answer) for t in generate(per_family=20, seed=8)]
    assert a != c


def test_unique_ids_and_prompts():
    assert len({t.id for t in TASKS}) == len(TASKS)
    assert len({t.prompt for t in TASKS}) == len(TASKS)


@pytest.mark.parametrize("task", TASKS[::7], ids=lambda t: t.id)
def test_gold_scores_one(task):
    assert score(f"Reasoning...\n<answer>{task.answer}</answer>", task.answer, task.kind) == 1.0


# --- Independent recomputation from the numbers shown in the prompt ---------------

def _nums(pattern, text):
    found = re.search(pattern, text)
    return [float(g.replace(",", "")) for g in found.groups()]


def test_position_size_recomputed_from_prompt():
    for t in (t for t in TASKS if t.family == "position_size"):
        equity = _nums(r"equity of \$([\d,]+)", t.prompt)[0]
        risk = _nums(r"risks ([\d.]+)% of equity", t.prompt)[0]
        entry = _nums(r"entry at (\d+\.\d{2})", t.prompt)[0]
        stop = _nums(r"stop loss at (\d+\.\d{2})", t.prompt)[0]
        pv = _nums(r"loses \$(\d+) for", t.prompt)[0]
        units = math.floor(equity * risk / 100 / (abs(entry - stop) * pv) + 1e-9)
        assert str(units) == t.answer, t.id
        assert units >= 1


def test_max_drawdown_recomputed_from_prompt():
    for t in (t for t in TASKS if t.family == "max_drawdown"):
        curve_line = t.prompt.split("\n")[1]
        curve = [float(x) for x in curve_line.split(", ")]
        worst = 0.0
        for j, v in enumerate(curve):
            worst = max(worst, (max(curve[: j + 1]) - v) / max(curve[: j + 1]))
        assert abs(round(worst * 100, 2) - float(t.answer)) < 1e-9, t.id


def test_expectancy_recomputed_from_prompt():
    for t in (t for t in TASKS if t.family == "expectancy"):
        trades, wins = _nums(r"has (\d+) trades: (\d+) winners", t.prompt)
        win = _nums(r"made \$(\d+\.\d{2})", t.prompt)[0]
        loss = _nums(r"lost \$(\d+\.\d{2})", t.prompt)[0]
        p = wins / trades
        assert abs(round(p * win - (1 - p) * loss, 2) - float(t.answer)) < 1e-9, t.id


def test_lookahead_exactly_one_biased_line():
    future = re.compile(r"\[t\+1\]|t\+2\]|open\[t\]")
    for t in (t for t in TASKS if t.family == "lookahead"):
        code = [ln for ln in t.prompt.split("\n") if re.match(r"^\d: ", ln)]
        assert len(code) == 5
        flagged = []
        for ln in code:
            num, body = ln.split(": ", 1)
            # A correct line may mention t+1 only as the slice end of the SMA
            # (excluded by Python rules) or as the next-bar fill open[t+1].
            body_check = body.replace("t+1])", "").replace("open[t+1]", "")
            if future.search(body_check):
                flagged.append(num)
        assert flagged == [t.answer], (t.id, flagged)


def test_lookahead_answers_cover_every_line():
    answers = {t.answer for t in TASKS if t.family == "lookahead"}
    assert answers == {"1", "2", "3", "4", "5"}


# --- Anti-gaming ---------------------------------------------------------------------

def test_no_tag_scores_zero():
    assert score("42", "42", "int") == 0.0
    assert score("", "42", "int") == 0.0
    assert score(None, "42", "int") == 0.0


def test_multiple_tags_score_zero():
    assert score("<answer>1</answer><answer>2</answer>", "1", "int") == 0.0


def test_hedged_content_scores_zero():
    assert score("<answer>1 or 2</answer>", "1", "int") == 0.0
    assert score("<answer>between 3 and 5</answer>", "4", "int") == 0.0
    assert score("<answer>1, 2, 3, 4, 5</answer>", "3", "int") == 0.0


def test_tolerated_formatting():
    assert score("<answer>$1,234.50</answer>", "1234.50", "float2") == 1.0
    assert score("<answer>12.34%</answer>", "12.34", "float2") == 1.0
    assert score("<answer>-$15.20</answer>", "-15.20", "float2") == 1.0


def test_float_tolerance_boundaries():
    assert score("<answer>12.35</answer>", "12.34", "float2") == 1.0
    assert score("<answer>12.36</answer>", "12.34", "float2") == 0.0
    assert score("<answer>-12.34</answer>", "12.34", "float2") == 0.0


def test_integer_must_be_exact():
    assert score("<answer>17</answer>", "17", "int") == 1.0
    assert score("<answer>17.0</answer>", "17", "int") == 1.0
    assert score("<answer>18</answer>", "17", "int") == 0.0


# --- verifiers integration ----------------------------------------------------------

def test_taskset_loads_and_validates():
    taskset = TradingRiskTaskset(TradingRiskConfig(per_family=5))
    tasks = list(taskset)
    assert len(tasks) == 5 * len(FAMILIES)
    assert [t.data.idx for t in tasks] == list(range(len(tasks)))
    for task in tasks:
        assert asyncio.run(task.validate(None)) is True


def test_unknown_family_rejected():
    with pytest.raises(ValueError):
        list(TradingRiskTaskset(TradingRiskConfig(families=["nope"])))
