"""Deterministic task generator for trading-risk-audit.

Every task is built from a seeded random generator, so the same seed always
produces the same tasks and the same gold answers. Gold answers are computed
here in plain Python; the grader never trusts the model's arithmetic.

Four task families:
  - position_size: how many units fit a fixed-risk budget
  - max_drawdown:  peak-to-trough drawdown of an equity curve, in percent
  - expectancy:    expected profit per trade from win rate and average win/loss
  - lookahead:     which line of a strategy uses data not yet available
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

FAMILIES = ("position_size", "max_drawdown", "expectancy", "lookahead")

ANSWER_FORMAT = (
    "Give your final answer inside a single <answer></answer> tag, "
    "with no units or extra text inside the tag."
)


@dataclass(frozen=True)
class GeneratedTask:
    id: str
    family: str
    prompt: str
    answer: str
    kind: str  # "int" | "float2"


def _position_size(rng: random.Random, i: int) -> GeneratedTask:
    equity = rng.randrange(10_000, 200_001, 500)
    risk_pct = rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    entry = round(rng.uniform(20, 45_000), 2)
    stop_dist = round(entry * rng.uniform(0.002, 0.03), 2)
    stop_dist = max(stop_dist, 0.05)
    side = rng.choice(["long", "short"])
    stop = round(entry - stop_dist if side == "long" else entry + stop_dist, 2)
    point_value = rng.choice([1, 2, 5, 10, 20, 50])
    risk_usd = equity * risk_pct / 100
    per_unit_loss = abs(entry - stop) * point_value
    units = math.floor(risk_usd / per_unit_loss + 1e-9)
    if units < 1:
        return _position_size(rng, i)
    prompt = (
        f"An account has equity of ${equity:,}. The trader risks {risk_pct}% of equity "
        f"per trade. They plan a {side} entry at {entry:.2f} with a stop loss at "
        f"{stop:.2f}. Each unit of the instrument gains or loses ${point_value} for "
        "every 1.00 move in price. Ignoring fees and slippage, what is the maximum "
        "whole number of units they can trade without exceeding the risk budget if "
        f"the stop is hit?\n\n{ANSWER_FORMAT}"
    )
    return GeneratedTask(f"position_size-{i}", "position_size", prompt, str(units), "int")


def _max_drawdown(rng: random.Random, i: int) -> GeneratedTask:
    n = rng.randint(12, 20)
    equity = [float(rng.randrange(8_000, 15_001, 50))]
    for _ in range(n - 1):
        equity.append(round(max(500.0, equity[-1] * (1 + rng.gauss(0.002, 0.04))), 2))
    peak = equity[0]
    mdd = 0.0
    for value in equity:
        peak = max(peak, value)
        mdd = max(mdd, (peak - value) / peak)
    mdd_pct = round(mdd * 100, 2)
    curve = ", ".join(f"{v:.2f}" for v in equity)
    prompt = (
        "Below is an account's end-of-day equity, in order:\n"
        f"{curve}\n\n"
        "What is the maximum drawdown, defined as the largest peak-to-trough decline "
        "measured from the highest prior equity value, expressed as a percentage of "
        "that peak? Round to 2 decimal places (for example 12.34). If equity never "
        f"falls below a prior peak, the answer is 0.00.\n\n{ANSWER_FORMAT}"
    )
    return GeneratedTask(f"max_drawdown-{i}", "max_drawdown", prompt, f"{mdd_pct:.2f}", "float2")


def _expectancy(rng: random.Random, i: int) -> GeneratedTask:
    trades = rng.choice([50, 80, 100, 120, 200, 250])
    wins = rng.randint(int(trades * 0.25), int(trades * 0.7))
    avg_win = round(rng.uniform(80, 900), 2)
    avg_loss = round(rng.uniform(60, 700), 2)
    p = wins / trades
    expectancy = round(p * avg_win - (1 - p) * avg_loss, 2)
    prompt = (
        f"A strategy's backtest has {trades} trades: {wins} winners and "
        f"{trades - wins} losers. The average winning trade made ${avg_win:.2f} and "
        f"the average losing trade lost ${avg_loss:.2f}. What is the expectancy, the "
        "average profit or loss per trade in dollars? Use a minus sign if it is "
        f"negative and round to 2 decimal places.\n\n{ANSWER_FORMAT}"
    )
    return GeneratedTask(f"expectancy-{i}", "expectancy", prompt, f"{expectancy:.2f}", "float2")


def _lookahead(rng: random.Random, i: int) -> GeneratedTask:
    period = rng.randint(10, 250)
    mult = rng.choice([1.0, 1.5, 2.0, 2.5, 3.0])
    lo, hi = period - 1, 1
    # (correct line, biased line). The decision is made after bar t closes and the
    # order fills at the next bar's open, so any value from bar t+1 or later used
    # in the decision, or a fill before bar t has closed, is look-ahead bias.
    lines = [
        (f"sma = mean(close[t-{lo} : t+{hi}])", f"sma = mean(close[t-{lo - 1} : t+{hi + 1}])"),
        ("long_signal = close[t] > sma", "long_signal = close[t+1] > sma"),
        (f"stop = low[t] - {mult} * atr[t]", f"stop = low[t+1] - {mult} * atr[t]"),
        (
            "units = floor(risk_usd / (close[t] - stop))",
            "units = floor(risk_usd / (high[t+1] - stop))",
        ),
        ("entry_price = open[t+1]", "entry_price = open[t]"),
    ]
    bad = rng.randrange(len(lines))
    code = "\n".join(
        f"{n + 1}: {biased if n == bad else ok}" for n, (ok, biased) in enumerate(lines)
    )
    prompt = (
        "A daily strategy is evaluated bar by bar. For each bar t, the decision is "
        "made after bar t has closed, and any order fills at the open of bar t+1. "
        "Slices follow Python rules (close[a : b] includes a and excludes b). "
        "Exactly one numbered line below introduces look-ahead bias, meaning it uses "
        "information that would not be available at the moment the decision or fill "
        f"happens:\n\n{code}\n\n"
        "Which line number introduces the look-ahead bias?\n\n"
        f"{ANSWER_FORMAT}"
    )
    return GeneratedTask(f"lookahead-{i}", "lookahead", prompt, str(bad + 1), "int")


_BUILDERS = {
    "position_size": _position_size,
    "max_drawdown": _max_drawdown,
    "expectancy": _expectancy,
    "lookahead": _lookahead,
}


def generate(per_family: int = 50, seed: int = 0, families: tuple[str, ...] = FAMILIES):
    """Return `per_family` tasks for each family, deterministic for a given seed."""
    tasks: list[GeneratedTask] = []
    seen: set[str] = set()
    for family in families:
        rng = random.Random(f"{seed}-{family}")
        for i in range(per_family):
            # Redraw on the rare duplicate so every prompt in a taskset is unique.
            for _ in range(100):
                task = _BUILDERS[family](rng, i)
                if task.prompt not in seen:
                    break
            else:
                raise RuntimeError(f"could not draw a unique {family} task")
            seen.add(task.prompt)
            tasks.append(task)
    return tasks
