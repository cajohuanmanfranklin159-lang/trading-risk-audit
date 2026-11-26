"""trading-risk-audit: risk-management and backtest-audit tasks for traders (single-turn).

The model answers quantitative trading questions a professional risk manager
checks every day: position sizing against a fixed risk budget, maximum drawdown
of an equity curve, strategy expectancy, and spotting look-ahead bias in
backtest code. Tasks are procedurally generated from a seed, gold answers are
computed in Python, and the reward is a pure function of the final reply.
"""

from __future__ import annotations

import verifiers.v1 as vf

from trading_risk_audit.generator import FAMILIES, generate
from trading_risk_audit.grader import score

SYSTEM = (
    "You are a quantitative risk analyst. Work through the problem carefully, then "
    "give exactly one final answer inside <answer></answer> tags."
)


class TradingRiskData(vf.TaskData):
    answer: str
    """Gold answer, computed by the generator."""
    kind: str
    """'int' for exact integer answers, 'float2' for 2-decimal answers (±0.01)."""
    family: str
    """Task family: position_size, max_drawdown, expectancy or lookahead."""


class TradingRiskTask(vf.Task[TradingRiskData]):
    @vf.stop
    async def single_turn(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= 1

    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace) -> float:
        return score(trace.last_reply, self.data.answer, self.data.kind)

    async def validate(self, runtime: vf.Runtime) -> bool:
        """The gold answer, formatted as the model is asked to, must score 1.0."""
        return score(f"<answer>{self.data.answer}</answer>", self.data.answer, self.data.kind) == 1.0


class TradingRiskConfig(vf.TasksetConfig):
    per_family: int = 50
    """Tasks generated per family."""
    seed: int = 0
    """Generator seed; the same seed always yields the same tasks."""
    families: list[str] = list(FAMILIES)
    """Subset of families to include."""


class TradingRiskTaskset(vf.Taskset[TradingRiskTask, TradingRiskConfig]):
    def load(self) -> list[TradingRiskTask]:
        unknown = set(self.config.families) - set(FAMILIES)
        if unknown:
            raise ValueError(f"unknown families: {sorted(unknown)}")
        return [
            TradingRiskTask(
                TradingRiskData(
                    id=t.id,
                    prompt=t.prompt,
                    system_prompt=SYSTEM,
                    answer=t.answer,
                    kind=t.kind,
                    family=t.family,
                ),
                self.config.task,
            )
            for t in generate(
                per_family=self.config.per_family,
                seed=self.config.seed,
                families=tuple(self.config.families),
            )
        ]
