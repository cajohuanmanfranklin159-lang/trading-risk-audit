# trading-risk-audit

Single-turn environment that tests whether a model can do the arithmetic and the
judgment a trading risk manager applies every day. Built on `verifiers` v1.

## Task families

| Family | What the model must do | Answer |
| --- | --- | --- |
| `position_size` | Size a long or short trade so a stop-out loses no more than a fixed % of equity | Exact integer |
| `max_drawdown` | Compute the peak-to-trough drawdown of an equity curve | Percent, ±0.01 |
| `expectancy` | Compute expected profit per trade from win rate and average win/loss | Dollars, ±0.01 |
| `lookahead` | Find the one line of a strategy that uses data not yet available | Line number |

Tasks are generated from a seed, so a given seed always produces the same tasks.
Gold answers are computed in Python and checked a second time in the tests by
recomputing them only from the numbers shown in the prompt.

## Scoring

The reward is 1.0 or 0.0, from the final reply only:

- exactly one `<answer>` tag is required; none or several score 0, so the model
  cannot hedge by listing several answers;
- the tag must hold one plain number (`$`, `%` and thousands commas are tolerated);
- integer answers must match exactly; 2-decimal answers must be within 0.01.

## Config

| Field | Default | Meaning |
| --- | --- | --- |
| `per_family` | 50 | Tasks generated per family |
| `seed` | 0 | Generator seed |
| `families` | all four | Subset of families to include |

## Checks

```bash
pip install -e ".[test]"
pytest -q                                   # 130 tests
vf-validate trading-risk-audit --runtime.type subprocess
```

`vf-validate` result on the default config: 200 of 200 tasks valid on both the
gold check (the reference answer scores 1.0) and the noop check (an empty reply
scores 0).

## Roadmap

Multi-turn versions where the model gets tools to load price data, run a
backtest and audit it; tasks drawn from real backtest bugs (survivorship bias,
fill assumptions, fee omission); portfolio-level risk (correlation, leverage
limits, margin calls).
