# Monte Carlo Risk Framework for Quantitative Trading Strategies

A model-agnostic, **evaluation-first** framework for tail-risk forecasting. It
compares Monte Carlo / simulation-based VaR and CVaR methods under rigorous
out-of-sample backtesting. The contribution is the *evaluation design*, not a
new model — the framework's job is to say, defensibly, **which method produces
better-calibrated tail risk on a given market**.

## Research question

> Do regime-aware Monte Carlo approaches produce better-calibrated VaR and CVaR
> forecasts than bootstrap-based and filtered historical simulation methods,
> across developed and emerging equity markets?

Falsifiable by design: the HMM regime model can win, tie, or lose. The literature
prior — and the synthetic sanity checks here — is that **GARCH-filtered FHS is
the strong all-rounder**, with **regime conditioning helping mainly in the deep
(1%) tail or in genuinely regime-heavy markets**. A null result is a real result.

## What makes it research-grade (not a calculator)

A VaR forecast is a falsifiable claim, so every method is scored, not just
computed:

| Layer | Method | Tests |
|---|---|---|
| VaR coverage | Kupiec POF (unconditional) | breach rate == alpha? |
| VaR clustering | Christoffersen independence + conditional coverage | breaches serially independent? |
| **ES (CVaR) calibration** | McNeil–Frey exceedance-residual bootstrap | is expected shortfall the right size? |
| **Model ranking** | Pinball (quantile) loss + Diebold–Mariano | is method A *significantly* better than B? |

The ES test matters: Kupiec/Christoffersen test the VaR **quantile only** and say
nothing about CVaR. Reporting CVaR without McNeil–Frey (or Acerbi–Székely) is the
gap reviewers flag. Diebold–Mariano on pinball loss is what turns "all five
passed, FHS looks nicest" into "FHS has significantly lower quantile loss
(DM p=0.03)".

## Engines (one-step-ahead conditional VaR/CVaR)

- `gaussian` — normal fit. Thin-tailed baseline (expected to fail on real data).
- `historical` — empirical quantile. No distributional assumption.
- `student_t` — fat-tailed parametric, with closed-form ES.
- `fhs_ewma` / `fhs_garch` — Filtered Historical Simulation: EWMA (RiskMetrics)
  or GARCH(1,1) vol filter + resampled standardised residuals. Usually the
  strongest baseline.
- `hmm_regime` — Gaussian-HMM regime mixture propagated one step through the
  transition matrix. Reuses regime-detection work.

## Install

```bash
pip install -r requirements.txt   # includes arch; if missing, the framework falls back to EWMA
```

## Quickstart

```bash
# Real NSE/BSE data
python scripts/run_comparison.py RELIANCE --exchange NSE --window 500 --plot
python scripts/run_comparison.py INFY --garch          # GARCH-filtered FHS

# Reproducible synthetic study (no network)
python scripts/run_comparison.py --synthetic regime --plot
python notebooks/case_study.py
```

For the full cross-market sweep, use the batch driver. It handles raw index
symbols like `^GSPC` and `^NSEI` directly, so no exchange suffix is forced on
them.

```bash
# Fast first pass across all markets
python scripts/run_batch.py --quick

# Full walk-forward sweep with GARCH-filtered FHS
python scripts/run_batch.py --garch

# Offline smoke test when Yahoo data is unavailable
python scripts/run_batch.py --synthetic --quick --no-hmm
```

`--garch` is the full daily walk-forward (every engine, every day) - hours;
run overnight. `--quick` (step 5) and `--no-hmm` are the fast previews. All
engines share one `--step` so the Diebold-Mariano series stay aligned.

That batch run writes the aggregate outputs to:

- `results/comparison_all.csv` for every market/model/alpha row
- `results/dm_all.csv` for pairwise Diebold-Mariano tables
- `results/group_summary.csv` for the developed-vs-emerging pass-rate summary
- `figures/<market>/` for the per-market breach, equity, and drawdown figures

```python
from mcrisk import data, compare_models, rank_by_calibration
from mcrisk.backtest.scoring import pairwise_dm_table

rets = data.simple_returns(data.load_prices("RELIANCE", "NSE"))
table, runs = compare_models(rets, alphas=(0.05, 0.01), window=500)
print(rank_by_calibration(table, 0.05))
print(pairwise_dm_table(runs, 0.05))
```

## Equity-curve / strategy risk

Feed any strategy's realised returns to simulate equity curves and read
drawdown distribution, risk of ruin, Sharpe/Calmar:

```python
from mcrisk.simulate import simulate_equity
from mcrisk.metrics import drawdown_distribution, risk_of_ruin

eq = simulate_equity(rets, method="block_bootstrap", horizon=252, n_sims=10000)
print(risk_of_ruin(eq, 0.5), drawdown_distribution(eq)["p95"])
```

Samplers: `iid_bootstrap`, `block_bootstrap` (Künsch), `stationary_bootstrap`
(Politis–Romano), `student_t`, `hmm_regime`.

## Layout

```
mcrisk/
  data.py                 NSE/BSE loader + synthetic DGPs
  engines/                gaussian, historical, student_t, fhs, hmm_regime
  backtest/
    coverage.py           Kupiec + Christoffersen
    scoring.py            McNeil–Frey ES test + pinball + Diebold–Mariano
    rolling.py            walk-forward harness
  metrics/risk.py         VaR/CVaR, drawdown, risk of ruin, Sharpe, Calmar
  simulate/equity.py      equity-curve simulators
  viz/plots.py            VaR-breach, equity fan, drawdown plots
  compare.py              top-level comparison + ranking
scripts/
  run_comparison.py       single-ticker / synthetic CLI
  run_batch.py            cross-market sweep -> results/ + figures//
figures/generate_figures.py   regenerates the synthetic demo figures
notebooks/case_study.py   reproducible 3-DGP study
tests/                    20 tests; coverage/ES validated against known-truth data
```

## Tests

```bash
python -m pytest -q        # 20 passing
```

The coverage and ES tests are validated by construction: a correctly-specified
model **passes** Kupiec/McNeil–Frey, and a deliberately mis-specified one
(half-volatility VaR, too-shallow ES) is **decisively rejected**.

## Scope notes for the paper

- **Keep LOB data out of the core.** VaR/CVaR is a return-distribution measure
  over a horizon; FI-2010-style LOB data is sub-second microstructure and a
  classification benchmark. Mixing daily market-level VaR with intraday
  microstructure VaR under one title dilutes the question. Make LOB tail risk a
  separate follow-up.
- **Cross-market** developed (e.g. ^GSPC, ^GDAXI) vs emerging (^NSEI, ^BSESN,
  ^BVSP) is the clean axis of variation.
- **Multiple testing:** with markets × methods × alphas, acknowledge / control
  the inflated false-positive rate when declaring methods "adequate".

## Key references

Kupiec (1995); Christoffersen (1998); McNeil & Frey (2000); Acerbi & Székely
(2014); Diebold & Mariano (1995); Künsch (1989); Politis & Romano (1994);
Hamilton (1989).
