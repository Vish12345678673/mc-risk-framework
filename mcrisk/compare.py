"""
Top-level model comparison: run every VaR engine through the rolling backtest
on the same series and assemble the results table that is the empirical core of
the study.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest.rolling import rolling_backtest
from .backtest.scoring import mcneil_frey_es_test
from .engines.fhs import FilteredHistoricalVaR
from .engines.hmm_regime import HMMRegimeVaR
from .engines.parametric import GaussianVaR, HistoricalVaR, StudentTVaR


def default_engines(use_garch: bool = False) -> list:
    return [
        GaussianVaR(),
        HistoricalVaR(),
        StudentTVaR(),
        FilteredHistoricalVaR(use_garch=use_garch),
        HMMRegimeVaR(n_states=3),
    ]


def compare_models(returns: np.ndarray, alphas=(0.05, 0.01), window: int = 500,
                   step: int = 1, engines: list | None = None,
                   use_garch: bool = False, dates=None) -> tuple[pd.DataFrame, dict]:
    """Run all engines at all alphas. Returns (results_table, raw_runs)."""
    engines = engines or default_engines(use_garch=use_garch)
    rows, runs = [], {}
    for model in engines:
        for alpha in alphas:
            run = rolling_backtest(returns, model, alpha, window=window,
                                   step=step, dates=dates)
            c = run.coverage
            es = mcneil_frey_es_test(run.realised, run.var, run.cvar)
            rows.append({
                "model": model.name,
                "alpha": alpha,
                "n": c.n_obs,
                "breaches": c.n_breaches,
                "breach_rate": round(c.breach_rate, 4),
                "expected": alpha,
                "mean_VaR": round(run.mean_var(), 5),
                "kupiec_p": round(c.kupiec_pvalue, 4),
                "christ_ind_p": round(c.christ_ind_pvalue, 4),
                "cc_p": round(c.cc_pvalue, 4),
                "es_p": round(es.pvalue, 4) if np.isfinite(es.pvalue) else np.nan,
                "verdict": c.verdict(),
            })
            runs[(model.name, alpha)] = run
    table = pd.DataFrame(rows).sort_values(["alpha", "model"]).reset_index(drop=True)
    return table, runs


def rank_by_calibration(table: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Rank models at a given alpha by closeness to nominal + CC p-value.
    Higher conditional-coverage p-value and breach_rate near alpha = better."""
    sub = table[table["alpha"] == alpha].copy()
    sub["abs_dev"] = (sub["breach_rate"] - alpha).abs()
    sub = sub.sort_values(["cc_p", "abs_dev"], ascending=[False, True])
    return sub[["model", "breach_rate", "kupiec_p", "christ_ind_p",
                "cc_p", "verdict"]].reset_index(drop=True)
