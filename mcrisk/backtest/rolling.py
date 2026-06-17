"""
Walk-forward rolling VaR backtester.

For each out-of-sample day t, fit/forecast on the trailing `window` of returns,
record VaR_alpha(t) and CVaR_alpha(t), then observe the realised return and
mark a breach if r_t < VaR_alpha(t). The resulting breach sequence is what the
coverage tests consume. This is genuine out-of-sample evaluation: nothing at
time t uses any information from t or later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engines.base import VaRModel
from .coverage import CoverageResult, conditional_coverage


@dataclass
class BacktestRun:
    model_name: str
    alpha: float
    dates: np.ndarray
    realised: np.ndarray
    var: np.ndarray
    cvar: np.ndarray
    breaches: np.ndarray
    coverage: CoverageResult

    def mean_var(self) -> float:
        return float(np.mean(self.var))

    def es_exceedance(self) -> float:
        """Mean realised loss on breach days minus mean predicted CVaR there.
        Near zero => the ES forecast is well sized on the tail."""
        mask = self.breaches.astype(bool)
        if not mask.any():
            return float("nan")
        return float(np.mean(self.realised[mask] - self.cvar[mask]))


def rolling_backtest(returns: np.ndarray, model: VaRModel, alpha: float,
                     window: int = 500, step: int = 1,
                     dates: np.ndarray | None = None) -> BacktestRun:
    returns = np.asarray(returns, dtype=float)
    n = returns.size
    if window >= n:
        raise ValueError(f"window ({window}) must be < series length ({n}).")
    if dates is None:
        dates = np.arange(n)

    idx, vars_, cvars_, real_ = [], [], [], []
    for t in range(window, n, step):
        w = returns[t - window:t]
        fc = model.forecast(w, alpha)
        idx.append(t)
        vars_.append(fc.var)
        cvars_.append(fc.cvar)
        real_.append(returns[t])

    vars_ = np.asarray(vars_)
    cvars_ = np.asarray(cvars_)
    real_ = np.asarray(real_)
    breaches = (real_ < vars_).astype(int)
    cov = conditional_coverage(breaches, alpha)

    return BacktestRun(
        model_name=model.name, alpha=alpha,
        dates=np.asarray(dates)[idx], realised=real_,
        var=vars_, cvar=cvars_, breaches=breaches, coverage=cov,
    )
