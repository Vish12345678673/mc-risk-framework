"""Risk metrics computed on simulated equity curves or return samples."""

from __future__ import annotations

import numpy as np


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Historical VaR and CVaR (expected shortfall) at level alpha. Signed."""
    returns = np.asarray(returns, dtype=float)
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    cvar = float(tail.mean()) if tail.size else var
    return var, cvar


def max_drawdown(equity: np.ndarray) -> float:
    """Max peak-to-trough drawdown of a single equity curve (fraction, >=0)."""
    equity = np.asarray(equity, dtype=float)
    running_peak = np.maximum.accumulate(equity)
    dd = (running_peak - equity) / running_peak
    return float(np.max(dd))


def drawdown_distribution(equity_paths: np.ndarray) -> dict:
    """Max-drawdown distribution across many simulated equity curves."""
    dds = np.array([max_drawdown(p) for p in equity_paths])
    return {
        "median": float(np.median(dds)),
        "p95": float(np.percentile(dds, 95)),
        "worst": float(np.max(dds)),
        "mean": float(np.mean(dds)),
        "samples": dds,
    }


def risk_of_ruin(equity_paths: np.ndarray, ruin_level: float = 0.5,
                 start_value: float = 1.0) -> float:
    """P(equity ever falls below ruin_level * start_value) across paths."""
    threshold = ruin_level * start_value
    mins = equity_paths.min(axis=1)
    return float(np.mean(mins < threshold))


def sharpe_ratio(returns: np.ndarray, periods: int = 252,
                 rf: float = 0.0) -> float:
    returns = np.asarray(returns, dtype=float)
    excess = returns - rf / periods
    sd = excess.std(ddof=1)
    return float(np.sqrt(periods) * excess.mean() / sd) if sd > 0 else 0.0


def calmar_ratio(equity: np.ndarray, periods: int = 252) -> float:
    """Annualised return divided by max drawdown for one equity curve."""
    equity = np.asarray(equity, dtype=float)
    n = len(equity) - 1
    if n <= 0:
        return 0.0
    total = equity[-1] / equity[0] - 1.0
    ann = (1 + total) ** (periods / n) - 1.0
    mdd = max_drawdown(equity)
    return float(ann / mdd) if mdd > 0 else 0.0
