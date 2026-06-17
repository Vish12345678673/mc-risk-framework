"""
Scoring and ES evaluation — the upgrades that let you claim *which* method is
better, and that the CVaR forecasts (not just the VaR quantiles) are calibrated.

Expected Shortfall backtest
---------------------------
Kupiec/Christoffersen test the VaR quantile only. ES needs its own test.
McNeil & Frey (2000): on breach days, the exceedance residual

    z_t = realised_t - ES_t        (or standardised by conditional vol)

should have zero mean under a correctly specified ES. We test
H0: mean(z on breaches) = 0  against  H1: mean < 0 (losses worse than ES, i.e.
ES underestimates the tail) using a one-sided bootstrap p-value.

Model comparison
----------------
Pinball (quantile) loss is the proper scoring rule for a VaR quantile:

    L_alpha(r, q) = (alpha - 1{r < q}) * (r - q)

Lower mean loss = better quantile forecast. To test whether model A genuinely
beats model B (not just by luck), apply a Diebold-Mariano test to the per-day
loss differential, with a HAC (Newey-West) variance and the Harvey small-sample
correction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Expected Shortfall backtest (McNeil-Frey exceedance residuals)
# --------------------------------------------------------------------------- #
@dataclass
class ESTestResult:
    n_breaches: int
    mean_residual: float
    statistic: float
    pvalue: float

    @property
    def passes(self) -> bool:
        return self.pvalue > 0.05


def mcneil_frey_es_test(realised: np.ndarray, var: np.ndarray, cvar: np.ndarray,
                        n_boot: int = 10_000, seed: int = 0) -> ESTestResult:
    """One-sided bootstrap test that exceedance residuals have zero mean.

    realised < var defines a breach; residual = realised - cvar on breach days.
    A significantly negative mean => ES too small (tail losses exceed ES).
    """
    realised = np.asarray(realised, dtype=float)
    var = np.asarray(var, dtype=float)
    cvar = np.asarray(cvar, dtype=float)

    mask = realised < var
    resid = realised[mask] - cvar[mask]
    k = resid.size
    if k < 2:
        return ESTestResult(k, float("nan"), float("nan"), float("nan"))

    obs_mean = float(resid.mean())
    # Bootstrap the sampling distribution of the mean under recentred residuals.
    rng = np.random.default_rng(seed)
    centred = resid - obs_mean
    boot_means = centred[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    # one-sided: P(bootstrap mean <= observed mean) tests H1: mean < 0
    pvalue = float(np.mean(boot_means <= obs_mean))
    stat = obs_mean / (resid.std(ddof=1) / np.sqrt(k) + 1e-12)
    return ESTestResult(k, obs_mean, float(stat), pvalue)


# --------------------------------------------------------------------------- #
# Pinball loss + Diebold-Mariano pairwise comparison
# --------------------------------------------------------------------------- #
def pinball_loss(realised: np.ndarray, var: np.ndarray, alpha: float) -> np.ndarray:
    """Per-observation quantile (pinball) loss for a VaR forecast."""
    realised = np.asarray(realised, dtype=float)
    var = np.asarray(var, dtype=float)
    indicator = (realised < var).astype(float)
    return (alpha - indicator) * (realised - var)


@dataclass
class DMResult:
    model_a: str
    model_b: str
    mean_loss_a: float
    mean_loss_b: float
    dm_stat: float
    pvalue: float

    def winner(self, level: float = 0.05) -> str:
        if not np.isfinite(self.pvalue) or self.pvalue > level:
            return "tie"
        return self.model_a if self.mean_loss_a < self.mean_loss_b else self.model_b


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1,
                    name_a: str = "A", name_b: str = "B") -> DMResult:
    """Two-sided DM test on loss differential with Newey-West HAC variance
    and Harvey-Leybourne-Newbold small-sample correction."""
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    n = d.size
    d_bar = d.mean()

    # Newey-West long-run variance with (h-1) lags.
    gamma0 = np.mean((d - d_bar) ** 2)
    lrv = gamma0
    for lag in range(1, h):
        w = 1.0 - lag / h
        cov = np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar))
        lrv += 2.0 * w * cov
    var_dbar = lrv / n
    if var_dbar <= 0:
        return DMResult(name_a, name_b, float(loss_a.mean()),
                        float(loss_b.mean()), 0.0, 1.0)

    dm = d_bar / np.sqrt(var_dbar)
    # Harvey-Leybourne-Newbold correction
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm *= corr
    pvalue = 2.0 * float(stats.t.sf(abs(dm), df=n - 1))
    return DMResult(name_a, name_b, float(loss_a.mean()),
                    float(loss_b.mean()), float(dm), pvalue)


def pairwise_dm_table(runs: dict, alpha: float):
    """Build a pairwise DM comparison table across all models at one alpha.
    `runs` maps (model_name, alpha) -> BacktestRun. Returns a DataFrame."""
    import pandas as pd

    items = [(name, run) for (name, a), run in runs.items() if a == alpha]
    losses = {name: pinball_loss(run.realised, run.var, alpha)
              for name, run in items}
    names = list(losses)
    rows = []
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            res = diebold_mariano(losses[a_name], losses[b_name],
                                  name_a=a_name, name_b=b_name)
            rows.append({
                "model_a": a_name, "model_b": b_name,
                "loss_a": round(res.mean_loss_a, 6),
                "loss_b": round(res.mean_loss_b, 6),
                "dm_stat": round(res.dm_stat, 3),
                "p": round(res.pvalue, 4),
                "winner": res.winner(),
            })
    return pd.DataFrame(rows)
