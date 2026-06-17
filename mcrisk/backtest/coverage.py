"""
VaR coverage backtests: the evaluation core of the framework.

A one-step-ahead VaR_alpha forecast is a falsifiable claim. If the model is
well specified, realised returns should breach the forecast with probability
alpha, and the breaches should be serially independent (not clustered in
crises). These tests check exactly that.

  Kupiec POF (1995)        -> unconditional coverage  (is the breach rate right?)
  Christoffersen (1998)    -> independence            (are breaches clustered?)
  Conditional coverage     -> joint test  (LR_uc + LR_ind ~ chi2(2))

All log-likelihood terms use safe handling of zero counts / boundary
probabilities, so the statistics are finite for any breach sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


def _xlog(count: float, prob: float) -> float:
    """count * log(prob), with the convention 0*log(0) = 0 and guards."""
    if count <= 0:
        return 0.0
    if prob <= 0.0:
        return -np.inf
    return count * np.log(prob)


@dataclass
class CoverageResult:
    alpha: float
    n_obs: int
    n_breaches: int
    breach_rate: float
    expected_rate: float
    kupiec_stat: float
    kupiec_pvalue: float
    christ_ind_stat: float
    christ_ind_pvalue: float
    cc_stat: float
    cc_pvalue: float

    def verdict(self, level: float = 0.05) -> str:
        ok_uc = self.kupiec_pvalue > level
        ok_ind = self.christ_ind_pvalue > level
        ok_cc = self.cc_pvalue > level
        return (f"coverage {'OK ' if ok_uc else 'BAD'} | "
                f"independence {'OK ' if ok_ind else 'BAD'} | "
                f"conditional {'OK ' if ok_cc else 'BAD'}")

    def as_row(self) -> dict:
        return {
            "alpha": self.alpha,
            "n": self.n_obs,
            "breaches": self.n_breaches,
            "breach_rate": round(self.breach_rate, 4),
            "expected": self.expected_rate,
            "kupiec_p": round(self.kupiec_pvalue, 4),
            "christ_ind_p": round(self.christ_ind_pvalue, 4),
            "cc_p": round(self.cc_pvalue, 4),
        }


def kupiec_pof(breaches: np.ndarray, alpha: float) -> tuple[float, float]:
    """Unconditional-coverage likelihood-ratio test. Returns (stat, pvalue)."""
    breaches = np.asarray(breaches).astype(int)
    n = breaches.size
    x = int(breaches.sum())
    pi_hat = x / n if n else 0.0

    ll_null = _xlog(x, alpha) + _xlog(n - x, 1 - alpha)
    ll_alt = _xlog(x, pi_hat) + _xlog(n - x, 1 - pi_hat)
    stat = -2.0 * (ll_null - ll_alt)
    stat = max(stat, 0.0)
    pvalue = float(stats.chi2.sf(stat, df=1))
    return float(stat), pvalue


def christoffersen_independence(breaches: np.ndarray) -> tuple[float, float]:
    """Markov independence LR test on the breach sequence. Returns (stat, p)."""
    b = np.asarray(breaches).astype(int)
    if b.size < 2:
        return 0.0, 1.0

    prev, cur = b[:-1], b[1:]
    n00 = int(np.sum((prev == 0) & (cur == 0)))
    n01 = int(np.sum((prev == 0) & (cur == 1)))
    n10 = int(np.sum((prev == 1) & (cur == 0)))
    n11 = int(np.sum((prev == 1) & (cur == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_alt = (_xlog(n00, 1 - pi01) + _xlog(n01, pi01)
              + _xlog(n10, 1 - pi11) + _xlog(n11, pi11))
    ll_null = _xlog(n00 + n10, 1 - pi) + _xlog(n01 + n11, pi)

    stat = -2.0 * (ll_null - ll_alt)
    if not np.isfinite(stat):
        stat = 0.0
    stat = max(stat, 0.0)
    pvalue = float(stats.chi2.sf(stat, df=1))
    return float(stat), pvalue


def conditional_coverage(breaches: np.ndarray, alpha: float) -> CoverageResult:
    """Full Christoffersen battery: Kupiec + independence + joint CC test."""
    breaches = np.asarray(breaches).astype(int)
    n = breaches.size
    x = int(breaches.sum())

    uc_stat, uc_p = kupiec_pof(breaches, alpha)
    ind_stat, ind_p = christoffersen_independence(breaches)
    cc_stat = uc_stat + ind_stat
    cc_p = float(stats.chi2.sf(cc_stat, df=2))

    return CoverageResult(
        alpha=alpha, n_obs=n, n_breaches=x,
        breach_rate=(x / n if n else 0.0), expected_rate=alpha,
        kupiec_stat=uc_stat, kupiec_pvalue=uc_p,
        christ_ind_stat=ind_stat, christ_ind_pvalue=ind_p,
        cc_stat=cc_stat, cc_pvalue=cc_p,
    )
