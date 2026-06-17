"""Correctness of the coverage tests: known answers + well/mis-specified pipelines."""

import numpy as np
from scipy import stats

from mcrisk.backtest.coverage import (kupiec_pof, christoffersen_independence,
                                      conditional_coverage)
from mcrisk.backtest.rolling import rolling_backtest
from mcrisk.engines.parametric import GaussianVaR
from mcrisk.engines.base import VaRForecast, VaRModel


class HalfSigmaGaussian(VaRModel):
    """Deliberately mis-specified: uses half the true volatility."""
    name = "half_sigma"

    def forecast(self, window, alpha):
        mu = float(np.mean(window))
        sigma = 0.5 * float(np.std(window, ddof=1))
        z = stats.norm.ppf(alpha)
        return VaRForecast(mu + sigma * z, mu + sigma * z * 1.2, alpha)


def test_kupiec_known_value():
    # 100 obs, exactly 5 breaches, alpha=0.05 => observed rate == expected.
    b = np.zeros(100, dtype=int); b[:5] = 1
    stat, p = kupiec_pof(b, 0.05)
    assert stat < 1e-8          # perfect calibration -> LR ~ 0
    assert p > 0.99


def test_kupiec_rejects_overbreach():
    b = np.zeros(100, dtype=int); b[:20] = 1   # 20% vs nominal 5%
    stat, p = kupiec_pof(b, 0.05)
    assert stat > 10 and p < 0.01


def test_kupiec_handles_zero_breaches():
    b = np.zeros(200, dtype=int)
    stat, p = kupiec_pof(b, 0.05)
    assert np.isfinite(stat) and stat > 0   # zero breaches still informative


def test_christoffersen_independence_iid_vs_clustered():
    rng = np.random.default_rng(0)
    iid = (rng.random(3000) < 0.05).astype(int)
    s_iid, p_iid = christoffersen_independence(iid)
    # clustered: breaches arrive in runs -> dependence
    clustered = np.zeros(3000, dtype=int)
    clustered[::50] = 1; clustered[1::50] = 1; clustered[2::50] = 1
    s_cl, p_cl = christoffersen_independence(clustered)
    assert p_iid > 0.05          # iid not flagged
    assert s_cl > s_iid          # clustering yields larger statistic


def test_wellspecified_gaussian_passes_coverage():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0003, 0.012, 2600)
    run = rolling_backtest(rets, GaussianVaR(), 0.05, window=500)
    assert 0.035 < run.coverage.breach_rate < 0.065
    assert run.coverage.kupiec_pvalue > 0.05
    assert run.coverage.cc_pvalue > 0.05


def test_misspecified_gaussian_fails_coverage():
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0003, 0.012, 2600)
    run = rolling_backtest(rets, HalfSigmaGaussian(), 0.05, window=500)
    assert run.coverage.breach_rate > 0.12      # far too many breaches
    assert run.coverage.kupiec_pvalue < 0.01    # decisively rejected


def test_conditional_coverage_combines_both():
    b = np.zeros(1000, dtype=int); b[:50] = 1
    res = conditional_coverage(b, 0.05)
    assert abs(res.cc_stat - (res.kupiec_stat + res.christ_ind_stat)) < 1e-9
