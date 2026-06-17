"""ES backtest and Diebold-Mariano scoring: validated against known-truth data."""

import numpy as np
from scipy import stats

from mcrisk.backtest.scoring import (mcneil_frey_es_test, pinball_loss,
                                     diebold_mariano)


def _gaussian_var_es(alpha=0.05):
    z = stats.norm.ppf(alpha)
    var = z
    es = -stats.norm.pdf(z) / alpha
    return var, es


def test_es_test_passes_when_correct():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(20000)
    var, es = _gaussian_var_es(0.05)
    res = mcneil_frey_es_test(r, np.full_like(r, var), np.full_like(r, es))
    assert res.n_breaches > 200
    assert abs(res.mean_residual) < 0.1     # residuals centred near zero
    assert res.pvalue > 0.05                # not rejected


def test_es_test_rejects_when_es_too_shallow():
    rng = np.random.default_rng(1)
    r = rng.standard_normal(20000)
    var, es = _gaussian_var_es(0.05)
    shallow_es = np.full_like(r, var)       # ES set equal to VaR (too shallow)
    res = mcneil_frey_es_test(r, np.full_like(r, var), shallow_es)
    assert res.mean_residual < 0            # realised tail worse than ES
    assert res.pvalue < 0.05                # rejected


def test_pinball_loss_lower_for_correct_quantile():
    rng = np.random.default_rng(2)
    r = rng.standard_normal(5000)
    var_true = stats.norm.ppf(0.05)
    good = pinball_loss(r, np.full_like(r, var_true), 0.05).mean()
    bad = pinball_loss(r, np.zeros_like(r), 0.05).mean()        # absurd VaR=0
    assert good < bad


def test_dm_detects_better_model():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(4000)
    var_true = stats.norm.ppf(0.05)
    loss_good = pinball_loss(r, np.full_like(r, var_true), 0.05)
    loss_bad = pinball_loss(r, np.zeros_like(r), 0.05)
    res = diebold_mariano(loss_good, loss_bad, name_a="good", name_b="bad")
    assert res.winner() == "good"
    assert res.pvalue < 0.05


def test_dm_ties_identical_models():
    rng = np.random.default_rng(4)
    loss = rng.random(1000)
    res = diebold_mariano(loss, loss.copy(), name_a="a", name_b="b")
    assert res.winner() == "tie"
