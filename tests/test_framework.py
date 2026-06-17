"""Engine sanity, risk metrics, and full end-to-end comparison."""

import numpy as np
import pandas as pd

from mcrisk import data
from mcrisk.engines import (GaussianVaR, HistoricalVaR, StudentTVaR,
                            FilteredHistoricalVaR, HMMRegimeVaR)
from mcrisk.metrics import (max_drawdown, drawdown_distribution, risk_of_ruin,
                            sharpe_ratio, calmar_ratio, var_cvar)
from mcrisk.simulate import simulate_equity
from mcrisk.compare import compare_models
from mcrisk.backtest.scoring import pairwise_dm_table


# --- engines --------------------------------------------------------------- #
def test_all_engines_produce_ordered_forecasts():
    w = data.synth_student_t(n=800, seed=5)
    engines = [GaussianVaR(), HistoricalVaR(), StudentTVaR(),
               FilteredHistoricalVaR(), HMMRegimeVaR(n_states=2, n_draws=5000)]
    for e in engines:
        fc = e.forecast(w, 0.05)
        assert np.isfinite(fc.var) and np.isfinite(fc.cvar)
        assert fc.var < 0                     # 5% return quantile is a loss
        assert fc.cvar <= fc.var + 1e-9       # ES at least as deep as VaR


def test_student_t_deeper_tail_than_gaussian():
    w = data.synth_student_t(n=1500, nu=3.0, seed=6)
    g = GaussianVaR().forecast(w, 0.01).var
    t = StudentTVaR().forecast(w, 0.01).var
    assert t < g                              # fat tails -> deeper 1% VaR


# --- metrics --------------------------------------------------------------- #
def test_max_drawdown_known():
    assert abs(max_drawdown(np.array([1, 2, 1, 2])) - 0.5) < 1e-9
    eq = np.array([1.0, 1.1, 0.9, 1.2])
    assert abs(max_drawdown(eq) - (0.2 / 1.1)) < 1e-9


def test_risk_of_ruin_and_dd_distribution():
    paths = np.array([[1.0, 0.9, 0.4, 0.6],     # ruined (<0.5)
                      [1.0, 1.1, 1.05, 1.2],     # safe
                      [1.0, 0.8, 0.7, 0.75]])    # safe
    assert abs(risk_of_ruin(paths, 0.5) - 1/3) < 1e-9
    dd = drawdown_distribution(paths)
    assert 0 <= dd["median"] <= dd["worst"] <= 1


def test_sharpe_and_calmar_signs():
    up = np.full(252, 0.001)
    assert sharpe_ratio(up) > 0
    # curve must contain a drawdown for Calmar to be defined (else mdd=0 -> 0)
    eq = np.concatenate([np.linspace(1, 1.5, 130), np.linspace(1.5, 1.3, 20),
                         np.linspace(1.3, 2.0, 103)])
    assert calmar_ratio(eq) > 0


def test_var_cvar_ordering():
    r = np.random.default_rng(0).standard_normal(10000)
    v, c = var_cvar(r, 0.05)
    assert c <= v < 0


# --- simulation ------------------------------------------------------------ #
def test_simulators_run_and_shape():
    rets = data.synth_regime_switching(n=1500, seed=0)
    for method in ["iid_bootstrap", "block_bootstrap",
                   "stationary_bootstrap", "student_t", "hmm_regime"]:
        eq = simulate_equity(rets, method=method, horizon=60, n_sims=300, seed=1)
        assert eq.shape == (300, 61)
        assert np.all(eq[:, 0] == 1.0)        # all start at V0=1


# --- end to end ------------------------------------------------------------ #
def test_end_to_end_comparison_on_regime_data():
    rets = data.synth_regime_switching(n=2200, seed=0)
    table, runs = compare_models(rets, alphas=(0.05,), window=500)
    assert isinstance(table, pd.DataFrame)
    assert set(table["model"]) == {"gaussian", "historical", "student_t",
                                   "fhs_ewma", "hmm_regime"}
    assert table[["breach_rate", "kupiec_p", "cc_p"]].notna().all().all()
    dm = pairwise_dm_table(runs, 0.05)
    assert len(dm) == 10                       # C(5,2) pairwise comparisons
    assert set(dm["winner"]).issubset({"gaussian", "historical", "student_t",
                                       "fhs_ewma", "hmm_regime", "tie"})
