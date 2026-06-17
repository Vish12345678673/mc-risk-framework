"""
Strategy-level equity-curve simulation.

Given a strategy's realised per-period returns, simulate many possible equity
curves to characterise its risk: drawdown distribution, risk of ruin, terminal
P&L spread. Several samplers, in increasing realism:

  iid_bootstrap    -- resample returns independently. Destroys autocorrelation.
  block_bootstrap  -- resample fixed-length blocks (Kunsch). Preserves momentum
                      and volatility clustering.
  stationary_bootstrap -- random geometric block lengths (Politis-Romano).
  student_t        -- parametric fat-tailed i.i.d. draws.
  hmm_regime       -- sequence regimes via HMM transition matrix, draw
                      regime-conditional returns. Preserves regime persistence.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logging.getLogger("hmmlearn").setLevel(logging.ERROR)
from scipy import stats


def _curves_from_returns(sim_returns: np.ndarray, start: float) -> np.ndarray:
    """Compound (n_sims, horizon) simple returns into equity curves with V0."""
    growth = np.cumprod(1.0 + sim_returns, axis=1)
    v0 = np.full((sim_returns.shape[0], 1), start)
    return np.hstack([v0, start * growth])


def iid_bootstrap(returns, horizon, n_sims, start=1.0, rng=None):
    rng = rng or np.random.default_rng()
    s = rng.choice(returns, size=(n_sims, horizon), replace=True)
    return _curves_from_returns(s, start)


def block_bootstrap(returns, horizon, n_sims, block=10, start=1.0, rng=None):
    rng = rng or np.random.default_rng()
    returns = np.asarray(returns)
    n = len(returns)
    n_blocks = int(np.ceil(horizon / block))
    out = np.empty((n_sims, n_blocks * block))
    for i in range(n_sims):
        starts = rng.integers(0, n - block + 1, size=n_blocks)
        out[i] = np.concatenate([returns[s:s + block] for s in starts])
    return _curves_from_returns(out[:, :horizon], start)


def stationary_bootstrap(returns, horizon, n_sims, mean_block=10,
                         start=1.0, rng=None):
    """Politis-Romano: geometric block lengths, p = 1/mean_block."""
    rng = rng or np.random.default_rng()
    returns = np.asarray(returns)
    n = len(returns)
    p = 1.0 / mean_block
    sim = np.empty((n_sims, horizon))
    for i in range(n_sims):
        t = 0
        idx = rng.integers(0, n)
        while t < horizon:
            sim[i, t] = returns[idx]
            t += 1
            if rng.random() < p:
                idx = rng.integers(0, n)
            else:
                idx = (idx + 1) % n
    return _curves_from_returns(sim, start)


def student_t_sim(returns, horizon, n_sims, start=1.0, rng=None):
    rng = rng or np.random.default_rng()
    nu, loc, scale = stats.t.fit(returns)
    nu = max(nu, 2.5)
    s = stats.t.rvs(nu, loc=loc, scale=scale,
                    size=(n_sims, horizon), random_state=rng)
    return _curves_from_returns(s, start)


def hmm_regime_sim(returns, horizon, n_sims, n_states=3, start=1.0,
                   seed=0, rng=None):
    """Sequence regimes through the HMM transition matrix, draw conditionally."""
    from hmmlearn.hmm import GaussianHMM

    rng = rng or np.random.default_rng(seed)
    X = np.asarray(returns, dtype=float).reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(n_components=n_states, covariance_type="diag",
                            n_iter=100, random_state=seed)
        model.fit(X)

    trans = model.transmat_
    means = model.means_.ravel()
    sigmas = np.sqrt(model.covars_.reshape(n_states, -1)[:, 0])
    start_probs = model.predict_proba(X)[-1]

    sim = np.empty((n_sims, horizon))
    state = np.array([rng.choice(n_states, p=start_probs) for _ in range(n_sims)])
    for t in range(horizon):
        state = np.array([rng.choice(n_states, p=trans[s]) for s in state])
        sim[:, t] = rng.normal(means[state], sigmas[state])
    return _curves_from_returns(sim, start)


SAMPLERS = {
    "iid_bootstrap": iid_bootstrap,
    "block_bootstrap": block_bootstrap,
    "stationary_bootstrap": stationary_bootstrap,
    "student_t": student_t_sim,
    "hmm_regime": hmm_regime_sim,
}


def simulate_equity(returns, method="block_bootstrap", horizon=252,
                    n_sims=10_000, start=1.0, seed=42, **kw):
    rng = np.random.default_rng(seed)
    if method not in SAMPLERS:
        raise ValueError(f"method must be one of {list(SAMPLERS)}")
    return SAMPLERS[method](returns, horizon, n_sims, start=start, rng=rng, **kw)
