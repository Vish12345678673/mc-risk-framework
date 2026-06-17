"""
Regime-aware VaR via a Gaussian Hidden Markov Model.

Fit an HMM to the return window, read the filtered state probabilities at the
window's end, propagate one step through the transition matrix to get the
next-state distribution, and form the next-return density as a mixture of the
regime-conditional Gaussians. VaR/CVaR are read from that mixture by sampling.

This is the engine that reuses your existing HMM regime work. The research
question is whether this regime conditioning actually buys better-calibrated
tail risk than FHS / block bootstrap, or just adds variance.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

logging.getLogger("hmmlearn").setLevel(logging.ERROR)

from .base import VaRForecast, VaRModel


class HMMRegimeVaR(VaRModel):
    name = "hmm_regime"

    def __init__(self, n_states: int = 3, n_draws: int = 20_000,
                 seed: int = 0, covariance_type: str = "diag"):
        self.n_states = n_states
        self.n_draws = n_draws
        self.seed = seed
        self.covariance_type = covariance_type

    def forecast(self, window: np.ndarray, alpha: float) -> VaRForecast:
        from hmmlearn.hmm import GaussianHMM

        X = np.asarray(window, dtype=float).reshape(-1, 1)
        rng = np.random.default_rng(self.seed)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(n_components=self.n_states,
                                covariance_type=self.covariance_type,
                                n_iter=100, random_state=self.seed)
            try:
                model.fit(X)
                post = model.predict_proba(X)       # smoothed == filtered at T
                filt_T = post[-1]                    # current-state probs
                next_state = filt_T @ model.transmat_
                means = model.means_.ravel()
                covs = model.covars_.reshape(self.n_states, -1)[:, 0]
                sigmas = np.sqrt(np.maximum(covs, 1e-12))
            except Exception:
                # Degenerate window: fall back to empirical quantile.
                var = float(np.quantile(window, alpha))
                return VaRForecast(var, self._empirical_cvar(window, var), alpha)

        # Sample the next-state mixture.
        states = rng.choice(self.n_states, size=self.n_draws, p=next_state)
        draws = rng.normal(means[states], sigmas[states])
        var = float(np.quantile(draws, alpha))
        cvar = self._empirical_cvar(draws, var)
        return VaRForecast(var=var, cvar=cvar, alpha=alpha)
