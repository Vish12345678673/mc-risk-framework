"""Parametric and non-parametric one-step VaR engines."""

from __future__ import annotations

import numpy as np
from scipy import stats

from .base import VaRForecast, VaRModel


class GaussianVaR(VaRModel):
    """Normal distribution fitted to the trailing window. The naive baseline."""
    name = "gaussian"

    def forecast(self, window: np.ndarray, alpha: float) -> VaRForecast:
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        z = stats.norm.ppf(alpha)
        var = mu + sigma * z
        # Closed-form Gaussian expected shortfall:
        cvar = mu - sigma * stats.norm.pdf(z) / alpha
        return VaRForecast(var=var, cvar=cvar, alpha=alpha)


class HistoricalVaR(VaRModel):
    """Empirical quantile of the window. No distributional assumption."""
    name = "historical"

    def forecast(self, window: np.ndarray, alpha: float) -> VaRForecast:
        var = float(np.quantile(window, alpha))
        cvar = self._empirical_cvar(window, var)
        return VaRForecast(var=var, cvar=cvar, alpha=alpha)


class StudentTVaR(VaRModel):
    """Student-t fitted to the window. Fat tails -> deeper VaR than Gaussian."""
    name = "student_t"

    def forecast(self, window: np.ndarray, alpha: float) -> VaRForecast:
        nu, loc, scale = stats.t.fit(window)
        nu = max(nu, 2.5)
        var = float(stats.t.ppf(alpha, nu, loc=loc, scale=scale))
        # ES for Student-t (Acerbi closed form):
        x = stats.t.ppf(alpha, nu)
        es_std = -(stats.t.pdf(x, nu) / alpha) * ((nu + x**2) / (nu - 1))
        cvar = float(loc + scale * es_std)
        return VaRForecast(var=var, cvar=cvar, alpha=alpha)
