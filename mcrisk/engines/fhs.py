"""
Filtered Historical Simulation (FHS).

Plain historical simulation assumes today's volatility equals the window
average. FHS fixes that: it filters returns by a conditional-volatility model,
takes the standardised residuals (which are closer to i.i.d.), and rescales
their empirical quantile by *tomorrow's* forecast volatility. This is the
standard fix for volatility clustering and usually the strongest non-regime
baseline.

Two vol filters:
  * EWMA (RiskMetrics, lambda=0.94) -- dependency-free, always available.
  * GARCH(1,1) via the `arch` package -- used if installed and use_garch=True.
"""

from __future__ import annotations

import numpy as np

from .base import VaRForecast, VaRModel


def ewma_volatility(returns: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """RiskMetrics EWMA conditional variance series (same length as returns)."""
    r = np.asarray(returns, dtype=float)
    var = np.empty_like(r)
    var[0] = np.var(r)
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(var)


class FilteredHistoricalVaR(VaRModel):
    name = "fhs_ewma"

    def __init__(self, lam: float = 0.94, use_garch: bool = False):
        self.lam = lam
        self.use_garch = use_garch
        if use_garch:
            self.name = "fhs_garch"

    def _garch_sigma(self, window: np.ndarray):
        """Return (sigma_series, sigma_next) from a GARCH(1,1) fit, or None."""
        try:
            from arch import arch_model
            scale = 100.0  # arch prefers percentage-scale returns
            am = arch_model(window * scale, mean="Constant", vol="GARCH",
                            p=1, q=1, dist="normal")
            res = am.fit(disp="off", show_warning=False)
            cond = res.conditional_volatility / scale
            fc = res.forecast(horizon=1, reindex=False)
            sigma_next = float(np.sqrt(fc.variance.values[-1, 0]) / scale)
            return cond, sigma_next
        except Exception:
            return None

    def forecast(self, window: np.ndarray, alpha: float) -> VaRForecast:
        window = np.asarray(window, dtype=float)
        mu = float(np.mean(window))

        sigma_series = sigma_next = None
        if self.use_garch:
            out = self._garch_sigma(window)
            if out is not None:
                sigma_series, sigma_next = out

        if sigma_series is None:  # EWMA path (also the fallback)
            sigma_series = ewma_volatility(window, self.lam)
            sigma_next = float(self.lam * sigma_series[-1] ** 2
                               + (1 - self.lam) * window[-1] ** 2) ** 0.5

        sigma_series = np.where(sigma_series <= 0, np.nan, sigma_series)
        std_resid = (window - mu) / sigma_series
        std_resid = std_resid[np.isfinite(std_resid)]

        q = float(np.quantile(std_resid, alpha))
        var = mu + sigma_next * q
        es_std = self._empirical_cvar(std_resid, q)
        cvar = mu + sigma_next * es_std
        return VaRForecast(var=var, cvar=cvar, alpha=alpha)
