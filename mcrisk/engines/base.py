"""
Common interface for one-step-ahead VaR/CVaR models.

Every engine implements `forecast(window) -> VaRForecast` where `window` is a
1-D array of past returns (most recent last). The forecast is the conditional
VaR_alpha and CVaR_alpha (expected shortfall) for the *next* return, expressed
as signed returns (a loss is negative). A breach occurs when the realised
next return < VaR_alpha.

Keeping a single interface is what makes the model comparison fair: the
rolling harness feeds every engine identical windows and scores identical
breach sequences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VaRForecast:
    var: float    # alpha-quantile of next return (signed; loss is negative)
    cvar: float   # mean return conditional on being below var (signed)
    alpha: float


class VaRModel:
    name: str = "base"

    def forecast(self, window: np.ndarray, alpha: float) -> VaRForecast:
        raise NotImplementedError

    @staticmethod
    def _empirical_cvar(samples: np.ndarray, var: float) -> float:
        tail = samples[samples <= var]
        return float(tail.mean()) if tail.size else float(var)
