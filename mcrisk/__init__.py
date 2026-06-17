"""Monte Carlo Risk Framework for Quantitative Trading Strategies."""
from .compare import compare_models, rank_by_calibration, default_engines
from .backtest.rolling import rolling_backtest
from .backtest.coverage import conditional_coverage, kupiec_pof, christoffersen_independence
from . import data, datasets, simulate, metrics

__version__ = "0.1.0"
__all__ = ["compare_models", "rank_by_calibration", "default_engines",
           "rolling_backtest", "conditional_coverage", "kupiec_pof",
           "christoffersen_independence", "data", "datasets", "simulate",
           "metrics"]
