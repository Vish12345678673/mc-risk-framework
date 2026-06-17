"""
Data layer. Loads NSE/BSE prices via yfinance, plus synthetic generators used
by the test suite and for reproducible demos without network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SUFFIX = {"NSE": ".NS", "BSE": ".BO"}
TRADING_DAYS = 252


def resolve_symbol(ticker: str, exchange: str = "NSE") -> str:
    ticker = ticker.strip()
    # Index symbols (^GSPC, ^NSEI, ^BSESN ...) and RAW mode pass through as-is.
    if ticker.startswith("^") or exchange.upper() == "RAW":
        return ticker
    ticker = ticker.upper()
    if ticker.endswith((".NS", ".BO")):
        return ticker
    return ticker + SUFFIX[exchange.upper()]


def load_prices(ticker: str, exchange: str = "NSE", period: str = "5y",
                interval: str = "1d", fallback: bool = True) -> pd.Series:
    """Adjusted-close series from yfinance, with NSE<->BSE fallback.
    Use exchange='RAW' (or a '^' symbol) for indices / pre-suffixed tickers.

    Live fetching is disabled unless MCRISK_DATA_MODE=live. Research runs
    should read frozen snapshots via mcrisk.datasets.load_frozen().
    """
    import os
    if os.environ.get("MCRISK_DATA_MODE", "frozen").lower() != "live":
        raise RuntimeError(
            "Live fetch disabled (MCRISK_DATA_MODE=frozen). Research code must "
            "read frozen snapshots via mcrisk.datasets.load_frozen(). To build "
            "a snapshot, run: MCRISK_DATA_MODE=live python scripts/ingest.py"
        )
    import yfinance as yf

    is_index = ticker.strip().startswith("^") or exchange.upper() == "RAW"
    if is_index:
        attempts = ["RAW"]
    else:
        attempts = [exchange.upper()] + (
            [e for e in SUFFIX if e != exchange.upper()] if fallback else [])
    last_err = None
    for exch in attempts:
        sym = resolve_symbol(ticker, exch)
        try:
            df = yf.download(sym, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df is not None and not df.empty:
                close = df["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close = close.dropna().astype(float).sort_index()
                if len(close) > 60:
                    close.name = sym
                    return close
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"No data for {ticker} on {attempts}: {last_err}")


def log_returns(prices: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return np.diff(np.log(p))


def simple_returns(prices: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(prices, dtype=float)
    return p[1:] / p[:-1] - 1.0


# --- synthetic generators (reproducible, no network) ----------------------- #
def synth_gaussian(n=2000, mu=0.0003, sigma=0.012, seed=0) -> np.ndarray:
    return np.random.default_rng(seed).normal(mu, sigma, n)


def synth_student_t(n=2000, nu=4.0, mu=0.0003, sigma=0.012, seed=0) -> np.ndarray:
    from scipy import stats
    rng = np.random.default_rng(seed)
    return stats.t.rvs(nu, loc=mu, scale=sigma, size=n, random_state=rng)


def synth_regime_switching(n=2000, seed=0) -> np.ndarray:
    """Two-regime returns with persistent vol clustering (calm vs crisis)."""
    rng = np.random.default_rng(seed)
    trans = np.array([[0.98, 0.02], [0.10, 0.90]])
    mus = np.array([0.0006, -0.0010])
    sigs = np.array([0.008, 0.030])
    out = np.empty(n)
    s = 0
    for t in range(n):
        s = rng.choice(2, p=trans[s])
        out[t] = rng.normal(mus[s], sigs[s])
    return out
