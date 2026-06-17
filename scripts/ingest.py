#!/usr/bin/env python
"""
Ingest live market data into a frozen snapshot.

This is the only script that should fetch live market data. Normal research
runs read data/raw/<snapshot>/*.csv through mcrisk.datasets.load_frozen().
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcrisk import datasets  # noqa: E402


def main(argv=None):
    if os.environ.get("MCRISK_DATA_MODE") != "live":
        sys.exit(
            "Refusing to fetch: set MCRISK_DATA_MODE=live to run ingest. "
            "(Research runs must stay frozen.)"
        )

    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (or use --period)")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD; pins the window end")
    parser.add_argument("--period", default="20y", help="used only if --start absent")
    parser.add_argument("--markets", nargs="+", help="subset of labels (default: all)")
    parser.add_argument(
        "--snapshot",
        default=None,
        help="snapshot name (default: --end date or today)",
    )
    args = parser.parse_args(argv)

    import yfinance as yf
    from datetime import date

    labels = args.markets or list(datasets.MARKETS)
    snapshot = args.snapshot or args.end or date.today().isoformat()

    series, fetched = {}, {}
    for label in labels:
        symbol = datasets.MARKETS[label][0]
        kwargs = dict(auto_adjust=True, progress=False)
        if args.start:
            df = yf.download(symbol, start=args.start, end=args.end, **kwargs)
        else:
            df = yf.download(symbol, period=args.period, end=args.end, **kwargs)
        if df is None or df.empty:
            print(f"  !! {label} ({symbol}) returned no data - skipped")
            continue
        close = df["Close"]
        if hasattr(close, "iloc") and close.ndim > 1:
            close = close.iloc[:, 0]
        series[label] = close.dropna().astype(float)
        fetched[label] = symbol
        print(
            f"  {label:8s} {symbol:9s} {len(series[label])} rows "
            f"{series[label].index.min().date()} -> "
            f"{series[label].index.max().date()}"
        )

    if not series:
        sys.exit("No data fetched; nothing written.")

    meta = {
        "yfinance_version": yf.__version__,
        "requested": {
            "start": args.start,
            "end": args.end,
            "period": args.period,
            "markets": fetched,
        },
    }
    out = datasets.write_snapshot(snapshot, series, meta)
    datasets.set_active_snapshot(snapshot)
    print(f"\nFrozen snapshot written: {out}")
    print(f"ACTIVE_SNAPSHOT -> {snapshot}")
    verification = datasets.verify_snapshot(snapshot)
    print("integrity:", "OK" if verification["all_ok"] else verification["per_market"])


if __name__ == "__main__":
    main()
