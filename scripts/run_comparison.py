#!/usr/bin/env python
"""
Run the full VaR-method comparison study on an NSE/BSE ticker (or synthetic
data) and emit the coverage-test results table plus diagnostic plots.

Examples
--------
  python scripts/run_comparison.py RELIANCE --exchange NSE --window 500 --plot
  python scripts/run_comparison.py --synthetic regime --plot
  python scripts/run_comparison.py INFY --garch
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcrisk import compare_models, rank_by_calibration  # noqa: E402
from mcrisk import data  # noqa: E402
from mcrisk.simulate import simulate_equity  # noqa: E402
from mcrisk.metrics import drawdown_distribution, risk_of_ruin  # noqa: E402


def get_returns(args) -> np.ndarray:
    if args.synthetic:
        gen = {"gaussian": data.synth_gaussian,
               "student_t": data.synth_student_t,
               "regime": data.synth_regime_switching}[args.synthetic]
        print(f"[data] synthetic '{args.synthetic}' returns")
        return gen(n=args.n, seed=args.seed)
    prices = data.load_prices(args.ticker, args.exchange, period=args.period)
    print(f"[data] {prices.name}: {len(prices)} prices "
          f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    return data.simple_returns(prices)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("ticker", nargs="?", default="RELIANCE")
    p.add_argument("--exchange", default="NSE", choices=["NSE", "BSE"])
    p.add_argument("--period", default="5y")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--step", type=int, default=1, help="rolling step (>1 = faster)")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.01])
    p.add_argument("--garch", action="store_true", help="use GARCH FHS (needs arch)")
    p.add_argument("--synthetic", choices=["gaussian", "student_t", "regime"])
    p.add_argument("--n", type=int, default=2500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args(argv)

    rets = get_returns(args)
    table, runs = compare_models(rets, alphas=tuple(args.alphas),
                                 window=args.window, step=args.step,
                                 use_garch=args.garch)

    import pandas as pd
    pd.set_option("display.width", 160, "display.max_columns", 20)
    print("\n================ VaR COMPARISON (out-of-sample) ================")
    print(table.to_string(index=False))
    for a in args.alphas:
        print(f"\n---- ranked by calibration @ alpha={a} ----")
        print(rank_by_calibration(table, a).to_string(index=False))

    if args.plot:
        from mcrisk.viz import plot_var_breaches, plot_equity_fan, plot_drawdown_dist
        a0 = args.alphas[0]
        for name in ["gaussian", "hmm_regime"]:
            run = runs.get((name, a0))
            if run is not None:
                print("saved:", plot_var_breaches(run, f"var_{name}.png"))
        eq = simulate_equity(rets, method="block_bootstrap", horizon=252,
                             n_sims=10000, seed=args.seed)
        print("saved:", plot_equity_fan(eq, "equity_fan.png"))
        dd = drawdown_distribution(eq)
        print("saved:", plot_drawdown_dist(dd["samples"], "drawdown_dist.png"))
        print(f"risk of ruin (-50%): {risk_of_ruin(eq, 0.5):.3%} | "
              f"median MDD {dd['median']:.2%} | 95th MDD {dd['p95']:.2%}")


if __name__ == "__main__":
    main()
