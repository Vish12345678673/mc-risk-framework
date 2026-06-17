#!/usr/bin/env python
"""
Batch driver — produces the full results set for the paper in one run.

Sweeps a basket of developed and emerging market indices, runs the rolling
VaR/CVaR backtest for every engine, and collects:

  results/comparison_all.csv   every (market, model, alpha) coverage + ES row
  results/dm_all.csv           pairwise Diebold-Mariano (pinball) per market
  results/group_summary.csv    developed-vs-emerging win-rates per method
  figures/<market>/...         per-market table + breach + equity figures

RUNTIME NOTE
------------
The HMM engine refits at every rolling step, so step=1 over the full frozen
snapshot is hours (fine overnight). Use --quick (step=5) for a tractable first
pass, or --no-hmm to drop the slow engine. All engines share one step so the pairwise
Diebold-Mariano series stay aligned.

Examples
--------
  python scripts/run_batch.py --quick                  # frozen-data preview
  python scripts/run_batch.py --garch                  # final frozen run
  python scripts/run_batch.py --markets Nifty50 SP500  # labeled subset
  python scripts/run_batch.py --synthetic              # offline smoke test
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcrisk import data, datasets, compare_models, rank_by_calibration  # noqa: E402
from mcrisk.backtest.scoring import pairwise_dm_table  # noqa: E402
from mcrisk.simulate import simulate_equity  # noqa: E402
from mcrisk.metrics import drawdown_distribution, risk_of_ruin  # noqa: E402
from mcrisk.viz import plot_var_breaches, plot_equity_fan, plot_drawdown_dist  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MARKETS = datasets.MARKETS


def get_returns(label, args):
    if args.synthetic:
        gen = {"SP500": data.synth_gaussian, "DAX": data.synth_gaussian}.get(
            label, data.synth_regime_switching)
        return gen(n=1600, seed=hash(label) % 1000)
    prices = datasets.load_frozen(label, snapshot=args.snapshot)
    return data.simple_returns(prices)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--markets", nargs="+",
                   help="subset of labels (default: all in the snapshot)")
    p.add_argument("--snapshot", default=None,
                   help="frozen snapshot date (default: data/ACTIVE_SNAPSHOT)")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--quick", action="store_true", help="shortcut for --step 5")
    p.add_argument("--no-hmm", action="store_true")
    p.add_argument("--garch", action="store_true")
    p.add_argument("--synthetic", action="store_true", help="offline smoke test")
    p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.01])
    args = p.parse_args(argv)
    if args.quick:
        args.step = 5

    selected = args.markets or list(MARKETS)
    res_dir = os.path.join(ROOT, "results")
    os.makedirs(res_dir, exist_ok=True)

    engines = None
    if args.no_hmm:
        from mcrisk.engines import (GaussianVaR, HistoricalVaR, StudentTVaR,
                                    FilteredHistoricalVaR)
        engines = [GaussianVaR(), HistoricalVaR(), StudentTVaR(),
                   FilteredHistoricalVaR(use_garch=args.garch)]

    all_rows, all_dm, used = [], [], []
    for label in selected:
        symbol, group = MARKETS.get(label, (label, "unknown"))
        print(f"\n[{label}] {symbol} ({group}) ...")
        try:
            rets = get_returns(label, args)
        except Exception as e:  # noqa: BLE001
            print(f"  !! skipped ({e})")
            continue
        used.append(label)
        print(f"  {len(rets)} returns")

        table, runs = compare_models(rets, alphas=tuple(args.alphas),
                                     window=args.window, step=args.step,
                                     engines=engines, use_garch=args.garch)
        table.insert(0, "group", group)
        table.insert(0, "market", label)
        all_rows.append(table)

        fdir = os.path.join(ROOT, "figures", label)
        os.makedirs(fdir, exist_ok=True)
        for a in args.alphas:
            dm = pairwise_dm_table(runs, a)
            dm.insert(0, "alpha", a); dm.insert(0, "group", group)
            dm.insert(0, "market", label)
            all_dm.append(dm)
        a0 = args.alphas[0]
        for name in {n for (n, _a) in runs}:
            run = runs.get((name, a0))
            if run is not None:
                plot_var_breaches(run, os.path.join(fdir, f"breaches_{name}.png"))
        eq = simulate_equity(rets, method="block_bootstrap", horizon=252,
                             n_sims=8000, seed=42)
        plot_equity_fan(eq, os.path.join(fdir, "equity_fan.png"),
                        title=f"{label} block-bootstrap equity (1y)")
        dd = drawdown_distribution(eq)
        plot_drawdown_dist(dd["samples"], os.path.join(fdir, "drawdown.png"))
        print(f"  ruin(-50%)={risk_of_ruin(eq,0.5):.2%} p95MDD={dd['p95']:.1%}")
        print(rank_by_calibration(table, a0).head(3)[
            ["model", "breach_rate", "cc_p"]].to_string(index=False))

    if not all_rows:
        print("\nNo markets produced data."); return

    comp = pd.concat(all_rows, ignore_index=True)
    dmall = pd.concat(all_dm, ignore_index=True)
    comp.to_csv(os.path.join(res_dir, "comparison_all.csv"), index=False)
    dmall.to_csv(os.path.join(res_dir, "dm_all.csv"), index=False)

    # developed-vs-emerging: how often is each method conditional-coverage OK?
    comp["cc_ok"] = comp["cc_p"] > 0.05
    summary = (comp.groupby(["group", "alpha", "model"])["cc_ok"]
               .mean().rename("cc_pass_rate").reset_index()
               .sort_values(["alpha", "group", "cc_pass_rate"],
                            ascending=[True, True, False]))
    summary.to_csv(os.path.join(res_dir, "group_summary.csv"), index=False)

    import json
    from datetime import datetime, timezone

    meta = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "code_commit": datasets.git_commit(),
            "args": {"window": args.window, "step": args.step,
                     "garch": args.garch, "no_hmm": args.no_hmm,
                     "alphas": args.alphas},
            "markets_used": used}
    if args.synthetic:
        meta["snapshot"] = "SYNTHETIC"
    else:
        try:
            snap = args.snapshot or datasets.active_snapshot()
            man = datasets.load_manifest(snap)
            manifest_path = os.path.join(datasets.snapshot_dir(snap), "manifest.json")
            meta["snapshot"] = snap
            meta["manifest_sha256"] = datasets.sha256_file(manifest_path)
            meta["dataset_fetched_utc"] = man.get("fetched_utc")
            meta["dataset_hashes"] = {
                k: v["sha256"] for k, v in man["markets"].items()
            }
            meta["snapshot_integrity"] = datasets.verify_snapshot(snap)["all_ok"]
        except Exception as e:  # noqa: BLE001
            meta["snapshot_error"] = str(e)
    with open(os.path.join(res_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n==================== RESULTS WRITTEN ====================")
    print("  snapshot                   :", meta.get("snapshot"))
    print("  results/comparison_all.csv :", len(comp), "rows")
    print("  results/dm_all.csv         :", len(dmall), "rows")
    print("  results/group_summary.csv  :", len(summary), "rows")
    print("  results/run_meta.json      : provenance recorded")
    print("\n--- conditional-coverage pass rate by group/method ---")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
