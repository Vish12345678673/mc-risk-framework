"""Visualization helpers. All headless (Agg) and save to PNG."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def plot_var_breaches(run, outfile="var_breaches.png"):
    """Realised returns vs VaR forecast, breaches highlighted."""
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(run.realised))
    ax.plot(x, run.realised, color="#4a5568", lw=0.6, label="realised return")
    ax.plot(x, run.var, color="#2b6cb0", lw=1.2,
            label=f"VaR {int((1-run.alpha)*100)}%")
    b = run.breaches.astype(bool)
    ax.scatter(x[b], run.realised[b], color="#c53030", s=14, zorder=5,
               label=f"breaches ({b.sum()})")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{run.model_name} — VaR backtest "
                 f"(breach rate {run.coverage.breach_rate:.3f} vs {run.alpha}, "
                 f"CC p={run.coverage.cc_pvalue:.3f})")
    ax.set_xlabel("out-of-sample day")
    ax.set_ylabel("return")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    return outfile


def plot_equity_fan(equity_paths, outfile="equity_fan.png", title="Equity curves"):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    steps = np.arange(equity_paths.shape[1])
    for lo, hi, a in [(5, 95, 0.12), (25, 75, 0.20)]:
        ax.fill_between(steps, np.percentile(equity_paths, lo, axis=0),
                        np.percentile(equity_paths, hi, axis=0),
                        alpha=a, color="#2b6cb0", label=f"{lo}-{hi}%")
    ax.plot(steps, np.percentile(equity_paths, 50, axis=0),
            color="#1a365d", lw=2, label="median")
    for i in range(min(30, len(equity_paths))):
        ax.plot(steps, equity_paths[i], color="grey", lw=0.3, alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("period")
    ax.set_ylabel("equity (V0=1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    return outfile


def plot_drawdown_dist(dd_samples, outfile="drawdown_dist.png"):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(dd_samples * 100, bins=60, color="#2b6cb0", alpha=0.8)
    ax.axvline(np.median(dd_samples) * 100, color="#1a365d", lw=1.5,
               label="median")
    ax.axvline(np.percentile(dd_samples, 95) * 100, color="#c53030", lw=1.5,
               label="95th pct")
    ax.set_title("Max-drawdown distribution")
    ax.set_xlabel("max drawdown (%)")
    ax.set_ylabel("frequency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=130)
    plt.close(fig)
    return outfile
