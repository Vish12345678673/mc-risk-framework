#!/usr/bin/env python
"""
Generate the full set of output figures into figures/ :
  * VaR-breach diagnostic for every engine
  * equity fan + drawdown distribution
  * comparison table, calibration rankings, and pairwise DM table as images

Run:  python figures/generate_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcrisk import data, compare_models, rank_by_calibration  # noqa: E402
from mcrisk.backtest.scoring import pairwise_dm_table  # noqa: E402
from mcrisk.simulate import simulate_equity  # noqa: E402
from mcrisk.metrics import drawdown_distribution, risk_of_ruin  # noqa: E402
from mcrisk.viz import plot_var_breaches, plot_equity_fan, plot_drawdown_dist  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))


def render_table(df: pd.DataFrame, title: str, fname: str):
    """Render a DataFrame as a clean table image."""
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(min(2 + 1.5 * n_cols, 18),
                                    1.1 + 0.45 * n_rows))
    ax.axis("off")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=14, loc="left")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1a365d")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#edf2f7")
        cell.set_edgecolor("#cbd5e0")
    fig.tight_layout()
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)
    return path


def main():
    print("[data] synthetic regime-switching returns (n=2200)")
    rets = data.synth_regime_switching(n=2200, seed=0)

    print("[run] rolling comparison (step=6 for tractable HMM refits)...")
    table, runs = compare_models(rets, alphas=(0.05, 0.01), window=450, step=6)

    # --- table images ---
    cols = ["model", "alpha", "n", "breaches", "breach_rate", "expected",
            "mean_VaR", "kupiec_p", "christ_ind_p", "cc_p", "es_p"]
    render_table(table[cols], "VaR/CVaR comparison — out-of-sample backtest "
                 "(synthetic regime data)", "01_comparison_table.png")
    render_table(rank_by_calibration(table, 0.05),
                 "Calibration ranking @ alpha = 5%", "02_ranking_5pct.png")
    render_table(rank_by_calibration(table, 0.01),
                 "Calibration ranking @ alpha = 1% (deep tail)",
                 "03_ranking_1pct.png")
    render_table(pairwise_dm_table(runs, 0.05),
                 "Pairwise Diebold–Mariano (pinball loss) @ 5%",
                 "04_dm_pairwise_5pct.png")

    # --- VaR-breach diagnostics for every engine @5% ---
    for i, name in enumerate(["gaussian", "historical", "student_t",
                              "fhs_ewma", "hmm_regime"], start=5):
        run = runs.get((name, 0.05))
        if run is not None:
            plot_var_breaches(run, os.path.join(OUT, f"{i:02d}_var_breaches_{name}.png"))
            print("saved:", f"{i:02d}_var_breaches_{name}.png")

    # --- equity-curve risk ---
    eq = simulate_equity(rets, method="block_bootstrap", horizon=252,
                         n_sims=10000, seed=42)
    plot_equity_fan(eq, os.path.join(OUT, "10_equity_fan.png"),
                    title="Block-bootstrap equity curves (1y, 10k sims)")
    print("saved: 10_equity_fan.png")
    dd = drawdown_distribution(eq)
    plot_drawdown_dist(dd["samples"], os.path.join(OUT, "11_drawdown_dist.png"))
    print("saved: 11_drawdown_dist.png")
    print(f"[risk] ruin(-50%)={risk_of_ruin(eq, 0.5):.3%} "
          f"medianMDD={dd['median']:.2%} p95MDD={dd['p95']:.2%}")


if __name__ == "__main__":
    main()
