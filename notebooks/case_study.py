#!/usr/bin/env python
"""
Reproducible case study (no network required).

Runs the full comparison on three synthetic data-generating processes that
stand in for different market characters, then prints the comparison + pairwise
Diebold-Mariano tables. This is the skeleton you replicate on real
developed/emerging series for the paper.

  gaussian  -> thin-tailed, no clustering  (Gaussian baseline should win)
  student_t -> fat-tailed, no clustering    (Student-t / FHS should win)
  regime    -> regime-switching + clustering (FHS strong; HMM should help in
               the deep 1% tail)

Run:  python notebooks/case_study.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcrisk import data, compare_models, rank_by_calibration  # noqa: E402
from mcrisk.backtest.scoring import pairwise_dm_table  # noqa: E402

pd.set_option("display.width", 170, "display.max_columns", 25)

DGPS = {
    "gaussian": lambda: data.synth_gaussian(n=2500, seed=0),
    "student_t": lambda: data.synth_student_t(n=2500, nu=4.0, seed=0),
    "regime": lambda: data.synth_regime_switching(n=2500, seed=0),
}


def main():
    for name, gen in DGPS.items():
        print("\n" + "=" * 78)
        print(f"DATA-GENERATING PROCESS: {name}")
        print("=" * 78)
        rets = gen()
        # step>1 keeps the HMM refits tractable; use step=1 for final results.
        table, runs = compare_models(rets, alphas=(0.05, 0.01),
                                     window=500, step=5)
        print(table[["model", "alpha", "breach_rate", "kupiec_p",
                     "christ_ind_p", "cc_p", "es_p", "verdict"]]
              .to_string(index=False))
        print(f"\n  best-calibrated @1% tail on '{name}':")
        print(rank_by_calibration(table, 0.01).head(3).to_string(index=False))
        print(f"\n  pairwise Diebold-Mariano (pinball loss) @5% on '{name}':")
        print(pairwise_dm_table(runs, 0.05).to_string(index=False))


if __name__ == "__main__":
    main()
