"""
Frozen-dataset layer: the reproducibility boundary.

Research runs read market data only through load_frozen(), which reads CSV
snapshots from disk and never touches the network. Live fetching lives in
scripts/ingest.py and is gated behind MCRISK_DATA_MODE=live, so a normal batch
run cannot silently change because Yahoo data moved.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

import pandas as pd

# Single source of truth for the cross-market universe.
# label -> (yfinance symbol, group)
MARKETS = {
    "SP500": ("^GSPC", "developed"),
    "DAX": ("^GDAXI", "developed"),
    "FTSE100": ("^FTSE", "developed"),
    "Nikkei": ("^N225", "developed"),
    "Nifty50": ("^NSEI", "emerging"),
    "Sensex": ("^BSESN", "emerging"),
    "Bovespa": ("^BVSP", "emerging"),
    "JSE40": ("^J200.JO", "emerging"),
}

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("MCRISK_DATA_DIR", os.path.join(_PKG_ROOT, "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
ACTIVE_FILE = os.path.join(DATA_DIR, "ACTIVE_SNAPSHOT")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_PKG_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def snapshot_dir(snapshot: str) -> str:
    return os.path.join(RAW_DIR, snapshot)


def list_snapshots() -> list[str]:
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(
        d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))
    )


def active_snapshot() -> str:
    """Return the snapshot research code reads by default."""
    if os.path.isfile(ACTIVE_FILE):
        with open(ACTIVE_FILE) as f:
            snapshot = f.read().strip()
        if snapshot:
            return snapshot
    snapshots = list_snapshots()
    if not snapshots:
        raise FileNotFoundError(
            "No frozen dataset found. Run `MCRISK_DATA_MODE=live "
            "python scripts/ingest.py` once to create one."
        )
    return snapshots[-1]


def set_active_snapshot(snapshot: str) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ACTIVE_FILE, "w") as f:
        f.write(snapshot + "\n")


def write_snapshot(
    snapshot: str, series: dict[str, pd.Series], fetch_meta: dict
) -> str:
    """Write one CSV per market plus a manifest with hashes and provenance."""
    dest = snapshot_dir(snapshot)
    os.makedirs(dest, exist_ok=True)
    entries = {}
    for label, values in series.items():
        path = os.path.join(dest, f"{label}.csv")
        values = values.dropna().sort_index()
        values.to_frame("close").to_csv(path, index_label="date")
        symbol, group = MARKETS.get(label, (label, "unknown"))
        entries[label] = {
            "file": f"{label}.csv",
            "symbol": symbol,
            "group": group,
            "rows": int(len(values)),
            "start": str(values.index.min().date()),
            "end": str(values.index.max().date()),
            "sha256": sha256_file(path),
        }

    manifest = {
        "snapshot": snapshot,
        "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_commit": git_commit(),
        "markets": entries,
        **fetch_meta,
    }
    with open(os.path.join(dest, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return dest


def load_manifest(snapshot: str | None = None) -> dict:
    snapshot = snapshot or active_snapshot()
    path = os.path.join(snapshot_dir(snapshot), "manifest.json")
    with open(path) as f:
        return json.load(f)


def load_frozen(label: str, snapshot: str | None = None) -> pd.Series:
    """Read a market price series from the frozen snapshot. Never fetches."""
    snapshot = snapshot or active_snapshot()
    path = os.path.join(snapshot_dir(snapshot), f"{label}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No frozen data for '{label}' in snapshot {snapshot} ({path}). "
            "Run ingest to create it."
        )
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    values = df["close"].astype(float).dropna().sort_index()
    values.name = label
    return values


def verify_snapshot(snapshot: str | None = None) -> dict:
    """Recompute file hashes and compare them to the manifest."""
    snapshot = snapshot or active_snapshot()
    manifest = load_manifest(snapshot)
    dest = snapshot_dir(snapshot)
    results = {}
    for label, meta in manifest["markets"].items():
        path = os.path.join(dest, meta["file"])
        ok = os.path.isfile(path) and sha256_file(path) == meta["sha256"]
        results[label] = bool(ok)
    return {
        "snapshot": snapshot,
        "all_ok": all(results.values()),
        "per_market": results,
    }
