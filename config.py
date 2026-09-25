"""Paths and month splits. Train is Aug 2025–Jun 2026, val July 2026, test August 2026."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "rawdata" / "filtered"
DATA = ROOT / "data"
ENGINE = ROOT / "engines" / "stockfish.exe"
RUNS = ROOT / "runs"
BENCH_NODES = 2_497_913          # Stockfish 19 `bench` must print this

def month_of(path):
    return Path(path).name.split("_")[1]   # lichess_2025-08_filtered.parquet

def month_files(split):
    files = sorted(RAW.glob("lichess_*_filtered.parquet"))
    def keep(path):
        ym = month_of(path)
        if split == "train": return "2025-08" <= ym <= "2026-06"
        if split == "val":   return ym == "2026-07"
        if split == "test":  return ym == "2026-08"
        raise SystemExit(f"unknown split {split}")
    return [p for p in files if keep(p)]
