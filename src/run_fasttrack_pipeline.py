"""One-click pipeline for fasttrack datasets.

This script automates the typical workflow for 신속통합기획 사례:
1. Step1 preprocessing (data_preprocessing.py --pipeline fasttrack)
2. Step2 feature engineering (feature_engineering.py)
3. Basic statistics generation (basic_statistics.py)

It reports the produced file paths so downstream analysis can continue immediately.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RAW_FASTTRACK_DIR = PROJECT_ROOT / "data" / "raw_data" / "fasttrack"
STEP1_FASTTRACK_DIR = PROJECT_ROOT / "data" / "step1_data" / "fasttrack"
STEP2_DIR = PROJECT_ROOT / "data" / "step2_data"
BASIC_DIR = PROJECT_ROOT / "data" / "basic_statistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fasttrack preprocessing + feature engineering + basic stats"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        help="Path to the raw fasttrack CSV. Defaults to the newest file under data/raw_data/fasttrack.",
    )
    parser.add_argument(
        "--encoding",
        default="cp949",
        help="Encoding hint passed to data_preprocessing (default: cp949)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show stdout/stderr from each underlying command.",
    )
    return parser.parse_args()


def _latest_csv(directory: Path) -> Path:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[-1]


def _run_command(cmd: list[str], verbose: bool) -> None:
    print(f"[CMD] {' '.join(str(part) for part in cmd)}")
    subprocess.run(cmd, check=True, capture_output=not verbose)


def _detect_newest(directory: Path, previous: Optional[Path]) -> Path:
    latest = _latest_csv(directory)
    if previous is None or latest != previous:
        return latest
    return latest


def run_pipeline(raw_csv: Optional[Path], encoding: str, verbose: bool) -> None:
    raw_path = raw_csv or _latest_csv(RAW_FASTTRACK_DIR)
    print(f"[INFO] Using raw CSV: {raw_path}")

    STEP1_FASTTRACK_DIR.mkdir(parents=True, exist_ok=True)
    STEP2_DIR.mkdir(parents=True, exist_ok=True)
    BASIC_DIR.mkdir(parents=True, exist_ok=True)

    prev_step1 = _latest_csv(STEP1_FASTTRACK_DIR) if list(STEP1_FASTTRACK_DIR.glob("*.csv")) else None
    cmd_pre = [
        sys.executable,
        str(SRC_DIR / "data_preprocessing.py"),
        "--pipeline",
        "fasttrack",
        "--input",
        str(raw_path),
        "--encoding",
        encoding,
    ]
    _run_command(cmd_pre, verbose)
    step1_csv = _detect_newest(STEP1_FASTTRACK_DIR, prev_step1)
    print(f"[INFO] Step1 output: {step1_csv}")

    prev_step2 = _latest_csv(STEP2_DIR) if list(STEP2_DIR.glob("*.csv")) else None
    cmd_step2 = [
        sys.executable,
        str(SRC_DIR / "feature_engineering.py"),
        "--input",
        str(step1_csv),
    ]
    _run_command(cmd_step2, verbose)
    step2_csv = _detect_newest(STEP2_DIR, prev_step2)
    print(f"[INFO] Step2 output: {step2_csv}")

    cmd_stats = [
        sys.executable,
        str(SRC_DIR / "basic_statistics.py"),
        "--input",
        str(step2_csv),
    ]
    _run_command(cmd_stats, verbose)
    latest_stat = _latest_csv(BASIC_DIR)
    print(f"[INFO] Basic statistics saved: {latest_stat}")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.raw, args.encoding, args.verbose)
