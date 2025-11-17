"""Filter general step1 dataset by latest-notice date before analysis.

This utility reads a step1 CSV (default: latest general step1 file), keeps only
rows whose `date_latest_notice` is on/after a specified cutoff (default:
2021-01-01), and writes the filtered dataframe to a timestamped CSV under
`data/step1_data/general_filtered`.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_GENERAL_DIR = PROJECT_ROOT / "data" / "step1_data" / "general"
OUTPUT_DIR = PROJECT_ROOT / "data" / "step1_data" / "general_filtered"
DEFAULT_DATE_COLUMN = "date_latest_notice"
DEFAULT_CUTOFF = "2021-01-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter general step1 data by date")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to step1 CSV. Defaults to the latest file under data/step1_data/general.",
    )
    parser.add_argument(
        "--date-column",
        default=DEFAULT_DATE_COLUMN,
        help="Name of the date column to filter on (default: date_latest_notice)",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_CUTOFF,
        help="Keep rows where date >= start-date (YYYY-MM-DD, default: 2021-01-01)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save filtered CSV (default: data/step1_data/general_filtered)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for reading/writing CSV files (default: utf-8-sig)",
    )
    return parser.parse_args()


def find_latest_step1() -> Path:
    csv_files = sorted(STEP1_GENERAL_DIR.glob("step1_general_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No general step1 files found in {STEP1_GENERAL_DIR}")
    return csv_files[-1]


def main() -> None:
    args = parse_args()
    input_path = args.input or find_latest_step1()
    print(f"[INFO] Loading data: {input_path}")
    df = pd.read_csv(input_path, encoding=args.encoding, parse_dates=[args.date_column])

    if args.date_column not in df.columns:
        raise KeyError(f"Column '{args.date_column}' not found in {input_path}")

    cutoff = pd.to_datetime(args.start_date)
    mask = df[args.date_column] >= cutoff
    filtered = df.loc[mask].copy()
    before, after = len(df), len(filtered)
    print(f"[INFO] Cutoff {args.start_date}: kept {after} of {before} rows")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(input_path).stem
    output_path = args.output_dir / f"{safe_source}_from_{args.start_date}_{timestamp}.csv"
    filtered.to_csv(output_path, index=False, encoding=args.encoding)
    print(f"[INFO] Saved filtered data to {output_path}")


if __name__ == "__main__":
    main()
