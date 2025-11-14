"""Step 3 basic statistics generator.

Loads the latest step2 dataset (or a user-provided CSV), identifies numeric
columns, computes descriptive statistics (min, max, mean, std) for each, and
writes the summary table to `data/basic_statistic`.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP2_DIR = PROJECT_ROOT / "data" / "step2_data"
BASIC_DIR = PROJECT_ROOT / "data" / "basic_statistic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute basic statistics from step2 data")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a step2 CSV. Defaults to the most recent file in data/step2_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASIC_DIR,
        help="Directory to save the statistics table (default: data/basic_statistic)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding used for both reading and writing CSV files (default: utf-8-sig)",
    )
    return parser.parse_args()


def find_latest_csv(directory: Path) -> Path:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[-1]


def load_dataframe(csv_path: Path, encoding: str) -> pd.DataFrame:
    print(f"[INFO] Loading step2 data: {csv_path}")
    return pd.read_csv(csv_path, encoding=encoding)


def numeric_columns(df: pd.DataFrame) -> List[str]:
    numeric_df = df.select_dtypes(include=["number"])
    cols = list(numeric_df.columns)
    if not cols:
        raise ValueError("No numeric columns found; unable to compute statistics")
    return cols


def compute_statistics(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    stats = df[columns].agg(["min", "max", "mean", "std"]).T
    stats = stats.rename(columns={"min": "min", "max": "max", "mean": "mean", "std": "std"})
    stats.index.name = "feature"
    stats = stats.reset_index()
    return stats


def save_output(df: pd.DataFrame, output_dir: Path, source_name: str, encoding: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"basic_statistic_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"[INFO] Saved statistics table to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    input_csv = args.input or find_latest_csv(STEP2_DIR)
    df_step2 = load_dataframe(input_csv, args.encoding)
    cols = numeric_columns(df_step2)
    print(f"[INFO] Numeric columns detected ({len(cols)}): {', '.join(cols)}")
    stats = compute_statistics(df_step2, cols)
    save_output(stats, args.output_dir, input_csv.name, args.encoding)


if __name__ == "__main__":
    main()
