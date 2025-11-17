"""Step 4 correlation analysis between duration and numeric metrics.

Loads the latest step2 dataset, performs Pearson correlation between
`duration_months_initial` and every other numeric column, and reports the
correlation coefficient, t-statistic, and two-tailed p-value (t-test). Results
are saved in `data/correlation_result`.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP2_DIR = PROJECT_ROOT / "data" / "step2_data"
CORR_DIR = PROJECT_ROOT / "data" / "correlation_result"
TARGET_COLUMN = "duration_months_initial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlation analysis for step2 data")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a step2 CSV. Defaults to the most recent file in data/step2_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CORR_DIR,
        help="Directory to save correlation results (default: data/correlation_result)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for reading/writing CSV files (default: utf-8-sig)",
    )
    parser.add_argument(
        "--mode",
        choices=("target", "matrix"),
        default="target",
        help="'target' computes correlations vs duration_months_initial (default). 'matrix' computes the full numeric correlation matrix.",
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
    if TARGET_COLUMN not in numeric_df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found or not numeric.")
    cols = [c for c in numeric_df.columns if c != TARGET_COLUMN]
    if not cols:
        raise ValueError("No numeric columns available for correlation analysis.")
    return cols


def numeric_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"]).copy()
    numeric_df = numeric_df.dropna(axis=1, how="all")
    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation matrix analysis.")
    return numeric_df


def compute_full_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = numeric_dataframe(df)
    matrix = numeric_df.corr(method="pearson")
    matrix.index.name = "feature"
    return matrix


def correlation_with_t_test(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    target = df[TARGET_COLUMN]
    records = []
    for col in columns:
        pair = pd.concat([target, df[col]], axis=1, keys=[TARGET_COLUMN, col]).dropna()
        n = len(pair)
        corr_value = pd.NA
        if n >= 2:
            if pair[TARGET_COLUMN].nunique() > 1 and pair[col].nunique() > 1:
                corr_value = pair[TARGET_COLUMN].corr(pair[col])
        t_stat = pd.NA
        p_value = pd.NA
        if n >= 3 and pd.notna(corr_value):
            dfree = n - 2
            denom = 1 - corr_value**2
            if denom <= 0:
                # Perfect correlation; treat as infinite t-stat with zero p-value
                t_stat = math.copysign(math.inf, corr_value)
                p_value = 0.0
            else:
                t_stat_calc = corr_value * math.sqrt(dfree / denom)
                p_value_calc = stats.t.sf(abs(t_stat_calc), dfree) * 2
                t_stat = t_stat_calc
                p_value = p_value_calc
        records.append(
            {
                "feature": col,
                "sample_size": n,
                "correlation": corr_value,
                "t_stat": t_stat,
                "p_value": p_value,
            }
        )
    result = pd.DataFrame(records)
    result = result.sort_values(by="correlation", key=lambda s: s.abs(), ascending=False)
    return result.reset_index(drop=True)


def save_output(df: pd.DataFrame, output_dir: Path, source_name: str, encoding: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"correlation_duration_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"[INFO] Saved correlation table to {output_path}")
    return output_path


def save_matrix(df: pd.DataFrame, output_dir: Path, source_name: str, encoding: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"correlation_matrix_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, encoding=encoding)
    print(f"[INFO] Saved correlation matrix to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    input_csv = args.input or find_latest_csv(STEP2_DIR)
    df_step2 = load_dataframe(input_csv, args.encoding)
    if args.mode == "target":
        cols = numeric_columns(df_step2)
        print(f"[INFO] Running correlations against '{TARGET_COLUMN}' for {len(cols)} features.")
        corr_table = correlation_with_t_test(df_step2, cols)
        save_output(corr_table, args.output_dir, input_csv.name, args.encoding)
    else:
        matrix = compute_full_matrix(df_step2)
        print(f"[INFO] Running full correlation matrix for {matrix.shape[0]} numeric features.")
        save_matrix(matrix, args.output_dir, input_csv.name, args.encoding)


if __name__ == "__main__":
    main()
