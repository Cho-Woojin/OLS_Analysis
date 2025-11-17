"""Correlation analysis grouped only by fasttrack vs general.

This script reads a Step 1 dataset (default: latest general step1 CSV), splits
rows into two classes (fast vs general) based on the presence of a fasttrack
flag, and computes Pearson correlations between a target column and all other
numeric columns (excluding latitude/longitude). Each class produces its own
correlation table saved under `data/correlation_result`.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_GENERAL_DIR = PROJECT_ROOT / "data" / "step1_data" / "general"
OUTPUT_DIR = PROJECT_ROOT / "data" / "correlation_result"
DEFAULT_FASTTRACK_COLUMN = "fasttrack_note"
DEFAULT_TARGET_COLUMN = "unit_total"
EXCLUDED_COLUMNS = {"lat", "lon"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correlation analysis grouped by fasttrack vs general"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a step1 CSV. Defaults to the most recent file in data/step1_data/general.",
    )
    parser.add_argument(
        "--fasttrack-column",
        default=DEFAULT_FASTTRACK_COLUMN,
        help="Column that indicates 신속통합 여부 (default: fasttrack_note)",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET_COLUMN,
        help="Numeric target column for correlation (default: unit_total)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for reading/writing CSV files (default: utf-8-sig)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory to save correlation tables (default: data/correlation_result)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum non-null sample size required to compute correlation (default: 3)",
    )
    return parser.parse_args()


def find_latest_step1() -> Path:
    csv_files = sorted(STEP1_GENERAL_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No step1 CSV files found under {STEP1_GENERAL_DIR}")
    return csv_files[-1]


def prepare_class_labels(series: pd.Series) -> pd.Series:
    normalized = series.fillna("").astype("string").str.strip()
    return normalized.apply(lambda text: "fast" if text else "general")


def numeric_columns(df: pd.DataFrame, target: str) -> List[str]:
    numeric_df = df.select_dtypes(include=["number"])
    if target not in numeric_df.columns:
        raise KeyError(f"Target column '{target}' not found or not numeric in dataset")
    columns = [col for col in numeric_df.columns if col != target and col not in EXCLUDED_COLUMNS]
    if not columns:
        raise ValueError("No numeric columns available for correlation analysis")
    return columns


def compute_correlations(
    df: pd.DataFrame,
    target: str,
    candidate_columns: Iterable[str],
    min_samples: int,
) -> pd.DataFrame:
    target_series = df[target]
    records: List[Dict[str, object]] = []
    for col in candidate_columns:
        pair = pd.concat([target_series, df[col]], axis=1, keys=[target, col]).dropna()
        n = len(pair)
        corr_value = pd.NA
        if n >= min_samples and pair[target].nunique() > 1 and pair[col].nunique() > 1:
            corr_value = pair[target].corr(pair[col])
        t_stat = pd.NA
        p_value = pd.NA
        if n >= max(min_samples, 3) and pd.notna(corr_value):
            dfree = n - 2
            denom = 1 - corr_value**2
            if denom <= 0:
                t_stat = math.copysign(math.inf, float(corr_value))
                p_value = 0.0
            else:
                t_stat_calc = float(corr_value) * math.sqrt(dfree / denom)
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


def save_output(df: pd.DataFrame, output_dir: Path, class_label: str, source_name: str, encoding: str, target: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"correlation_{target}_{class_label}_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"[INFO] Saved correlation table for '{class_label}' -> {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    csv_path = args.input or find_latest_step1()
    print(f"[INFO] Loading data: {csv_path}")
    df = pd.read_csv(csv_path, encoding=args.encoding)

    for col in (args.fasttrack_column, args.target_column):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in {csv_path}")

    df["class_label"] = prepare_class_labels(df[args.fasttrack_column])

    try:
        candidate_cols = numeric_columns(df, args.target_column)
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"[ERROR] {exc}")

    source_name = csv_path.name
    for class_label, subset in df.groupby("class_label"):
        print(f"[INFO] Processing class '{class_label}' with {len(subset)} rows")
        corr_df = compute_correlations(subset, args.target_column, candidate_cols, args.min_samples)
        save_output(corr_df, args.output_dir, class_label, source_name, args.encoding, args.target_column)

    print("[INFO] Completed class-level correlation analysis")


if __name__ == "__main__":
    main()
