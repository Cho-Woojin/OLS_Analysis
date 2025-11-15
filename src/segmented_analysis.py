"""Segmented basic statistics and correlation analysis for general datasets.

This script filters the latest general step1 CSV (or a user-provided file)
into subgroups based on the `region` column (권역) and whether the
`fasttrack_note` column indicates a 신속통합기획 사업장. For each subgroup,
we compute:

1. Basic statistics (min, max, mean, std) for numeric columns (lat/lon excluded).
2. Pearson correlation between a target numeric column (default: unit_total) and
   all other numeric columns in the subset, along with t-statistics and p-values.

Outputs are saved to the same directories as the existing pipeline:
- data/basic_statistic
- data/correlation_result

Each file is tagged with a label such as `fast_동남권` or `general_도심권` to
make the subgroup explicit.
"""
from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_GENERAL_DIR = PROJECT_ROOT / "data" / "step1_data" / "general"
BASIC_DIR = PROJECT_ROOT / "data" / "basic_statistic"
CORR_DIR = PROJECT_ROOT / "data" / "correlation_result"
EXCLUDED_COLUMNS = {"lat", "lon"}
DEFAULT_TARGET = "unit_total"
DEFAULT_FASTTRACK_COLUMN = "fasttrack_note"
DEFAULT_REGION_COLUMN = "region"


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segmented basic statistics and correlation analysis"
    )
    parser.add_argument(
        "--step1-input",
        type=Path,
        help="Path to a step1 CSV. Defaults to the most recent file in data/step1_data/general.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for reading and writing CSV files (default: utf-8-sig)",
    )
    parser.add_argument(
        "--fasttrack-column",
        default=DEFAULT_FASTTRACK_COLUMN,
        help="Column that indicates 신속통합기획 구분 (default: fasttrack_note)",
    )
    parser.add_argument(
        "--region-column",
        default=DEFAULT_REGION_COLUMN,
        help="Column containing 권역 정보 (default: region)",
    )
    parser.add_argument(
        "--target-column",
        default=DEFAULT_TARGET,
        help="Numeric column used as correlation target (default: unit_total)",
    )
    return parser


def find_latest_step1(directory: Path) -> Path:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[-1]


def sanitize_label(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    sanitized = re.sub(r"[^0-9A-Za-z가-힣_]+", "", compact)
    return sanitized or "unknown"


def classification_label(value) -> str:
    if value is None or value is pd.NA:
        return "general"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "<na>"}:
        return "general"
    return "fast"


def compute_basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.select_dtypes(include=["number"]).columns
        if col not in EXCLUDED_COLUMNS
    ]
    if not numeric_cols:
        raise ValueError("No numeric columns available for statistics")
    stats_df = df[numeric_cols].agg(["min", "max", "mean", "std"]).T
    stats_df = stats_df.rename(columns={"min": "min", "max": "max", "mean": "mean", "std": "std"})
    stats_df.index.name = "feature"
    return stats_df.reset_index()


def compute_correlations(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=["number"])
    if target_column not in numeric_df.columns:
        raise KeyError(
            f"Target column '{target_column}' not found or not numeric in subset"
        )
    candidate_cols = [
        col
        for col in numeric_df.columns
        if col not in EXCLUDED_COLUMNS and col != target_column
    ]
    if not candidate_cols:
        raise ValueError("No numeric columns available for correlation analysis")

    target = numeric_df[target_column]
    records: List[Dict[str, object]] = []
    for col in candidate_cols:
        pair = pd.concat([target, numeric_df[col]], axis=1)
        pair = pair.dropna()
        n = len(pair)
        corr_value = pd.NA
        if n >= 2 and pair[target_column].nunique() > 1 and pair[col].nunique() > 1:
            corr_value = pair[target_column].corr(pair[col])
        t_stat = pd.NA
        p_value = pd.NA
        if n >= 3 and pd.notna(corr_value):
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


def ensure_columns(df: pd.DataFrame, columns: Tuple[str, ...]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    step1_csv = args.step1_input or find_latest_step1(STEP1_GENERAL_DIR)
    print(f"[INFO] Loading step1 data: {step1_csv}")
    df = pd.read_csv(step1_csv, encoding=args.encoding)

    ensure_columns(df, (args.fasttrack_column, args.region_column, args.target_column))

    fasttrack_series = df[args.fasttrack_column].astype("string").str.strip()
    fasttrack_series = fasttrack_series.fillna("")
    region_series = df[args.region_column].astype("string").str.strip()
    region_series = region_series.fillna("미분류")
    df["__class"] = fasttrack_series.apply(classification_label)
    df["__region"] = region_series.mask(region_series == "", "미분류")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_tag = Path(step1_csv).stem

    BASIC_DIR.mkdir(parents=True, exist_ok=True)
    CORR_DIR.mkdir(parents=True, exist_ok=True)

    outputs: List[Tuple[str, Path, Path]] = []

    for (class_label, region_label), subset in df.groupby(["__class", "__region"]):
        subset = subset.copy()
        subset = subset.drop(columns=["__class", "__region"], errors="ignore")
        if subset.empty:
            continue
        display_label = f"{class_label}_{region_label}"
        safe_label = sanitize_label(display_label)
        print(f"[INFO] Processing subgroup '{display_label}' with {len(subset)} rows")

        try:
            stats_df = compute_basic_statistics(subset)
        except ValueError as exc:
            print(f"[WARN] Skipping stats for {display_label}: {exc}")
            continue
        stats_path = (
            BASIC_DIR
            / f"basic_statistic_{safe_label}_{source_tag}_{timestamp}.csv"
        )
        stats_df.to_csv(stats_path, index=False, encoding=args.encoding)

        try:
            corr_df = compute_correlations(subset, args.target_column)
        except (ValueError, KeyError) as exc:
            print(f"[WARN] Skipping correlation for {display_label}: {exc}")
            corr_path = None
        else:
            corr_path = (
                CORR_DIR
                / f"correlation_{args.target_column}_{safe_label}_{source_tag}_{timestamp}.csv"
            )
            corr_df.to_csv(corr_path, index=False, encoding=args.encoding)
        outputs.append((display_label, stats_path, corr_path))

    if not outputs:
        raise RuntimeError("No subgroup outputs were generated. Check input data.")

    print("[INFO] Completed segmented analysis. Outputs:")
    for label, stats_path, corr_path in outputs:
        print(f"    - {label}: stats -> {stats_path}")
        if corr_path is not None:
            print(f"                 corr  -> {corr_path}")
        else:
            print("                 corr  -> skipped")


if __name__ == "__main__":
    main()
