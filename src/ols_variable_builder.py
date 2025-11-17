"""Derive regression-ready variables from step1 datasets.

This script loads a cleaned Step1 CSV (fasttrack or general), builds
log/ratio/weighted FAR features required for OLS, expands region dummies, and
saves the processed table under data/ols_result with the naming pattern
`<source>_preprossed_ols_variable.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = PROJECT_ROOT / "data" / "step1_data"
OLS_DIR = PROJECT_ROOT / "data" / "ols_result"
REGION_ORDER: Tuple[str, ...] = ("동남권", "서남권", "서북권", "동북권", "도심권")
ZONE_INDICES = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build regression variables from step1 CSV")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a step1 CSV. Defaults to the newest file under data/step1_data.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for reading/writing CSV files (default: utf-8-sig)",
    )
    return parser.parse_args()


def find_latest_step1() -> Path:
    candidates = sorted(STEP1_DIR.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No step1 CSV files found in data/step1_data")
    return candidates[-1]


def load_dataframe(csv_path: Path, encoding: str) -> pd.DataFrame:
    print(f"[INFO] Loading step1 data: {csv_path}")
    return pd.read_csv(csv_path, encoding=encoding)


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for OLS variable builder: {missing}")


def compute_public_space_ratio(df: pd.DataFrame) -> pd.Series:
    numerator = df[["area_road", "area_park", "area_public_open"]].sum(axis=1, min_count=1)
    denom = df["area_total"].astype("Float64")
    ratio = numerator / denom
    ratio[(denom <= 0) | denom.isna()] = pd.NA
    return ratio.astype("Float64")


def compute_public_infra_ratio(df: pd.DataFrame) -> pd.Series:
    numerator = df[["area_road", "area_park", "area_public_open", "area_green"]].sum(
        axis=1, min_count=1
    )
    denom = df["area_total"].astype("Float64")
    ratio = numerator / denom
    ratio[(denom <= 0) | denom.isna()] = pd.NA
    return ratio.astype("Float64")


def compute_rent_ratio(df: pd.DataFrame) -> pd.Series:
    rent = df["unit_rent_total"].astype("Float64")
    total = df["unit_total"].astype("Float64")
    ratio = rent / total
    ratio[(total <= 0) | total.isna() | rent.isna()] = pd.NA
    return ratio.astype("Float64")


def compute_rent_small_ratio(df: pd.DataFrame) -> pd.Series:
    small = df["unit_rent_s"].astype("Float64")
    rent_total = df["unit_rent_total"].astype("Float64")
    ratio = small / rent_total
    ratio[(rent_total <= 0) | rent_total.isna() | small.isna()] = pd.NA
    return ratio.astype("Float64")


def compute_incentive_metrics(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    incentive_values = []
    baseline_values = []
    for _, row in df.iterrows():
        incentive, baseline = _per_row_incentive(row)
        incentive_values.append(incentive)
        baseline_values.append(baseline)
    return pd.Series(incentive_values, dtype="Float64"), pd.Series(baseline_values, dtype="Float64")


def _per_row_incentive(row: pd.Series) -> tuple[pd._libs.missing.NAType | float, pd._libs.missing.NAType | float]:
    total_area = 0.0
    weighted_diff = 0.0
    weighted_base = 0.0
    diff_valid = False
    base_valid = False
    for idx in ZONE_INDICES:
        area = row.get(f"zone{idx}_area")
        base = row.get(f"zone{idx}_far_base")
        plan = row.get(f"zone{idx}_far_plan")
        if pd.isna(area) or area <= 0:
            continue
        area = float(area)
        total_area += area
        if pd.notna(base):
            weighted_base += area * float(base)
            base_valid = True
        if pd.notna(plan) and pd.notna(base):
            weighted_diff += area * (float(plan) - float(base))
            diff_valid = True
    if total_area == 0:
        return pd.NA, pd.NA
    incentive = weighted_diff / total_area if diff_valid else pd.NA
    baseline = weighted_base / total_area if base_valid else pd.NA
    return incentive, baseline


def add_region_dummies(df: pd.DataFrame) -> pd.DataFrame:
    dummy = pd.get_dummies(df["region"], prefix="region_dummy")
    for region in REGION_ORDER:
        column = f"region_dummy_{region}"
        if column not in dummy.columns:
            dummy[column] = 0
    dummy = dummy[[f"region_dummy_{region}" for region in REGION_ORDER]]
    return pd.concat([df, dummy.astype("Int64")], axis=1)


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    required_numeric = [
        "log_area_total",
        "log_unit_total",
        "floors",
        "height",
        "area_road",
        "area_park",
        "area_public_open",
        "area_green",
        "area_total",
        "unit_rent_total",
        "unit_total",
        "unit_rent_s",
        "zone1_area",
        "zone1_far_base",
        "zone1_far_plan",
    ]
    ensure_columns(df, required_numeric)
    df = add_region_dummies(df)
    if "duration_months_initial" not in df.columns and "duration_days_initial" in df.columns:
        df["duration_months_initial"] = (
            df["duration_days_initial"].astype("Float64") / 30.0
        ).round(3)
    df["public_space_ratio"] = compute_public_space_ratio(df)
    df["public_infra_ratio"] = compute_public_infra_ratio(df)
    df["rent_ratio"] = compute_rent_ratio(df)
    df["rent_small_ratio"] = compute_rent_small_ratio(df)
    incentive, baseline = compute_incentive_metrics(df)
    df["incentive_weighted"] = incentive
    df["baseline_far"] = baseline

    selected_cols = [
        "PRESENT_SN",
        "DGM_NM",
        "region",
        "duration_days_initial",
        "duration_months_initial",
        "log_area_total",
        "log_unit_total",
        "floors",
        "height",
        "public_space_ratio",
        "rent_ratio",
        "rent_small_ratio",
        "incentive_weighted",
        "baseline_far",
        "public_infra_ratio",
    ] + [f"region_dummy_{region}" for region in REGION_ORDER]

    selected_cols = [col for col in selected_cols if col in df.columns]
    return df[selected_cols]


def save_output(df: pd.DataFrame, source_path: Path, encoding: str) -> Path:
    OLS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OLS_DIR / f"{source_path.stem}_preprossed_ols_variable.csv"
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"[INFO] Saved OLS variable table to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    input_csv = args.input or find_latest_step1()
    df_step1 = load_dataframe(input_csv, args.encoding)
    output_df = build_output(df_step1)
    save_output(output_df, input_csv, args.encoding)


if __name__ == "__main__":
    main()
