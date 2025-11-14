"""Step 2 feature engineering script.

Loads the latest step1 dataset (or a user-provided CSV), creates derived
variables required for downstream analysis, and stores the enriched dataset in
`data/step2_data` with a timestamped filename.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP1_DIR = PROJECT_ROOT / "data" / "step1_data"
STEP2_DIR = PROJECT_ROOT / "data" / "step2_data"
ZONE_INDICES: Tuple[int, ...] = (1, 2, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate derived variables (step 2)")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to a step1 CSV. Defaults to the most recent file in data/step1_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STEP2_DIR,
        help="Directory for the enriched CSV (default: data/step2_data)",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Encoding for both reading and writing CSV files (default: utf-8-sig)",
    )
    return parser.parse_args()


def find_latest_csv(directory: Path) -> Path:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[-1]


def load_dataframe(csv_path: Path, encoding: str) -> pd.DataFrame:
    print(f"[INFO] Loading step1 data: {csv_path}")
    return pd.read_csv(csv_path, encoding=encoding)


def add_duration_months(df: pd.DataFrame) -> pd.Series:
    days = df["duration_days_initial"].astype("Float64")
    months = days / 30.0
    return months.round(3)


def _best_far(row: pd.Series, index: int):
    for suffix in ("far_plan", "far_base", "far_legal"):
        value = row.get(f"zone{index}_{suffix}")
        if pd.notna(value):
            return value
    return pd.NA


def compute_area_weighted_far(df: pd.DataFrame) -> pd.Series:
    def _per_row(row: pd.Series):
        total_area = 0.0
        weighted_sum = 0.0
        for idx in ZONE_INDICES:
            area = row.get(f"zone{idx}_area")
            if pd.isna(area) or area == 0:
                continue
            far_value = _best_far(row, idx)
            if pd.isna(far_value):
                continue
            total_area += float(area)
            weighted_sum += float(area) * float(far_value)
        if total_area == 0:
            return pd.NA
        return weighted_sum / total_area

    result = df.apply(_per_row, axis=1)
    return result.astype("Float64")


def compute_rental_ratio(df: pd.DataFrame) -> pd.Series:
    rent = df["unit_rent_total"].astype("Float64")
    total = df["unit_total"].astype("Float64")
    ratio = rent / total
    ratio[(total <= 0) | total.isna() | rent.isna()] = pd.NA
    return ratio.astype("Float64")


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_months_initial"] = add_duration_months(df)
    df["zone_weighted_far"] = compute_area_weighted_far(df)
    df["unit_rent_ratio"] = compute_rental_ratio(df)
    return df


def save_output(df: pd.DataFrame, output_dir: Path, source_name: str, encoding: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"step2_features_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, index=False, encoding=encoding)
    print(f"[INFO] Saved enriched data to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    input_csv = args.input or find_latest_csv(STEP1_DIR)
    df_step1 = load_dataframe(input_csv, args.encoding)
    df_enriched = enrich_dataframe(df_step1)
    print("[INFO] Derived columns added: duration_months_initial, zone_weighted_far, unit_rent_ratio")
    save_output(df_enriched, args.output_dir, input_csv.name, args.encoding)


if __name__ == "__main__":
    main()
