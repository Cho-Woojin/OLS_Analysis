"""Utility to backfill required columns in the raw CSV with synthetic-yet-plausible values.

The script inspects the latest raw data (or a user-provided file), applies deterministic
filler rules tailored to each required column, validates that all REQUIRED_TARGET_COLUMNS
are now populated, and saves a sibling CSV that mirrors the raw schema.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from data_preprocessing import (
    COLUMN_SPECS,
    RAW_DATA_DIR,
    REQUIRED_TARGET_COLUMNS,
)

SPEC_BY_SOURCE = {spec.source: spec for spec in COLUMN_SPECS}
REQUIRED_SOURCE_COLUMNS = [
    spec.source for spec in COLUMN_SPECS if spec.target in REQUIRED_TARGET_COLUMNS
]
DEFAULT_FASTTRACK_DATE = pd.Timestamp("2021-12-27")
DEFAULT_OUTPUT_NAME = "251114 신속통합기획_가라_DB.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic raw CSV with filled required fields.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR / "251114 신속통합기획_DB_1951.csv",
        help="Path to the source raw CSV (default: 공식 제공 1951 파일).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DATA_DIR / DEFAULT_OUTPUT_NAME,
        help="Destination CSV path (default: data/raw_data/251114 신속통합기획_가라_DB.csv)",
    )
    parser.add_argument(
        "--encoding",
        default="cp949",
        help="CSV encoding for both read/write operations (default: cp949).",
    )
    return parser.parse_args()


def read_raw_csv(path: Path, encoding: str) -> pd.DataFrame:
    errors: List[str] = []
    for candidate in dict.fromkeys([encoding, "cp949", "utf-8-sig"]):
        try:
            return pd.read_csv(path, encoding=candidate)
        except UnicodeDecodeError as exc:  # pragma: no cover - informative logging only
            errors.append(f"{candidate}: {exc}")
            continue
    details = "; ".join(errors)
    raise UnicodeDecodeError(
        "fallback",
        b"",
        0,
        1,
        f"Unable to decode {path} with tried encodings ({details})",
    )


def normalise_strings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    for column in cleaned.select_dtypes(include=["object", "string"]).columns:
        series = cleaned[column].astype("string").str.replace("\u3000", " ", regex=False)
        series = series.str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        cleaned[column] = series
    return cleaned


def to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype("string").str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace("%", "", regex=False)
    cleaned = cleaned.str.replace(" ", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


def to_datetime_series(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def compute_mode(series: pd.Series) -> str | None:
    data = series.dropna()
    if data.empty:
        return None
    modes = data.mode()
    if modes.empty:
        return None
    return str(modes.iloc[0])


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def fill_dgm_name(df: pd.DataFrame, logs: List[str]) -> None:
    column = "DGM_NM"
    missing = df[column].isna()
    if not missing.any():
        return
    filled = df[column].copy()
    counter = 1
    for idx in df.index[missing]:
        district = df.at[idx, "자치구"]
        prefix = district if isinstance(district, str) else "가라단지"
        filled.at[idx] = f"{prefix}_가라{counter:02d}"
        counter += 1
    df[column] = filled
    logs.append(f"{column}: filled {missing.sum()} rows with synthetic 단지명.")


def _extract_location_tokens(value: str | float | int | None) -> Tuple[str | None, str | None]:
    if not isinstance(value, str):
        return None, None
    tokens = value.replace(",", " ").split()
    district = next((tok for tok in tokens if tok.endswith("구")), None)
    dong = next((tok for tok in tokens if tok.endswith("동")), None)
    return district, dong


def fill_district(df: pd.DataFrame, logs: List[str]) -> None:
    column = "자치구"
    missing = df[column].isna()
    if not missing.any():
        return
    filled = df[column].copy()
    counter = 1
    for idx in df.index[missing]:
        location = df.at[idx, "정비구역 위치"]
        district, _ = _extract_location_tokens(location)
        if district is None:
            district = f"가상구{counter:02d}"
            counter += 1
        filled.at[idx] = district
    df[column] = filled
    logs.append(f"{column}: parsed/created {missing.sum()} districts from 위치 정보.")


def fill_dong(df: pd.DataFrame, logs: List[str]) -> None:
    column = "법정동"
    missing = df[column].isna()
    if not missing.any():
        return
    filled = df[column].copy()
    counter = 1
    for idx in df.index[missing]:
        location = df.at[idx, "정비구역 위치"]
        _, dong = _extract_location_tokens(location)
        if dong is None:
            district = df.at[idx, "자치구"]
            prefix = district if isinstance(district, str) else "가상"
            dong = f"{prefix}동{counter:02d}"
            counter += 1
        filled.at[idx] = dong
    df[column] = filled
    logs.append(f"{column}: derived {missing.sum()} entries using 위치/자치구.")


def fill_region(df: pd.DataFrame, logs: List[str]) -> None:
    column = "권역"
    missing = df[column].isna()
    if not missing.any():
        return
    mapping = (
        df.loc[~df[column].isna() & ~df["자치구"].isna()]
        .groupby("자치구")[column]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else pd.NA)
    )
    fallback = compute_mode(df[column]) or "기타권역"
    filled = df[column].copy()
    for idx in df.index[missing]:
        district = df.at[idx, "자치구"]
        candidate = mapping.get(district, fallback)
        filled.at[idx] = candidate
    df[column] = filled
    logs.append(f"{column}: aligned {missing.sum()} rows via 자치구 mode mapping.")


def fill_fasttrack_date(df: pd.DataFrame, logs: List[str]) -> None:
    column = "신속통합기획 후보지 선정일"
    dates = to_datetime_series(df[column])
    missing = dates.isna()
    if not missing.any():
        df[column] = dates.dt.strftime("%Y-%m-%d")
        return
    notice_dates = to_datetime_series(df["최초 지정 일자"])
    estimates = notice_dates - pd.to_timedelta(120, unit="D")
    estimates = estimates.where(~estimates.isna(), DEFAULT_FASTTRACK_DATE)
    dates = dates.fillna(estimates)
    dates = dates.fillna(DEFAULT_FASTTRACK_DATE)
    df[column] = dates.dt.strftime("%Y-%m-%d")
    logs.append(f"{column}: synthesised {missing.sum()} dates using notice_date-120d rule.")


def fill_duration_initial(df: pd.DataFrame, logs: List[str]) -> None:
    column = "후보지-지정고시"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    fasttrack = to_datetime_series(df["신속통합기획 후보지 선정일"])
    notice_initial = to_datetime_series(df["최초 지정 일자"])
    estimates = (notice_initial - fasttrack).dt.days
    total_days = to_numeric(df["총 소요일수"])
    fallback = numeric.median()
    if pd.isna(fallback):
        fallback = 450
    fill_values = estimates.where(~estimates.isna(), total_days)
    fill_values = fill_values.where(~fill_values.isna(), fallback)
    fill_values = fill_values.clip(lower=0).round()
    df.loc[missing, column] = fill_values[missing].astype(int).astype(str)
    logs.append(f"{column}: rebuilt {missing.sum()} durations via date deltas/fallback median.")


def fill_area_total(df: pd.DataFrame, logs: List[str]) -> None:
    column = "정비구역면적(㎡)"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    median_value = numeric.median()
    if pd.isna(median_value):
        median_value = 30000
    fill_series = numeric.fillna(median_value).round(2)
    df.loc[missing, column] = fill_series[missing]
    logs.append(f"{column}: filled {missing.sum()} rows with global median {median_value:.2f}㎡.")


def fill_area_land(df: pd.DataFrame, logs: List[str]) -> None:
    column = "택지면적(㎡)"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    total = to_numeric(df["정비구역면적(㎡)"])
    ratio = (numeric / total).replace([np.inf, -np.inf], pd.NA)
    ratio = ratio[(ratio > 0) & (ratio <= 1)].median()
    if pd.isna(ratio):
        ratio = 0.82
    base = total.fillna(total.median()).fillna(30000)
    fill_series = (base * ratio).round(2)
    df.loc[missing, column] = fill_series[missing]
    logs.append(f"{column}: applied area_total * {ratio:.2%} for {missing.sum()} rows.")


def _primary_zone_name(value: str | float | int | None) -> str | None:
    if not isinstance(value, str):
        return None
    candidates = value.replace(",", " ").split()
    return candidates[0] if candidates else None


def fill_zoning_type(df: pd.DataFrame, logs: List[str]) -> None:
    column = "용도지역"
    missing = df[column].isna()
    if not missing.any():
        return
    fallback = compute_mode(df[column]) or "제2종일반주거지역"
    df.loc[missing, column] = fallback
    logs.append(f"{column}: fallback '{fallback}' injected for {missing.sum()} rows.")


def fill_zone1_name(df: pd.DataFrame, logs: List[str]) -> None:
    column = "Zone1"
    missing = df[column].isna()
    if not missing.any():
        return
    filled = df[column].copy()
    for idx in df.index[missing]:
        zoning = df.at[idx, "용도지역"]
        candidate = _primary_zone_name(zoning) or "제2종일반주거지역"
        filled.at[idx] = candidate
    df[column] = filled
    logs.append(f"{column}: set {missing.sum()} values using primary 용도지역 token.")


def fill_zone1_area(df: pd.DataFrame, logs: List[str]) -> None:
    column = "Zone1_면적"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    land = to_numeric(df["택지면적(㎡)"])
    ratio = (numeric / land).replace([np.inf, -np.inf], pd.NA)
    ratio = ratio[(ratio > 0) & (ratio <= 1)].median()
    if pd.isna(ratio):
        ratio = 0.65
    fill_series = (land * ratio).round(2)
    df.loc[missing, column] = fill_series[missing]
    logs.append(f"{column}: inferred via 택지면적 * {ratio:.2%} for {missing.sum()} rows.")


def fill_zone1_far_base(df: pd.DataFrame, logs: List[str]) -> None:
    column = "Zone1_기준용적률(%이하)"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    median_value = numeric.median()
    if pd.isna(median_value):
        median_value = 190
    df.loc[missing, column] = format_percent(median_value)
    logs.append(f"{column}: backfilled {missing.sum()} rows with {median_value:.2f}%.")


def fill_zone1_far_legal(df: pd.DataFrame, logs: List[str]) -> None:
    column = "Zone1_법적상한용적률"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    plan = to_numeric(df.get("Zone1_상한용적률"))
    derived = plan.add(10, fill_value=0)
    median_value = numeric.median()
    if pd.isna(median_value):
        median_value = 250
    fill_values = derived.where(~derived.isna(), median_value).fillna(median_value)
    df.loc[missing, column] = fill_values[missing].apply(format_percent)
    logs.append(
        f"{column}: defaulted to 주변 상한+10pp (fallback {median_value:.2f}%)."
    )


def fill_height(df: pd.DataFrame, logs: List[str]) -> None:
    column = "높이(m)"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    median_value = numeric.median()
    if pd.isna(median_value):
        median_value = 110
    df.loc[missing, column] = median_value
    logs.append(f"{column}: set {missing.sum()} heights to median {median_value:.1f}m.")


def fill_floors(df: pd.DataFrame, logs: List[str]) -> None:
    column = "지상층수"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    median_value = numeric.median()
    if pd.isna(median_value):
        median_value = 30
    df.loc[missing, column] = int(round(median_value))
    logs.append(f"{column}: used median {median_value:.0f}층 for {missing.sum()} rows.")


def fill_unit_rent_total(df: pd.DataFrame, logs: List[str]) -> None:
    column = "임대세대총수"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    df[column] = df[column].astype("string")
    total = to_numeric(df["총 세대수"])
    ratio = (numeric / total).replace([np.inf, -np.inf], pd.NA)
    ratio = ratio[(ratio >= 0) & (ratio <= 1)].median()
    if pd.isna(ratio):
        ratio = 0.18
    total_filled = total.fillna(total.median()).fillna(800)
    fill_series = (total_filled * ratio).round().astype(int)
    df.loc[missing, column] = fill_series[missing].astype(int).astype(str)
    logs.append(f"{column}: applied median 임대비중 {ratio:.1%} to {missing.sum()} rows.")


def fill_unit_total(df: pd.DataFrame, logs: List[str]) -> None:
    column = "총 세대수"
    numeric = to_numeric(df[column])
    missing = df[column].isna()
    if not missing.any():
        return
    df[column] = df[column].astype("string")
    sale = to_numeric(df.get("분양세대총수"))
    rent = to_numeric(df.get("임대세대총수"))
    combined = sale.fillna(0) + rent.fillna(0)
    combined = combined.where(combined > 0)
    fallback = numeric.median()
    if pd.isna(fallback):
        fallback = combined.median()
    if pd.isna(fallback):
        fallback = 900
    fill_series = combined.fillna(fallback).round().astype(int)
    df.loc[missing, column] = fill_series[missing].astype(str)
    logs.append(f"{column}: synthesised via 분양+임대 / median ({missing.sum()} rows).")


def ensure_required_populated(df: pd.DataFrame) -> None:
    missing_summary: Dict[str, int] = {}
    for column in REQUIRED_SOURCE_COLUMNS:
        if column not in df.columns:
            missing_summary[column] = -1
            continue
        series = df[column]
        nulls = series.isna().sum()
        if nulls:
            missing_summary[column] = nulls
    if missing_summary:
        raise ValueError(f"Required columns still missing: {missing_summary}")


def apply_fillers(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    logs: List[str] = []
    fill_dgm_name(df, logs)
    fill_district(df, logs)
    fill_dong(df, logs)
    fill_region(df, logs)
    fill_fasttrack_date(df, logs)
    fill_duration_initial(df, logs)
    fill_area_total(df, logs)
    fill_area_land(df, logs)
    fill_zoning_type(df, logs)
    fill_zone1_name(df, logs)
    fill_zone1_area(df, logs)
    fill_zone1_far_base(df, logs)
    fill_zone1_far_legal(df, logs)
    fill_height(df, logs)
    fill_floors(df, logs)
    fill_unit_total(df, logs)
    fill_unit_rent_total(df, logs)
    return df, logs


def main() -> None:
    args = parse_args()
    df_raw = read_raw_csv(args.input, args.encoding)
    df_clean = normalise_strings(df_raw)
    df_filled, logs = apply_fillers(df_clean)
    ensure_required_populated(df_filled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_filled.to_csv(args.output, index=False, encoding=args.encoding)
    print(f"[INFO] Saved synthetic raw CSV to {args.output}")
    print("[INFO] Filler summary:")
    for log in logs:
        print(f"  - {log}")


if __name__ == "__main__":
    main()
