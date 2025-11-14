"""Step 1 preprocessing script for the OLS Analysis project.

This module loads the latest raw CSV (or a user-provided file), keeps only the
columns required for downstream analysis, standardises column names and data
types, reports missing values, removes incomplete rows, and saves the cleaned
output into `data/step1_data` with a timestamped filename.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"
STEP1_DIR = PROJECT_ROOT / "data" / "step1_data"


@dataclass(frozen=True)
class ColumnSpec:
    source: str
    dtype: str
    target: str


COLUMN_SPECS: Tuple[ColumnSpec, ...] = (
    ColumnSpec("PRESENT_SN", "string", "PRESENT_SN"),
    ColumnSpec("DGM_NM", "string", "DGM_NM"),
    ColumnSpec("권역", "category", "region"),
    ColumnSpec("자치구", "category", "district"),
    ColumnSpec("법정동", "string", "dong"),
    ColumnSpec("정비구역 위치", "string", "location"),
    ColumnSpec("최초 지정 고시번호", "string", "notice_no_initial"),
    ColumnSpec("최초 지정 일자", "date", "notice_date_initial"),
    ColumnSpec("최종 변경결정 고시번호", "string", "notice_no_latest"),
    ColumnSpec("최종 변경결정 일자", "date", "notice_date_latest"),
    ColumnSpec("총 소요일수", "int", "duration_days_total"),
    ColumnSpec("후보지-지정고시", "int", "duration_days_initial"),
    ColumnSpec("신속통합기획 후보지 선정일", "date", "fasttrack_proposal_date"),
    ColumnSpec("사례유형", "category", "case_type"),
    ColumnSpec("정비구역면적(㎡)", "float", "area_total"),
    ColumnSpec("건축연면적(㎡)", "float", "area_gfa"),
    ColumnSpec("용도지역", "category", "zoning_type"),
    ColumnSpec("택지면적(㎡)", "float", "area_land"),
    ColumnSpec("주차장면적(㎡)", "float", "area_parking"),
    ColumnSpec("공공시설면적(㎡)", "float", "area_public_facility"),
    ColumnSpec("도로면적(㎡)", "float", "area_road"),
    ColumnSpec("공원면적(㎡)", "float", "area_park"),
    ColumnSpec("녹지면적(㎡)", "float", "area_green"),
    ColumnSpec("공공공지면적(㎡)", "float", "area_public_open"),
    ColumnSpec("학교면적(㎡)", "float", "area_school"),
    ColumnSpec("기타면적(㎡)", "float", "area_other"),
    ColumnSpec("건폐율", "float", "building_coverage"),
    ColumnSpec("용도지역변경 여부", "int", "zoning_change"),
    ColumnSpec("Zone1", "category", "zone1_name"),
    ColumnSpec("Zone1_면적", "float", "zone1_area"),
    ColumnSpec("Zone1_기준용적률(%이하)", "float", "zone1_far_base"),
    ColumnSpec("Zone1_상한용적률", "float", "zone1_far_plan"),
    ColumnSpec("Zone1_법적상한용적률", "float", "zone1_far_legal"),
    ColumnSpec("Zone2", "category", "zone2_name"),
    ColumnSpec("Zone2_면적", "float", "zone2_area"),
    ColumnSpec("Zone2_기준용적률(%이하)", "float", "zone2_far_base"),
    ColumnSpec("Zone2_상한용적률", "float", "zone2_far_plan"),
    ColumnSpec("Zone2_법적상한용적률", "float", "zone2_far_legal"),
    ColumnSpec("Zone3", "category", "zone3_name"),
    ColumnSpec("Zone3_법정용적률", "float", "zone3_far_base"),
    ColumnSpec("Zone3_계획용적률", "float", "zone3_far_plan"),
    ColumnSpec("Zone3_상한용적률", "float", "zone3_far_legal"),
    ColumnSpec("높이(m)", "float", "height"),
    ColumnSpec("지상층수", "int", "floors"),
    ColumnSpec("지하층수", "int", "floors_basement"),
    ColumnSpec("기존세대수", "int", "unit_existing"),
    ColumnSpec("총 세대수", "int", "unit_total"),
    ColumnSpec("분양세대총수", "int", "unit_sale"),
    ColumnSpec("60㎡이하", "int", "unit_s"),
    ColumnSpec("60㎡초과85㎡이하", "int", "unit_m"),
    ColumnSpec("85㎡초과", "int", "unit_l"),
    ColumnSpec("임대세대총수", "int", "unit_rent_total"),
    ColumnSpec("(임대)40㎡이하", "int", "unit_rent_xxs"),
    ColumnSpec("(임대)40㎡초과50㎡이하", "int", "unit_rent_xs"),
    ColumnSpec("(임대)50㎡초과", "int", "unit_rent_s"),
)

COLUMN_ORDER = [spec.source for spec in COLUMN_SPECS]
TARGET_NAMES = {spec.source: spec.target for spec in COLUMN_SPECS}
TARGET_DTYPES = {spec.target: spec.dtype for spec in COLUMN_SPECS}

REQUIRED_TARGET_COLUMNS = {
    "DGM_NM",
    "region",
    "district",
    "dong",
    "duration_days_initial",
    "fasttrack_proposal_date",
    "area_total",
    "zoning_type",
    "area_land",
    "zone1_name",
    "zone1_area",
    "zone1_far_base",
    "zone1_far_legal",
    "height",
    "floors",
    "unit_total",
    "unit_rent_total",
}
# "case_type", 나중에 추가해야함.

def _clean_string(series: pd.Series) -> pd.Series:
    str_series = series.astype("string").str.strip()
    return str_series.replace({"": pd.NA})


def _clean_numeric_string(series: pd.Series) -> pd.Series:
    str_series = series.astype("string")
    str_series = str_series.str.replace(",", "", regex=False)
    str_series = str_series.str.replace("%", "", regex=False)
    str_series = str_series.str.replace(" ", "", regex=False)
    str_series = str_series.replace({"": pd.NA})
    return str_series


def _to_int(series: pd.Series) -> pd.Series:
    cleaned = _clean_numeric_string(series)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    return numeric.round().astype("Int64")


def _to_float(series: pd.Series) -> pd.Series:
    cleaned = _clean_numeric_string(series)
    return pd.to_numeric(cleaned, errors="coerce").astype("Float64")


def _to_category(series: pd.Series) -> pd.Series:
    cleaned = _clean_string(series)
    return cleaned.astype("category")


def _to_date(series: pd.Series) -> pd.Series:
    cleaned = _clean_string(series)
    return pd.to_datetime(cleaned, errors="coerce").dt.normalize()


def _to_string(series: pd.Series) -> pd.Series:
    return _clean_string(series)


TYPE_CASTERS: Dict[str, Callable[[pd.Series], pd.Series]] = {
    "string": _to_string,
    "category": _to_category,
    "int": _to_int,
    "float": _to_float,
    "date": _to_date,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 preprocessing pipeline")
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the raw CSV file. Defaults to the latest file in data/raw_data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=STEP1_DIR,
        help="Directory to save the cleaned CSV (default: data/step1_data)",
    )
    parser.add_argument(
        "--encoding",
        default="cp949",
        help="Primary CSV encoding (default: cp949). The script will fall back to other common encodings automatically if needed.",
    )
    return parser.parse_args()


def find_latest_raw_csv(directory: Path) -> Path:
    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return csv_files[-1]


def load_raw_dataframe(csv_path: Path, encoding: str) -> pd.DataFrame:
    print(f"[INFO] Loading raw data: {csv_path}")
    tried_messages = []
    for candidate in dict.fromkeys([encoding, "cp949", "utf-8-sig"]):
        try:
            df = pd.read_csv(csv_path, encoding=candidate)
        except UnicodeDecodeError as exc:
            tried_messages.append(f"encoding {candidate}: {exc}")
            continue

        if all(col in df.columns for col in COLUMN_ORDER):
            if candidate != encoding:
                print(
                    f"[INFO] Required columns recovered after switching encoding to '{candidate}'."
                )
            return df

        missing_preview = [col for col in COLUMN_ORDER if col not in df.columns]
        display_missing = ", ".join(missing_preview[:5])
        tried_messages.append(
            f"encoding {candidate}: missing columns ({display_missing}{'...' if len(missing_preview) > 5 else ''})"
        )

    details = "; ".join(tried_messages)
    raise KeyError(
        f"Unable to load required columns from {csv_path}. Tried encodings {details}"
    )


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in raw data: {missing}")


def select_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, COLUMN_ORDER)
    trimmed = df[COLUMN_ORDER].copy()
    return trimmed.rename(columns=TARGET_NAMES)


def cast_column_types(df: pd.DataFrame) -> pd.DataFrame:
    for column, dtype in TARGET_DTYPES.items():
        caster = TYPE_CASTERS[dtype]
        df[column] = caster(df[column])
    return df


def report_and_drop_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    required_missing = missing_counts[
        (missing_counts > 0) & (missing_counts.index.isin(REQUIRED_TARGET_COLUMNS))
    ]
    optional_missing = missing_counts[
        (missing_counts > 0) & (~missing_counts.index.isin(REQUIRED_TARGET_COLUMNS))
    ]

    if not required_missing.empty:
        print("[WARN] Required fields with missing values (before drop):")
        for col, count in required_missing.items():
            print(f"    - {col}: {count} missing")
    if not optional_missing.empty:
        print("[INFO] Optional fields with missing values will be preserved:")
        preview = optional_missing.sort_values(ascending=False)
        for col, count in preview.items():
            print(f"    - {col}: {count} missing")

    cleaned = df.dropna(subset=sorted(REQUIRED_TARGET_COLUMNS))
    dropped = len(df) - len(cleaned)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows missing required fields.")
    else:
        print("[INFO] No rows dropped; all required fields populated.")
    return cleaned


def save_output(df: pd.DataFrame, output_dir: Path, source_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"step1_preprocessed_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved cleaned data to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    raw_csv = args.input or find_latest_raw_csv(RAW_DATA_DIR)
    df_raw = load_raw_dataframe(raw_csv, args.encoding)
    print(f"[INFO] Raw shape: {df_raw.shape}")

    df_selected = select_and_rename(df_raw)
    df_casted = cast_column_types(df_selected)
    df_clean = report_and_drop_missing(df_casted)
    print(f"[INFO] Cleaned shape: {df_clean.shape}")

    save_output(df_clean, args.output_dir, raw_csv.name)


if __name__ == "__main__":
    main()
