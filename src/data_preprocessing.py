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
from typing import Callable, Dict, Iterable, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw_data"

STEP1_DIR = PROJECT_ROOT / "data" / "step1_data"


@dataclass(frozen=True)
class ColumnSpec:
    source: str
    dtype: str
    target: str


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    column_specs: Tuple[ColumnSpec, ...]
    required_targets: Set[str]
    raw_subdir: str
    step1_subdir: str
    column_order: Tuple[str, ...]
    target_names: Dict[str, str]
    target_dtypes: Dict[str, str]
    required_source_columns: Set[str]


def _build_config(
    name: str,
    specs: Tuple[ColumnSpec, ...],
    required: Set[str],
    raw_subdir: str,
    step1_subdir: str,
) -> PipelineConfig:
    column_order = tuple(spec.source for spec in specs)
    target_names = {spec.source: spec.target for spec in specs}
    target_dtypes = {spec.target: spec.dtype for spec in specs}
    required_sources = {spec.source for spec in specs if spec.target in required}
    return PipelineConfig(
        name=name,
        column_specs=specs,
        required_targets=required,
        raw_subdir=raw_subdir,
        step1_subdir=step1_subdir,
        column_order=column_order,
        target_names=target_names,
        target_dtypes=target_dtypes,
        required_source_columns=required_sources,
    )


FASTTRACK_COLUMN_SPECS: Tuple[ColumnSpec, ...] = (
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

FASTTRACK_REQUIRED_TARGETS: Set[str] = {
    "DGM_NM",
    "region",
    "district",
    "dong",
    "duration_days_initial",
    "fasttrack_proposal_date",
    "area_total",
    "area_land",
    "zone1_name",
    "zone1_area",
    "height",
    "floors",
    "unit_total",
    "unit_rent_total",
}


GENERAL_COLUMN_SPECS: Tuple[ColumnSpec, ...] = (
    ColumnSpec("사업번호", "string", "PRESENT_SN"),
    ColumnSpec("신속통합기획", "category", "fasttrack_note"),
    ColumnSpec("권역", "category", "region"),
    ColumnSpec("자치구", "category", "district"),
    ColumnSpec("법정동", "string", "dong"),
    ColumnSpec("운영구분", "category", "operation_type"),
    ColumnSpec("진행단계", "category", "progress_stage"),
    ColumnSpec("상태", "category", "status"),
    ColumnSpec("토지등 소유자 수", "int", "owner_count"),
    ColumnSpec("정비구역명칭", "string", "DGM_NM"),
    ColumnSpec("정비구역위치", "string", "location"),
    ColumnSpec("정비구역면적(㎡)", "float", "area_total"),
    ColumnSpec("용도지역", "category", "zoning_type"),
    ColumnSpec("택지면적(㎡)", "float", "area_land"),
    ColumnSpec("도로면적(㎡)", "float", "area_road"),
    ColumnSpec("공원면적(㎡)", "float", "area_park"),
    ColumnSpec("녹지면적(㎡)", "float", "area_green"),
    ColumnSpec("공공공지면적(㎡)", "float", "area_public_open"),
    ColumnSpec("학교면적(㎡)", "float", "area_school"),
    ColumnSpec("기타면적(㎡)", "float", "area_other"),
    ColumnSpec("건폐율", "float", "building_coverage"),
    ColumnSpec("계획용적률", "float", "far_plan"),
    ColumnSpec("높이(m)", "float", "height"),
    ColumnSpec("지상층수", "int", "floors"),
    ColumnSpec("지하층수", "int", "floors_basement"),
    ColumnSpec("분양세대총수", "int", "unit_sale"),
    ColumnSpec("60㎡이하", "int", "unit_s"),
    ColumnSpec("60㎡초과85㎡이하", "int", "unit_m"),
    ColumnSpec("85㎡초과", "int", "unit_l"),
    ColumnSpec("임대세대총수", "int", "unit_rent_total"),
    ColumnSpec("(임대)40㎡이하", "int", "unit_rent_xxs"),
    ColumnSpec("(임대)40㎡초과50㎡이하", "int", "unit_rent_xs"),
    ColumnSpec("(임대)50㎡초과", "int", "unit_rent_s"),
    ColumnSpec("lat", "float", "lat"),
    ColumnSpec("lon", "float", "lon"),
    ColumnSpec("일반/재촉지구", "category", "redevelopment_type"),
    ColumnSpec("기존 가구수(멸실량)", "int", "unit_existing"),
    ColumnSpec("최초", "date", "date_initial_notice"),
    ColumnSpec("변경(최종)", "date", "date_latest_notice"),
    ColumnSpec("추진위원회", "date", "date_committee"),
    ColumnSpec("조합설립인가(사업시행자 지정일)", "date", "date_association"),
    ColumnSpec("건축심의", "date", "date_arch_review"),
    ColumnSpec("사업시행인가_최초", "date", "date_business_initial"),
    ColumnSpec("사업시행인가_변경(최종)", "date", "date_business_latest"),
    ColumnSpec("관리처분계획인가_최초", "date", "date_disposition_initial"),
    ColumnSpec("관리처분계획인가_변경(최종)", "date", "date_disposition_latest"),
    ColumnSpec("이주시작일", "date", "date_relocation_start"),
    ColumnSpec("이주종료일", "date", "date_relocation_end"),
    ColumnSpec("착공", "date", "date_construction_start"),
    ColumnSpec("총세대수", "int", "unit_total"),
)

GENERAL_REQUIRED_TARGETS: Set[str] = {
    "PRESENT_SN",
    "district",
    "DGM_NM",
    "area_total",
    "unit_total",
    "area_land",
    "unit_rent_total",
    "region",
}


PIPELINE_CONFIGS: Dict[str, PipelineConfig] = {
    "fasttrack": _build_config(
        "fasttrack",
        FASTTRACK_COLUMN_SPECS,
        FASTTRACK_REQUIRED_TARGETS,
        raw_subdir="fasttrack",
        step1_subdir="fasttrack",
    ),
    "general": _build_config(
        "general",
        GENERAL_COLUMN_SPECS,
        GENERAL_REQUIRED_TARGETS,
        raw_subdir="general",
        step1_subdir="general",
    ),
}

LOG_FEATURES: Dict[str, str] = {
    # 규모/면적 계열
    "area_total": "log_area_total",
    "area_land": "log_area_land",
    "area_gfa": "log_area_gfa",
    "area_parking": "log_area_parking",
    "area_public_facility": "log_area_public_facility",
    "area_road": "log_area_road",
    "area_park": "log_area_park",
    "area_green": "log_area_green",
    "area_public_open": "log_area_public_open",
    "area_school": "log_area_school",
    "area_other": "log_area_other",
    "zone1_area": "log_zone1_area",
    "zone2_area": "log_zone2_area",
    # 세대수 계열
    "unit_existing": "log_unit_existing",
    "unit_total": "log_unit_total",
    "unit_sale": "log_unit_sale",
    "unit_s": "log_unit_s",
    "unit_m": "log_unit_m",
    "unit_l": "log_unit_l",
    "unit_rent_total": "log_unit_rent_total",
    "unit_rent_xxs": "log_unit_rent_xxs",
    "unit_rent_xs": "log_unit_rent_xs",
    "unit_rent_s": "log_unit_rent_s",
}

COLUMN_ALIASES: Dict[str, Dict[str, str]] = {
    "fasttrack": {
        "후보지~지정고시": "후보지-지정고시",
        "60㎡초과~85㎡이하": "60㎡초과85㎡이하",
        "(임대)40㎡초과~50㎡이하": "(임대)40㎡초과50㎡이하",
        "학교면 적(㎡)": "학교면적(㎡)",
    }
}


def _clean_numeric_string(series: pd.Series) -> pd.Series:
    str_series = series.astype("string")
    str_series = str_series.str.replace(",", "", regex=False)
    str_series = str_series.str.replace(" ", "", regex=False)
    str_series = str_series.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    return str_series


def _clean_string(series: pd.Series) -> pd.Series:
    str_series = series.astype("string")
    str_series = str_series.str.strip()
    str_series = str_series.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA})
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


def _compute_infrastructure_gap(df: pd.DataFrame) -> pd.Series:
    total = df["area_total"].astype("Float64")
    land = df["area_land"].astype("Float64")
    result = total - land
    return result.astype("Float64")


def _compute_rental_ratio(df: pd.DataFrame) -> pd.Series:
    rent = df["unit_rent_total"].astype("Float64")
    total = df["unit_total"].astype("Float64")
    ratio = rent / total
    ratio[(total <= 0) | total.isna() | rent.isna()] = pd.NA
    return ratio.astype("Float64")


def add_pipeline_specific_features(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    df = df.copy()
    if {
        "area_total",
        "area_land",
    }.issubset(df.columns):
        df["infrastructure_ratio"] = _compute_infrastructure_gap(df)
    if {
        "unit_rent_total",
        "unit_total",
    }.issubset(df.columns):
        df["rental_unit_ratio"] = _compute_rental_ratio(df)
    if config.name == "general" and "infrastructure_ratio" in df.columns:
        invalid_mask = df["infrastructure_ratio"] < 0
        invalid_count = int(invalid_mask.sum())
        if invalid_count:
            preview_ids = df.loc[invalid_mask, "PRESENT_SN"].dropna().astype(str).tolist()
            preview = ", ".join(preview_ids[:5])
            suffix = "..." if len(preview_ids) > 5 else ""
            print(
                f"[WARN] Dropping {invalid_count} rows with negative infrastructure ratio (PRESENT_SN: {preview}{suffix})."
            )
            df = df.loc[~invalid_mask].copy()
    df = _add_log_features(df)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1 preprocessing pipeline")
    parser.add_argument(
        "--pipeline",
        choices=sorted(PIPELINE_CONFIGS.keys()),
        default="fasttrack",
        help="Which pipeline configuration to use (default: fasttrack)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the raw CSV file. Defaults to the latest file in the pipeline-specific raw folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save the cleaned CSV. Defaults to data/step1_data/<pipeline>",
    )
    parser.add_argument(
        "--encoding",
        default="cp949",
        help="Primary CSV encoding (default: cp949). The script will fall back to other common encodings automatically if needed.",
    )
    return parser.parse_args()


def find_latest_raw_csv(config: PipelineConfig) -> Path:
    search_dir = RAW_DATA_DIR / config.raw_subdir
    if not search_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory {search_dir} does not exist for pipeline '{config.name}'."
        )
    candidates = sorted(search_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {search_dir} for pipeline '{config.name}'."
        )
    return candidates[-1]


def load_raw_dataframe(
    csv_path: Path, encoding: str, config: PipelineConfig
) -> pd.DataFrame:
    print(f"[INFO] Loading raw data: {csv_path}")
    tried_messages = []
    for candidate in dict.fromkeys([encoding, "cp949", "utf-8-sig"]):
        try:
            df = pd.read_csv(csv_path, encoding=candidate)
        except UnicodeDecodeError as exc:
            tried_messages.append(f"encoding {candidate}: {exc}")
            continue

        df = _standardize_column_names(df)
        df = _apply_column_aliases(df, config.name)

        if all(col in df.columns for col in config.required_source_columns):
            if candidate != encoding:
                print(
                    f"[INFO] Required columns recovered after switching encoding to '{candidate}'."
                )
            return df

        missing_preview = [
            col for col in config.required_source_columns if col not in df.columns
        ]
        display_missing = ", ".join(missing_preview[:5])
        tried_messages.append(
            f"encoding {candidate}: missing columns ({display_missing}{'...' if len(missing_preview) > 5 else ''})"
        )

    details = "; ".join(tried_messages)
    raise KeyError(
        f"Unable to load required columns from {csv_path}. Tried encodings {details}"
    )


def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
    return df


def _apply_column_aliases(df: pd.DataFrame, pipeline_name: str) -> pd.DataFrame:
    alias_map = COLUMN_ALIASES.get(pipeline_name, {})
    rename_map = {
        alias: target
        for alias, target in alias_map.items()
        if alias in df.columns and target not in df.columns
    }
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    for source, target in LOG_FEATURES.items():
        if source not in df.columns:
            continue
        df[target] = _log1p_series(df[source])
    return df


def _log1p_series(series: pd.Series) -> pd.Series:
    numeric = series.astype("Float64")
    mask = numeric.notna()
    result = numeric.copy()
    result[mask] = np.log1p(numeric[mask])
    return result.astype("Float64")


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in raw data: {missing}")


def select_and_rename(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    ensure_columns(df, config.required_source_columns)
    missing_optional = [
        col
        for col in config.column_order
        if col not in df.columns and col not in config.required_source_columns
    ]
    if missing_optional:
        preview = ", ".join(missing_optional[:5])
        suffix = "..." if len(missing_optional) > 5 else ""
        print(
            f"[INFO] Optional columns missing from raw data will be filled with NA: {preview}{suffix}"
        )
    trimmed = df.reindex(columns=config.column_order).copy()
    return trimmed.rename(columns=config.target_names)


def cast_column_types(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    for column, dtype in config.target_dtypes.items():
        caster = TYPE_CASTERS[dtype]
        df[column] = caster(df[column])
    return df


def report_and_drop_missing(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    required_missing = missing_counts[
        (missing_counts > 0) & (missing_counts.index.isin(config.required_targets))
    ]
    optional_missing = missing_counts[
        (missing_counts > 0) & (~missing_counts.index.isin(config.required_targets))
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

    cleaned = df.dropna(subset=sorted(config.required_targets))
    dropped = len(df) - len(cleaned)
    if dropped:
        print(f"[INFO] Dropped {dropped} rows missing required fields.")
    else:
        print("[INFO] No rows dropped; all required fields populated.")
    return cleaned


def save_output(
    df: pd.DataFrame, output_dir: Path, source_name: str, config: PipelineConfig
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_source = Path(source_name).stem
    output_path = output_dir / f"step1_{config.name}_{safe_source}_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    print(f"[INFO] Saved cleaned data to {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    config = PIPELINE_CONFIGS[args.pipeline]
    raw_csv = args.input or find_latest_raw_csv(config)
    df_raw = load_raw_dataframe(raw_csv, args.encoding, config)
    print(f"[INFO] Raw shape: {df_raw.shape}")

    df_selected = select_and_rename(df_raw, config)
    df_casted = cast_column_types(df_selected, config)
    df_enriched = add_pipeline_specific_features(df_casted, config)
    df_clean = report_and_drop_missing(df_enriched, config)
    print(f"[INFO] Cleaned shape: {df_clean.shape}")

    default_output_dir = STEP1_DIR / config.step1_subdir
    output_dir = args.output_dir or default_output_dir
    save_output(df_clean, output_dir, raw_csv.name, config)


if __name__ == "__main__":
    main()
