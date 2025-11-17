import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEP2_DIR = PROJECT_ROOT / "data" / "step2_data"
OLS_DIR = PROJECT_ROOT / "data" / "ols_result"

def find_latest_step2_csv() -> Optional[Path]:
    files = sorted(STEP2_DIR.glob("*.csv"))
    return files[-1] if files else None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OLS on latest step2 dataset.")
    parser.add_argument("--input", type=Path, help="Specific step2 CSV path.")
    parser.add_argument(
        "--target",
        default="duration_months_initial",
        help="Dependent variable.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Rows required after cleaning.",
    )
    return parser.parse_args()

def prepare_data(df: pd.DataFrame, target: str) -> tuple[pd.Series, pd.DataFrame]:
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not in dataset.")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(axis=1, how="all").copy()
    if target not in numeric.columns:
        raise ValueError(f"Target '{target}' is not numeric after coercion.")
    y = numeric[target]
    X = numeric.drop(columns=[target])
    X = X.loc[:, X.std(ddof=0) > 0]
    combined = pd.concat([y, X], axis=1).dropna()
    y_clean = combined[target]
    X_clean = combined.drop(columns=[target])
    if X_clean.empty:
        raise ValueError("No predictors remain after cleaning.")
    return y_clean, X_clean

def run_ols(y: pd.Series, X: pd.DataFrame):
    X_const = sm.add_constant(X, has_constant="add")
    return sm.OLS(y, X_const).fit()

def save_outputs(model, source: Path):
    OLS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = source.stem
    summary_path = OLS_DIR / f"ols_summary_{stem}_{ts}.txt"
    coef_path = OLS_DIR / f"ols_coefficients_{stem}_{ts}.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(model.summary().as_text())

    coef_df = (
        model.params.to_frame("coef").assign(
            std_err=model.bse,
            t_value=model.tvalues,
            p_value=model.pvalues,
        )
    )
    coef_df.to_csv(coef_path, encoding="utf-8-sig")

    print(f"[INFO] Saved summary → {summary_path}")
    print(f"[INFO] Saved coefficients → {coef_path}")

def main():
    args = parse_args()
    csv_path = args.input or find_latest_step2_csv()
    if not csv_path:
        raise FileNotFoundError("No step2 CSV found in data/step2_data.")
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    y, X = prepare_data(df, args.target)

    if len(y) < args.min_samples:
        raise ValueError(
            f"Only {len(y)} rows after cleaning (min {args.min_samples})."
        )

    model = run_ols(y, X)
    save_outputs(model, csv_path)

if __name__ == "__main__":
    main()
