# scripts/predict_latest.py
# -*- coding: utf-8 -*-
"""
Predict the latest N weeks with the exported Gradient Boosting models
and save a tidy CSV for each target plus a combined file.

Usage:
    python -m scripts.predict_latest --weeks 8
    python -m scripts.predict_latest --weeks 8 --data data/processed/pharma_sales_features_v2_clean.csv
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# --------------------
# Configuration
# --------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DEFAULT = BASE_DIR / "data" / "processed" / "pharma_sales_features_v2_clean.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results" / "final"

DATE_COL = "datum"
TARGETS = ["N02BE", "M01AB"]  # adjust here if you add/remove targets


# --------------------
# Utilities
# --------------------
def _ensure_datetime(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _load_model_for_target(tgt: str):
    path = MODELS_DIR / f"gb_{tgt}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found for {tgt}: {path}")
    model = joblib.load(path)
    # Prefer the features the model was trained with (no mismatch headaches)
    if hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)
    else:
        # Fallback: try to read from a sidecar text file if you created one previously
        # or raise a clear error.
        raise AttributeError(
            f"Model for {tgt} does not expose 'feature_names_in_'. "
            "Re-train with a pandas DataFrame so sklearn stores column names, "
            "or add custom logic to supply the reduced-safe feature list."
        )
    return model, feat_names


def predict_window(df: pd.DataFrame, weeks: int = 8) -> pd.DataFrame:
    """
    Builds predictions for the last `weeks` rows for every target using its
    corresponding Gradient Boosting model. Returns a tidy long DataFrame.
    """
    df = _ensure_datetime(df, DATE_COL).sort_values(DATE_COL).reset_index(drop=True)
    dfw = df.tail(weeks).copy()

    all_parts = []

    for tgt in TARGETS:
        model, feat_names = _load_model_for_target(tgt)

        # Align dataframe to the exact training feature set
        missing = [c for c in feat_names if c not in dfw.columns]
        if missing:
            raise KeyError(
                f"[{tgt}] Missing {len(missing)} feature(s) required by the model, "
                f"e.g. {missing[:5]} ... Make sure you used the SAME feature builder."
            )

        Xw = dfw[feat_names]  # <-- DataFrame (no .values)
        yhat = model.predict(Xw)  # sklearn keeps feature names & avoids warnings

        # y_true may or may not be present (e.g., future inference). Build Series safely.
        if tgt in dfw.columns:
            y_true_series = dfw[tgt]
        else:
            y_true_series = pd.Series([np.nan] * len(dfw), index=dfw.index)

        # Build output part — NO .values anywhere
        part = pd.DataFrame(
            {
                "date": dfw[DATE_COL],
                "target": tgt,
                "y_true": y_true_series,
                "y_pred": pd.Series(yhat, index=dfw.index, dtype="float64"),
            }
        )
        all_parts.append(part)

    out = pd.concat(all_parts, axis=0, ignore_index=True)
    return out


# --------------------
# CLI
# --------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict latest N weeks with exported GB models."
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(DATA_DEFAULT),
        help="Path to processed feature CSV",
    )
    p.add_argument(
        "--weeks", type=int, default=8, help="How many recent weeks to predict"
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(RESULTS_DIR / "latest_predictions.csv"),
        help="Output CSV path (combined)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Using features: {data_path}")
    print(f"[INFO] Models dir    : {MODELS_DIR}")
    print(f"[INFO] Results dir   : {out_path.parent}")

    # Load data
    df = pd.read_csv(data_path)
    df = _ensure_datetime(df, DATE_COL)

    # Predict
    preds_long = predict_window(df, weeks=args.weeks)

    # Save per-target and combined
    for tgt in TARGETS:
        tgt_df = preds_long[preds_long["target"] == tgt].copy()
        tgt_path = out_path.parent / f"latest_{tgt}.csv"
        tgt_df.to_csv(tgt_path, index=False)
        print(f"[SAVE] {tgt} → {tgt_path}")

    preds_long.to_csv(out_path, index=False)
    print(f"[DONE] Combined predictions → {out_path}")


if __name__ == "__main__":
    main()
