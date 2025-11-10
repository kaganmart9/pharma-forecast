# scripts/train_and_export.py
import re
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -----------------------------
# Configuration
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results" / "final"

FEATURE_FILE_CANDIDATES = [
    DATA_PROCESSED / "pharma_sales_features_v2_clean.csv",
    DATA_PROCESSED / "pharma_sales_features.csv",
]
TARGETS = ["N02BE", "M01AB"]
TRAIN_END = pd.Timestamp("2018-09-16")
TEST_START = pd.Timestamp("2018-09-23")

# Tuned GB parameters per target (from Day 7 tuning)
BEST_PARAMS = {
    "N02BE": {
        "learning_rate": 0.01,
        "max_depth": 4,
        "max_features": None,
        "n_estimators": 400,
        "random_state": 42,
        "subsample": 0.6,
    },
    "M01AB": {
        "learning_rate": 0.05,
        "max_depth": 4,
        "max_features": None,
        "n_estimators": 400,
        "random_state": 42,
        "subsample": 1.0,
    },
}


# -----------------------------
# Utilities
# -----------------------------
def rmse(y_true, y_pred):
    """RMSE without using the newer 'squared' kwarg (for older sklearn)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def smape(y_true, y_pred):
    """Symmetric MAPE (%) for scale-robust accuracy on low volumes."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    denom[denom == 0] = 1e-12
    return 100.0 * np.mean(np.abs(y_true - y_pred) / denom)


def locate_feature_file():
    """Return the first existing feature file path among candidates."""
    for p in FEATURE_FILE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"Feature file not found. Checked: {FEATURE_FILE_CANDIDATES}"
    )


def select_reduced_safe_columns(columns, targets):
    """
    Build the reduced-safe feature set:
    - Include only lag(1,2,3,4,8,12) features
    - Optionally include minimal calendar features if present (weekofyear, month, year)
    """
    lag_pat = re.compile(r"_lag(1|2|3|4|8|12)$")
    feat = [c for c in columns if c not in targets]
    reduced = [c for c in feat if lag_pat.search(c)]
    for cal in ["weekofyear", "month", "year"]:
        if cal in feat:
            reduced.append(cal)
    if not reduced:
        raise ValueError("Reduced-safe feature selection returned 0 columns.")
    return reduced


# -----------------------------
# Main
# -----------------------------
def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    feat_path = locate_feature_file()
    print(f"[INFO] Using features: {feat_path}")
    df = (
        pd.read_csv(feat_path, parse_dates=["datum"])
        .sort_values("datum")
        .set_index("datum")
    )

    # Time-based split
    train = df.loc[:TRAIN_END].copy()
    test = df.loc[TEST_START:].copy()

    # Feature matrix (reduced-safe)
    reduced_cols = select_reduced_safe_columns(df.columns, TARGETS)
    X_train = train[reduced_cols].copy()
    X_test = test[reduced_cols].copy()
    y_train = train[TARGETS].copy()
    y_test = test[TARGETS].copy()

    # Sanity checks
    for name, arr in [
        ("X_train", X_train),
        ("X_test", X_test),
        ("y_train", y_train),
        ("y_test", y_test),
    ]:
        if arr.isnull().values.any():
            raise ValueError(f"NaNs found in {name}")

    # Train per target and export
    rows = []
    for tgt in TARGETS:
        print(f"[TRAIN] Gradient Boosting for {tgt}")
        params = BEST_PARAMS[tgt]
        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train[tgt])

        # Save model
        model_path = MODELS_DIR / f"gb_{tgt}.pkl"
        joblib.dump(model, model_path)
        print(f"[SAVE] {tgt} → {model_path}")

        # Evaluate on test
        yhat = model.predict(X_test)
        rmse_val = rmse(y_test[tgt], yhat)
        mae = mean_absolute_error(y_test[tgt], yhat)
        smape_val = smape(y_test[tgt], yhat)
        mae = mean_absolute_error(y_test[tgt], yhat)
        smape_val = smape(y_test[tgt], yhat)

        rows.append(
            {
                "Target": tgt,
                "Test_RMSE": round(rmse_val, 4),
                "Test_MAE": round(mae, 4),
                "Test_sMAPE(%)": round(smape_val, 4),
            }
        )

        # Save per-target predictions
        preds = pd.DataFrame(
            {"date": y_test.index, "y_true": y_test[tgt].values, "y_pred": yhat}
        )
        preds_path = RESULTS_DIR / f"gb_{tgt}_test_preds.csv"
        preds.to_csv(preds_path, index=False)
        print(f"[SAVE] {tgt} test predictions → {preds_path}")

    # Save metrics summary
    summ = pd.DataFrame(rows)
    metrics_path = RESULTS_DIR / "gb_test_metrics_rebuild.csv"
    summ.to_csv(metrics_path, index=False)
    print(f"[SAVE] Metrics summary → {metrics_path}")

    print("\n[DONE] Training & export completed.")
    print(summ)


if __name__ == "__main__":
    main()
