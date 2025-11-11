# app.py
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from typing import List
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------- App config --------------------------
st.set_page_config(page_title="Pharma Forecast (GB)", layout="wide")
st.title(" Pharma Forecast — Gradient Boosting")

# -------------------------- Project paths (relative; no absolute printing) --------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

FEATURES_CSV = DATA_DIR / "pharma_sales_features_v2_clean.csv"
RAW_DATA_CSV = BASE_DIR / "data" / "raw" / "pharma_sales.csv"
DATE_COL = "datum"
TARGETS = ["N02BE", "M01AB"]


# -------------------------- Helpers --------------------------
@st.cache_data(show_spinner=False)
def load_canonical_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_CSV)
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    return df


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_CSV.exists():
        return pd.DataFrame()

    df = pd.read_csv(RAW_DATA_CSV)

    date_col_guess = None
    if DATE_COL in df.columns:
        date_col_guess = DATE_COL
    elif "date" in df.columns:
        date_col_guess = "date"

    if date_col_guess:
        try:
            df[date_col_guess] = pd.to_datetime(df[date_col_guess], errors="coerce")
        except Exception:
            pass

    return df


@st.cache_data(show_spinner=False)
def get_feature_names() -> List[str]:
    df = load_canonical_features()
    return [c for c in df.columns if c not in {DATE_COL, *TARGETS}]


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if DATE_COL not in work.columns:
        raise KeyError(f"Required date column '{DATE_COL}' not found.")
    work[DATE_COL] = pd.to_datetime(work[DATE_COL])

    needed = get_feature_names()
    for c in needed:
        if c not in work.columns:
            work[c] = np.nan

    if "weekofyear" in work.columns:
        work["weekofyear"] = work[DATE_COL].dt.isocalendar().week.astype(int)
    if "year" in work.columns:
        work["year"] = work[DATE_COL].dt.year.astype(int)

    return work.sort_values(DATE_COL).reset_index(drop=True)


def impute_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    work = df.copy()
    cols = [c for c in features if c in work.columns]
    work[cols] = work[cols].ffill().bfill()
    med = work[cols].median(numeric_only=True)
    work[cols] = work[cols].fillna(med).fillna(0)
    return work


def safe_tail_row(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    row = df.tail(1)[features].copy().ffill().bfill()
    for c in features:
        if row[c].isna().any():
            col_med = pd.to_numeric(df[c], errors="coerce").median()
            row[c] = row[c].fillna(0 if pd.isna(col_med) else col_med)
    return row


@st.cache_resource(show_spinner=False)
def load_model(tgt: str):
    p = MODELS_DIR / f"gb_{tgt}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Model file missing: {p.name}")
    return joblib.load(p)


# -------------------------- Prediction routines --------------------------
def predict_latest_weeks(df_raw: pd.DataFrame, tgt: str, weeks: int) -> pd.DataFrame:
    model = load_model(tgt)
    features = (
        model.feature_names_in_.tolist()
        if hasattr(model, "feature_names_in_")
        else get_feature_names()
    )

    df = impute_features(ensure_features(df_raw), features)
    tail = df.tail(weeks).copy()

    y_hat = model.predict(tail[features])
    y_true = (
        pd.to_numeric(tail[tgt], errors="coerce") if tgt in tail.columns else np.nan
    )

    return pd.DataFrame(
        {
            "date": tail[DATE_COL],
            "target": tgt,
            "y_true": y_true,
            "y_pred": y_hat.astype(float),
        }
    )


def forecast_next_weeks(df_raw: pd.DataFrame, tgt: str, weeks: int) -> pd.DataFrame:
    model = load_model(tgt)
    features = (
        model.feature_names_in_.tolist()
        if hasattr(model, "feature_names_in_")
        else get_feature_names()
    )

    cur = impute_features(ensure_features(df_raw), features)
    out = []

    for _ in range(weeks):
        Xrow = safe_tail_row(cur, features)
        yhat = float(model.predict(Xrow)[0])

        next_date = pd.to_datetime(cur[DATE_COL].max()) + pd.Timedelta(days=7)
        new_row = {DATE_COL: next_date}

        for col in features:
            if col.endswith("_lag1"):
                base = col[:-5]
                new_row[col] = cur[base].iloc[-1] if base in cur.columns else np.nan
            elif "_lag" in col:
                try:
                    base, lag = col.rsplit("_lag", 1)
                    lag = int(lag)
                    prev_col = f"{base}_lag{lag-1}"
                    new_row[col] = (
                        cur[prev_col].iloc[-1]
                        if (prev_col in cur.columns and lag > 1)
                        else np.nan
                    )
                except Exception:
                    new_row[col] = np.nan
            elif col == "weekofyear":
                new_row[col] = int(pd.to_datetime(next_date).isocalendar().week)
            elif col == "year":
                new_row[col] = int(pd.to_datetime(next_date).year)
            else:
                new_row[col] = cur[col].iloc[-1] if col in cur.columns else np.nan

        new_row[tgt] = yhat
        cur = pd.concat([cur, pd.DataFrame([new_row])], ignore_index=True)
        cur = impute_features(cur, features)
        out.append({"date": next_date, "target": tgt, "y_true": np.nan, "y_pred": yhat})

    return pd.DataFrame(out)


# -------------------------- Plotly figure (with vertical grid lines) --------------------------
def plot_interactive(df_pred: pd.DataFrame, title: str):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Scatter(
            x=df_pred["date"],
            y=df_pred["y_true"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Actual: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df_pred["date"],
            y=df_pred["y_pred"],
            mode="lines",
            name="Predicted (GB)",
            line=dict(color="#d62728", width=2, dash="dash"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Predicted: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=20, t=60, b=30),
        height=420,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#303552", tickformat="%Y-%m-%d")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------- Sidebar UI --------------------------
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Mode", ["Latest N weeks", "Forecast next N weeks"], index=0)
tgt = st.sidebar.selectbox("Target", TARGETS, index=0)

# --- YENİ DEĞİŞİKLİK BURADA ---
# Mode'a göre slider için max hafta sayısını dinamik olarak ayarla
if mode == "Forecast next N weeks":
    max_weeks = 15
else:  # "Latest N weeks"
    max_weeks = 56

# Orijinal varsayılan değer (30), yeni max_weeks'ten (15) büyükse hata verir.
# Bu yüzden varsayılan değeri, izin verilen max ile 30'un minimumu olarak ayarla.
default_val = min(30, max_weeks)

weeks = st.sidebar.slider(
    "Weeks",
    min_value=5,
    max_value=max_weeks,  # Dinamik max değer
    value=default_val,  # Hata vermeyen dinamik varsayılan değer
    step=1,
)
# --- YENİ DEĞİŞİKLİK SONU ---

uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **CSV Upload Instructions**

    The uploaded CSV must match the structure of the training data:
    * Must include the date column: `datum` (format: `YYYY-MM-DD`).
    * Should include target columns: `N02BE`, `M01AB` (if available for comparison).
    * Must include all feature columns (lags, promo flags, etc.) used by the model.
    * Missing columns/values will be imputed (FFill/BFill/Median), which may affect accuracy.
    """
)

# choose data source
if uploaded is not None:
    df_input = pd.read_csv(uploaded)
else:
    df_input = load_canonical_features().copy()

# -------------------------- Run & render --------------------------
try:
    if mode == "Latest N weeks":
        pred = predict_latest_weeks(df_input, tgt, weeks)
        plot_interactive(pred, f"{tgt} — Latest {weeks} Weeks")
        st.subheader(f"Latest {weeks} Weeks — {tgt}")
    else:
        pred = forecast_next_weeks(df_input, tgt, weeks)
        plot_interactive(pred, f"{tgt} — Forecast Next {weeks} Weeks")
        st.subheader(f"Forecast Next {weeks} Weeks — {tgt}")

    with st.expander("Predictions table (click to open)", expanded=False):
        st.dataframe(pred, use_container_width=True, hide_index=True)

    with st.expander("Preview (Processed) data (first 10 rows)", expanded=False):
        st.dataframe(df_input.head(10), use_container_width=True)

    with st.expander("Preview RAW data (first 10 rows)", expanded=False):
        df_raw_preview = load_raw_data()
        if not df_raw_preview.empty:
            st.dataframe(df_raw_preview.head(10), use_container_width=True)
        else:
            st.info(
                f"Raw data file (data/raw/pharma_sales.csv) not found in project directory."
            )

    csv_bytes = pred.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions as CSV",
        data=csv_bytes,
        file_name=(
            "latest_predictions.csv"
            if mode == "Latest N weeks"
            else "future_forecast.csv"
        ),
        mime="text/csv",
    )

except Exception as e:
    st.error(f"Forecasting error: {e}")
    st.exception(e)
