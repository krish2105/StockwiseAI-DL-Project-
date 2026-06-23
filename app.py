"""
StockWise AI — Demand Forecasting & Auto-Reorder Dashboard
Deep Learning · MAIB · Sept 2025 · Term 3
Team: Krishna Mathur · Yash Petkar · Atharva Soundankar · (Member 4)

Run:  streamlit run dashboard/app.py
The dashboard works out-of-the-box from the pre-computed outputs/ files.
If the trained Keras model is present it ALSO does live next-day prediction.
"""
import os, json
# Import tensorflow first to prevent thread deadlocks on macOS Apple Silicon
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import streamlit as st
import joblib

@st.cache_resource
def get_model(path):
    return load_model(path)

@st.cache_resource
def get_scaler(path):
    return joblib.load(path)

# ---------- paths (work whether run from repo root or /dashboard) ----------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
def p(*a): return os.path.join(ROOT, *a)

st.set_page_config(page_title="StockWise AI", page_icon="🛒", layout="wide")

# ---------- styling ----------
st.markdown("""
<style>
:root {}
.block-container {padding-top: 2rem;}
.metric-card {background:#F6FBF8;border:1px solid #D1FAE5;border-radius:14px;
  padding:14px 18px;text-align:center;}
.metric-card h2 {color:#0F5132;margin:0;font-size:30px;}
.metric-card span {color:#64748B;font-size:13px;}
.verdict {border-radius:14px;padding:22px;text-align:center;color:white;font-weight:700;}
.title {color:#0F5132;}
</style>
""", unsafe_allow_html=True)

# ---------- data loaders ----------
@st.cache_data
def load_sales():
    f = p("data", "grocery_sales.csv")
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["date"])
    return None

@st.cache_data
def load_forecast():
    f = p("outputs", "forecast_results.csv")
    if os.path.exists(f):
        return pd.read_csv(f, parse_dates=["date"])
    return None

@st.cache_data
def load_metrics():
    f = p("outputs", "metrics.json")
    if os.path.exists(f):
        return json.load(open(f))
    return {"RMSE": "—", "MAE": "—", "MAPE": "—", "R2": "—"}

sales = load_sales()
fc = load_forecast()
metrics = load_metrics()

# ---------- header ----------
st.markdown("<h1 class='title'>🛒 StockWise AI</h1>", unsafe_allow_html=True)
st.markdown("**Smart Demand Forecasting & Automated Reorder Engine** — Deep Learning · MAIB · Sept 2025 · Term 3")
st.caption("Team: Krishna Mathur · Yash Petkar · Atharva Soundankar · (Member 4)")
st.divider()

# ---------- sidebar ----------
st.sidebar.header("⚙️ Controls")
if sales is not None:
    product = st.sidebar.selectbox("Product", sorted(sales.product_id.unique()), index=0)
    store   = st.sidebar.selectbox("Store",   sorted(sales.store_id.unique()),   index=0)
else:
    product, store = "MILK_1L", "DXB_MARINA"
    st.sidebar.info("grocery_sales.csv not found — using demo forecast only.")

st.sidebar.markdown("### Reorder simulator")
current_stock = st.sidebar.slider("Current stock on shelf (units)", 0, 2000, 600, 50)
safety_stock  = st.sidebar.slider("Safety stock buffer (units)", 0, 400, 50, 10)

# ---------- KPI row ----------
st.subheader("📈 Model performance (unseen test data)")
m1, m2, m3, m4 = st.columns(4)
for col, (k, lab) in zip([m1, m2, m3, m4],
                          [("MAPE", "mean % error"), ("MAE", "units avg error"),
                           ("RMSE", "units avg error"), ("R2", "variance explained")]):
    val = metrics.get(k, "—")
    suffix = "%" if k == "MAPE" else ""
    col.markdown(f"<div class='metric-card'><h2>{val}{suffix}</h2><span>{k} · {lab}</span></div>",
                 unsafe_allow_html=True)

st.divider()

# ---------- forecast chart ----------
left, right = st.columns([2, 1])
with left:
    st.subheader("🔮 Actual vs Forecast")
    if fc is not None:
        chart_df = fc.set_index("date")[["actual", "forecast"]].tail(60)
        st.line_chart(chart_df, color=["#F59E0B", "#0F5132"])
        st.caption("Last 60 test days · orange = actual, green = StockWise forecast")
    else:
        st.warning("Run the notebook first to create outputs/forecast_results.csv")

with right:
    st.subheader("📊 Demand drivers")
    if sales is not None:
        s = sales[(sales.product_id == product) & (sales.store_id == store)]
        avg_norm = s[s.is_holiday == 0].units_sold.mean()
        avg_hol  = s[s.is_holiday == 1].units_sold.mean()
        avg_wknd = s[s.is_weekend == 1].units_sold.mean()
        avg_promo= s[s.promo_flag == 1].units_sold.mean()
        st.write(pd.DataFrame({
            "Scenario": ["Normal day", "Weekend", "Holiday (Eid)", "Promotion"],
            "Avg units": [round(avg_norm), round(avg_wknd), round(avg_hol), round(avg_promo)]
        }).set_index("Scenario"))
    else:
        st.info("Load sales data to see demand drivers.")

st.divider()

# ---------- next-7-day forecast + decision engine ----------
st.subheader("🚚 Reorder Decision Engine")

def next7_demand():
    """Sum of next-7-day demand. Uses live model if available, else last-7 forecast mean pattern."""
    # try live model
    model_f = p("models", "stockwise_lstm.keras")
    scal_f  = p("models", "scaler.joblib")
    if sales is not None and os.path.exists(model_f) and os.path.exists(scal_f):
        try:
            bundle = get_scaler(scal_f)
            xsc, ysc = bundle["xsc"], bundle["ysc"]
            WINDOW = bundle["window"]; FEATURES = bundle["features"]
            EXOG = ["is_weekend", "is_payday", "is_holiday", "promo_flag", "avg_temp_c"]
            s = sales[(sales.product_id == product) & (sales.store_id == store)].sort_values("date").reset_index(drop=True)
            exog_next = s[EXOG].shift(-1)
            feat = pd.concat([s[["units_sold"]], exog_next], axis=1).iloc[:-1].reset_index(drop=True)
            fs = xsc.transform(feat)
            model = get_model(model_f)
            # iteratively predict 7 steps using the last available window
            preds = []
            window = fs[-WINDOW:].copy()
            for _ in range(7):
                yhat = model.predict(window[np.newaxis, :, :], verbose=0).flatten()[0]
                units = ysc.inverse_transform([[yhat]])[0][0]
                preds.append(max(0, units))
                # roll window: append a synthetic next row (predicted units + last known exog)
                newrow = window[-1].copy(); newrow[0] = yhat
                window = np.vstack([window[1:], newrow])
            return float(np.sum(preds)), preds, "live LSTM model"
        except Exception as e:
            st.caption(f"(live model unavailable: {e}) — using precomputed forecast")
    # fallback: use precomputed forecast tail
    if fc is not None:
        last7 = fc["forecast"].tail(7).values
        return float(np.sum(last7)), list(last7), "precomputed forecast"
    return 7 * 130.0, [130.0] * 7, "demo default"

total7, daily7, src = next7_demand()

def reorder_decision(forecast_7d, current_stock, safety_stock=50):
    need = forecast_7d + safety_stock - current_stock
    if need <= 0:
        return "APPROVE", 0, "#157347"
    elif current_stock < safety_stock * 0.5:
        return "URGENT", int(round(need)), "#F59E0B"
    else:
        return "REORDER", int(round(need)), "#14B8A6"

verdict, qty, color = reorder_decision(total7, current_stock, safety_stock)

c1, c2, c3 = st.columns(3)
c1.metric("Forecast demand (next 7 days)", f"{total7:,.0f} units", help=f"source: {src}")
c2.metric("Current stock", f"{current_stock:,} units")
c3.metric("Suggested reorder qty", f"{qty:,} units")

msg = {"APPROVE": "Stock covers forecast + buffer. No action needed.",
       "REORDER": "Shortfall detected. Place a standard reorder.",
       "URGENT":  "Stock near empty with high demand. Flag for same-day delivery!"}[verdict]
st.markdown(f"<div class='verdict' style='background:{color}'>VERDICT: {verdict} — {msg}</div>",
            unsafe_allow_html=True)

with st.expander("See the next-7-day forecast breakdown"):
    st.bar_chart(pd.DataFrame({"day": [f"D{i+1}" for i in range(7)],
                               "forecast units": [round(x) for x in daily7]}).set_index("day"))

st.divider()
st.caption("Formula:  reorder need = forecast(7d) + safety stock − current stock  ·  "
           "Rules are fully explainable so a manager can trust every recommendation.")
