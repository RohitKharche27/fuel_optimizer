import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import date
import os

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(page_title="Fuel Price Optimization", layout="wide")

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "run" not in st.session_state:
    st.session_state.run = False

# ----------------------------------
# DARK THEME
# ----------------------------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
h1, h2, h3 { color: white; }
.stButton>button {
    background: linear-gradient(90deg,#ff4b4b,#ff6b6b);
    color: white;
    font-size: 18px;
    height: 3em;
    border-radius: 10px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------------
# TITLE
# ----------------------------------
st.title("⛽ Fuel Price Optimization – ML Based Recommender")
st.caption("Recommend optimal daily fuel price using ML predictions")

# ----------------------------------
# LOAD DATA
# ----------------------------------
DATA_PATH = "oil.csv"

@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error("❌ oil.csv not found")
        st.stop()
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df.sort_values("date", inplace=True)
    return df

df = load_data()

# ----------------------------------
# FEATURE ENGINEERING
# ----------------------------------
def feature_engineering(df):
    df = df.copy()
    df["comp_avg_price"] = df[["comp1_price","comp2_price","comp3_price"]].mean(axis=1)
    df["price_vs_comp"] = df["price"] - df["comp_avg_price"]
    df["price_vs_cost"] = df["price"] - df["cost"]
    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_7"] = df["price"].shift(7)
    df["price_ma_7"] = df["price"].rolling(7).mean()
    df["price_ma_14"] = df["price"].rolling(14).mean()
    df["day_of_week"] = df["date"].dt.weekday
    df["month"] = df["date"].dt.month
    df.dropna(inplace=True)
    df["demand_proxy"] = -df["price_vs_comp"] + 0.5 * df["price_ma_7"]
    return df

df_fe = feature_engineering(df)

# ----------------------------------
# TRAIN MODEL
# ----------------------------------
@st.cache_resource
def train_model(df):
    FEATURES = [
        "price","cost","comp_avg_price",
        "price_vs_comp","price_vs_cost",
        "price_lag_1","price_lag_7",
        "price_ma_7","price_ma_14",
        "day_of_week","month"
    ]

    X = df[FEATURES]
    y = df["demand_proxy"]

    X_train, _, y_train, _ = train_test_split(X, y, shuffle=False)

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, FEATURES

model, FEATURES = train_model(df_fe)

# ----------------------------------
# SIDEBAR RULES
# ----------------------------------
st.sidebar.header("🎯 Business Rules")

MAX_DAILY_CHANGE = st.sidebar.slider("Max price change (₹)", 0.1, 2.0, 0.75)
MIN_MARGIN = st.sidebar.slider("Min margin (₹)", 0.1, 3.0, 0.50)
MAX_COMP_DIFF = st.sidebar.slider("Max above competitors (₹)", 0.1, 3.0, 1.50)

# ----------------------------------
# INPUTS
# ----------------------------------
st.subheader("📥 Today's Market Inputs")

c1, c2 = st.columns(2)

with c1:
    input_date = st.date_input("Date", date.today())
    last_price = st.number_input("Last Company Price (₹)", value=float(df["price"].iloc[-1]))
    cost = st.number_input("Today's Cost (₹)", value=float(df["cost"].iloc[-1]))

with c2:
    comp1 = st.number_input("Competitor 1 (₹)", value=float(df["comp1_price"].iloc[-1]))
    comp2 = st.number_input("Competitor 2 (₹)", value=float(df["comp2_price"].iloc[-1]))
    comp3 = st.number_input("Competitor 3 (₹)", value=float(df["comp3_price"].iloc[-1]))

# ----------------------------------
# ANALYSIS FUNCTION
# ----------------------------------
def build_analysis_df():
    comp_avg = np.mean([comp1, comp2, comp3])
    prices = np.linspace(last_price - MAX_DAILY_CHANGE,
                         last_price + MAX_DAILY_CHANGE, 25)

    rows = []
    for p in prices:
        if p < cost + MIN_MARGIN or p > comp_avg + MAX_COMP_DIFF:
            continue

        row = pd.DataFrame([{
            "price": p,
            "cost": cost,
            "comp_avg_price": comp_avg,
            "price_vs_comp": p - comp_avg,
            "price_vs_cost": p - cost,
            "price_lag_1": last_price,
            "price_lag_7": last_price,
            "price_ma_7": last_price,
            "price_ma_14": last_price,
            "day_of_week": input_date.weekday(),
            "month": input_date.month
        }])[FEATURES]

        demand = model.predict(row)[0]
        profit = (p - cost) * demand

        rows.append({
            "Price (₹)": round(p,2),
            "Expected Volume": int(demand),
            "Profit Margin": round(p-cost,2),
            "Expected Profit": round(profit,2)
        })

    return pd.DataFrame(rows)

# ----------------------------------
# BUTTON
# ----------------------------------
if st.button("🔍 Run Price Optimization"):
    st.session_state.run = True

# ----------------------------------
# RESULTS
# ----------------------------------
if st.session_state.run:
    df_analysis = build_analysis_df()
    best = df_analysis.loc[df_analysis["Expected Profit"].idxmax()]

    st.success("✅ Price Optimization Completed")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Recommended Price", f"₹{best['Price (₹)']}")
    k2.metric("Expected Volume", f"{best['Expected Volume']:,} L")
    k3.metric("Expected Profit", f"₹{best['Expected Profit']:,}")
    k4.metric("Profit Margin", f"₹{best['Profit Margin']}/L")

    st.subheader("📊 Price Optimization Analysis")

    tab1, tab2, tab3 = st.tabs(["📈 Profit Curve", "📊 Volume vs Price", "📄 Data Table"])

    with tab1:
        st.line_chart(df_analysis.set_index("Price (₹)")["Expected Profit"])

    with tab2:
        st.bar_chart(df_analysis.set_index("Price (₹)")["Expected Volume"])

    with tab3:
        st.dataframe(df_analysis, use_container_width=True)

st.caption("⚠️ ML-based recommendation. Always validate with business context.")
