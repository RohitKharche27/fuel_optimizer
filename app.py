import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from datetime import date
import os

if "run" not in st.session_state:
    st.session_state.run = False

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Fuel Price Optimization",
    layout="wide"
)

# ----------------------------------
# CUSTOM DARK THEME
# ----------------------------------
st.markdown("""
<style>
.stApp { background-color: #0e1117; color: white; }
h1, h2, h3, h4 { color: #ffffff; }
.stButton>button {
    background: linear-gradient(90deg,#ff4b4b,#ff6b6b);
    color: white;
    font-size: 18px;
    height: 3em;
    border-radius: 10px;
    width: 100%;
}
.stMetric {
    background-color: #161a23;
    padding: 15px;
    border-radius: 10px;
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
        st.error("❌ oil.csv not found in project folder")
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

    # Demand proxy
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
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model, FEATURES

model, FEATURES = train_model(df_fe)

# ----------------------------------
# SIDEBAR – BUSINESS RULES
# ----------------------------------
st.sidebar.header("🎯 Business Rules / Constraints")

MAX_DAILY_CHANGE = st.sidebar.slider(
    "Max price change per day (₹)",
    0.1, 2.0, 0.75
)

MIN_MARGIN = st.sidebar.slider(
    "Minimum margin per liter (₹)",
    0.1, 3.0, 0.50
)

MAX_COMP_DIFF = st.sidebar.slider(
    "Max price above competitors (₹)",
    0.1, 3.0, 1.50
)

st.sidebar.info("💡 Adjust constraints based on market strategy")

# ----------------------------------
# MAIN INPUTS
# ----------------------------------
st.subheader("📥 Enter Today's Market Inputs")

col1, col2 = st.columns(2)

with col1:
    input_date = st.date_input("Date", date.today())
    last_price = st.number_input("Last Observed Company Price (₹)", value=float(df["price"].iloc[-1]))
    cost = st.number_input("Today's Cost per Liter (₹)", value=float(df["cost"].iloc[-1]))

with col2:
    comp1 = st.number_input("Competitor 1 Price (₹)", value=float(df["comp1_price"].iloc[-1]))
    comp2 = st.number_input("Competitor 2 Price (₹)", value=float(df["comp2_price"].iloc[-1]))
    comp3 = st.number_input("Competitor 3 Price (₹)", value=float(df["comp3_price"].iloc[-1]))

# ----------------------------------
# PRICE OPTIMIZATION LOGIC
# ----------------------------------
def recommend_price():
    comp_avg = np.mean([comp1, comp2, comp3])

    candidate_prices = np.linspace(
        last_price - MAX_DAILY_CHANGE,
        last_price + MAX_DAILY_CHANGE,
        25
    )
def generate_analysis_data():
    comp_avg = np.mean([comp1, comp2, comp3])

    prices = np.linspace(
        last_price - MAX_DAILY_CHANGE,
        last_price + MAX_DAILY_CHANGE,
        30
    )

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
            "Price (₹)": round(p, 2),
            "Expected Volume (L)": int(demand),
            "Profit Margin (₹/L)": round(p - cost, 2),
            "Expected Profit (₹)": round(profit, 2)
        })

    return pd.DataFrame(rows)


    best_price, best_profit = None, -np.inf

    for p in candidate_prices:
        if p < cost + MIN_MARGIN:
            continue
        if p > comp_avg + MAX_COMP_DIFF:
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

        if profit > best_profit:
            best_profit = profit
            best_price = p

    return round(best_price, 2), round(best_profit, 2)

# ----------------------------------
# RUN BUTTON
# ----------------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Run Price Optimization"):
    st.session_state.run = True


  if st.session_state.run:

    # ----------------------------
    # RUN LOGIC
    # ----------------------------
    price, profit = recommend_price()
    analysis_df = generate_analysis_data()

    expected_volume = int(
        analysis_df.loc[
            analysis_df["Price (₹)"] == price,
            "Expected Volume (L)"
        ].values[0]
    )

    margin = round(price - cost, 2)

    # ----------------------------
    # SUCCESS BANNER
    # ----------------------------
    st.markdown(
        """
        <div style="
            background:#123f2a;
            padding:15px;
            border-radius:8px;
            color:#7CFFB2;
            font-size:16px;">
            ✅ Price Recommendation Generated Successfully
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------
    # KPI CARDS
    # ----------------------------
    k1, k2, k3, k4 = st.columns(4)

    k1.metric("Recommended Price", f"₹{price}", f"+{round(price-last_price,2)}")
    k2.metric("Expected Volume", f"{expected_volume:,} L")
    k3.metric("Expected Profit", f"₹{round(profit,2):,}")
    k4.metric("Profit Margin", f"₹{margin}/L")

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Price Optimization Analysis")

    # ----------------------------
    # TABS
    # ----------------------------
    tab1, tab2, tab3 = st.tabs(
        ["📈 Profit Curve", "📊 Volume vs Price", "📄 Data Table"]
    )

    # PROFIT CURVE
    with tab1:
        fig, ax = plt.subplots()
        ax.plot(
            analysis_df["Price (₹)"],
            analysis_df["Expected Profit (₹)"],
            marker="o"
        )
        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Expected Profit (₹)")
        st.pyplot(fig)

    # VOLUME VS PRICE
    with tab2:
        fig, ax = plt.subplots()
        ax.bar(
            analysis_df["Price (₹)"],
            analysis_df["Expected Volume (L)"]
        )
        ax.set_xlabel("Price (₹)")
        ax.set_ylabel("Expected Volume (L)")
        st.pyplot(fig)

    # DATA TABLE
    with tab3:
        st.dataframe(analysis_df, use_container_width=True)


