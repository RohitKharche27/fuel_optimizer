import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Fuel Price Prediction", layout="centered")

st.title("⛽ Fuel Price / Demand Prediction App")

# Load model
@st.cache_resource
def load_model():
    with open("fuel_prediction.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Load dataset (optional – for reference)
@st.cache_data
def load_data():
    return pd.read_csv("oil.csv")

df = load_data()

st.subheader("📊 Sample Data")
st.dataframe(df.head())

st.subheader("🔢 Enter Input Features")

# ---- INPUT FIELDS (adjust based on your model features) ----
price = st.number_input("Current Price", value=100.0)
cost = st.number_input("Cost", value=80.0)
comp_avg_price = st.number_input("Competitor Avg Price", value=98.0)

price_vs_comp = price - comp_avg_price
price_vs_cost = price - cost

price_lag_1 = st.number_input("Price Lag 1", value=99.0)
price_lag_7 = st.number_input("Price Lag 7", value=97.0)

price_ma_7 = st.number_input("7 Day Moving Avg", value=98.5)
price_ma_14 = st.number_input("14 Day Moving Avg", value=97.8)

day_of_week = st.selectbox("Day of Week (0=Mon)", list(range(7)))
month = st.selectbox("Month", list(range(1, 13)))

# Create input array
input_data = np.array([[ 
    price, cost, comp_avg_price,
    price_vs_comp, price_vs_cost,
    price_lag_1, price_lag_7,
    price_ma_7, price_ma_14,
    day_of_week, month
]])

# Prediction
if st.button("🔮 Predict"):
    prediction = model.predict(input_data)
    st.success(f"✅ Predicted Value: **{prediction[0]:.2f}**")
