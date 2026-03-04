import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor

# Load trained XGBoost model
model = joblib.load("receipt_forecast_model_xgb.pkl")



def get_key(dictionary, target_value):
    return [key for key, value in dictionary.items() if value == target_value][0]


st.write("""
# Fetch Rewards: Monthly Receipt Count Forecaster
*Monthly Receipt Counts in the year 2021*
""")

# Load and preprocess data
data_daily = pd.read_csv("data_daily.csv")
data_daily['# Date'] = pd.to_datetime(data_daily['# Date'])
data_monthly = data_daily.groupby(data_daily['# Date'].dt.to_period("M"))['Receipt_Count'].sum().reset_index()
data_monthly.columns = ['Month', 'Receipt_Count']
data_monthly['Month'] = data_monthly['Month'].astype(str)

# Normalization parameters (save from training)
max_val = data_monthly['Receipt_Count'].max()
min_val = data_monthly['Receipt_Count'].min()
data_normalized = (data_monthly['Receipt_Count'] - min_val) / (max_val - min_val)

# Use last 3 normalized values as initial input
forecast = data_normalized[-3:].tolist()

# UI Layout
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(data_monthly['Month'], data_monthly['Receipt_Count'], marker='o', linestyle='-')
    ax.set_title('Monthly Receipt Counts for 2021')
    ax.set_xlabel('Month')
    ax.set_ylabel('Receipt Count')
    ax.grid(True)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()

    st.pyplot(fig) 


with col2:
    st.write(data_monthly)

# Sidebar for month selection
month_dict = {
    "January" : 1, "February":2 , "March":3 , "April":4, "May":5, "June":6,
    "July":7, "August":8, "September":9, "October":10, "November":11, "December":12
}

with st.sidebar:
    st.title("2022 Monthly Receipt Count Forecaster: ")
    st.markdown("---")
    st.write("*How it works:*")
    st.write("When you select a month in 2022, the forecaster predicts receipt counts from January to the selected month.")
    st.write("*Note:* January 2022 is selected by default.")
    st.markdown("---")

    month = st.selectbox("Select Month in 2022 to forecast:", list(month_dict.keys()))

    for _ in range(month_dict[month]):
        X = np.array(forecast[-3:]).reshape(1, -1)  # use last 3 values as input
        next_val = model.predict(X)[0]
        forecast.append(next_val)

    forecasted_values = forecast[-month_dict[month]:]
    forecast_denorm = np.array(forecasted_values) * (max_val - min_val) + min_val

    for i, value in enumerate(forecast_denorm):
        st.write(f"{get_key(month_dict, i+1)}'s receipt count = {value:.0f}")