import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
st.title("Sales Forecasting with Prophet")
st.sidebar.header(" Select Forecasting Period")
period = st.sidebar.slider("Days to Forecast", 30, 365, 60)
np.random.seed(42)
date_range = pd.date_range(start='2022-01-01', periods=500, freq='D')
df = pd.DataFrame({
    'ds': date_range,
    'y': np.random.randint(5000, 15000, size=len(date_range))  # Sales data
})
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)
st.subheader("🔍 Forecasted Data")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))
st.subheader(" Forecast Plot")
fig1 = model.plot(forecast)
st.pyplot(fig1)
st.subheader(" Trend, Seasonality & Holidays Impact")
fig2 = model.plot_components(forecast)
st.pyplot(fig2)

st.success("Forecasting Completed! Adjust the forecast period using the sidebar.")

