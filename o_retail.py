import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting",
    layout="wide"
)


st.title("Retail Sales Forecasting Dashboard")
st.markdown(
    """
    This dashboard demonstrates a **sales forecasting system** built using historical retail data.
    The goal is to support **inventory planning, staffing, and cash-flow decisions**.
    """
)

# ----------------------------
# Load & clean data
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("Online Retail.xlsx")

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

    df['Sales'] = df['Quantity'] * df['UnitPrice']

    monthly_sales = (
        df.set_index('InvoiceDate')
          .resample('M')['Sales']
          .sum()
    )

    return monthly_sales

monthly_sales = load_data()

# ----------------------------
# Moving Average
# ----------------------------
moving_avg_3 = monthly_sales.rolling(window=3).mean()

# ----------------------------
# Train-test split
# ----------------------------
train_size = int(len(monthly_sales) * 0.8)
train = monthly_sales.iloc[:train_size]
test = monthly_sales.iloc[train_size:]

# ----------------------------
# ARIMA Model
# ----------------------------
model = SARIMAX(
    train,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False
)

model_fit = model.fit(disp=False)

# ----------------------------
# Evaluation
# ----------------------------
predictions = model_fit.predict(
    start=test.index[0],
    end=test.index[-1]
)

mae = mean_absolute_error(test, predictions)
avg_sales = monthly_sales.mean()
mae_pct = (mae / avg_sales) * 100

# ----------------------------
# Forecast future
# ----------------------------
future_steps = 6
forecast = model_fit.get_forecast(steps=future_steps)
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

forecast_table = pd.DataFrame({
    "Forecasted Sales": forecast_values.round(2),
    "Lower CI": conf_int.iloc[:, 0].round(2),
    "Upper CI": conf_int.iloc[:, 1].round(2)
})

forecast_table.index = forecast_table.index.strftime("%Y-%m")

# ============================
# DASHBOARD SECTIONS
# ============================

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Average Monthly Sales", f"{avg_sales:,.0f}")
col2.metric("MAE", f"{mae:,.0f}")
col3.metric("MAE (% of Sales)", f"{mae_pct:.1f}%")

# ----------------------------
# Sales Trend
# ----------------------------
st.subheader("Monthly Sales Trend")

fig1, ax1 = plt.subplots(figsize=(8,4))
ax1.plot(monthly_sales, label="Monthly Sales", marker='o')
ax1.plot(moving_avg_3, label="3-Month Moving Average", linestyle='--')
ax1.set_xlabel("Date")
ax1.set_ylabel("Sales")
ax1.legend()
st.pyplot(fig1)

# ----------------------------
# Forecast Plot
# ----------------------------
st.subheader("Sales Forecast (Next 6 Months)")

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(monthly_sales, label="Historical Sales")
ax2.plot(forecast_values, label="Forecast", linestyle="--")
ax2.fill_between(
    conf_int.index,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    alpha=0.3
)
ax2.set_xlabel("Date")
ax2.set_ylabel("Sales")
ax2.legend()
st.pyplot(fig2)

# ----------------------------
# Forecast Table
# ----------------------------
st.subheader("Forecast Summary Table")
st.table(forecast_table)

# --- Derived business metrics ---
avg_forecast_sales = forecast_values.mean()
peak_sales_date = forecast_values.idxmax()
peak_sales_value = forecast_values.max()

st.subheader("Forecast Highlights")



col1, col2 = st.columns(2)

col1.metric(
    label="Average Forecasted Monthly Sales",
    value=f"{avg_forecast_sales:,.0f}"
)

col2.metric(
    label="Peak Forecast Month",
    value=peak_sales_date.strftime("%Y-%m")
)

# ----------------------------
# Business Interpretation
# ----------------------------
st.subheader("Business Insights")

st.markdown(
    """
    - The forecast captures **overall sales direction**, useful for strategic planning.
    - Due to **limited historical data**, the model focuses on trend rather than seasonality.
    - The MAE represents ~60% of average monthly sales, which is acceptable for **high-level planning**.
    """
)
