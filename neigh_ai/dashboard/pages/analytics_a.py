import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.title("Analytics")

# Generate fake time-series data
np.random.seed(42)
dates = pd.date_range("2024-01-01", periods=50)
values = np.cumsum(np.random.randn(50))

df = pd.DataFrame({
    "date": dates,
    "value": values,
})

fig = px.line(
    df,
    x="date",
    y="value",
    title="Random Time Series",
)

st.plotly_chart(fig, width="stretch")
