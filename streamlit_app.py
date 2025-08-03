import streamlit as st
import pandas as pd

st.title("AI IPO Portfolio Dashboard")

st.markdown("## Portfolio Allocation")
try:
    allocation = pd.read_csv("results/allocation.csv", index_col=0)
    st.bar_chart(allocation)
except Exception as e:
    st.warning("Allocation data not found. Run pipeline first.")

st.markdown("## Drift Report")
try:
    with open("results/drift_report.html", "r") as f:
        drift_html = f.read()
    st.components.v1.html(drift_html, height=600, scrolling=True)
except Exception as e:
    st.warning("Drift report not available.")

st.markdown("## Backtest Plot")
try:
    st.image("results/backtest_equity_curve.png")
except:
    st.warning("Backtest plot not found.")