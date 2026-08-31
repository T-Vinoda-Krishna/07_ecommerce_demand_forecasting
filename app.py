import streamlit as st, pandas as pd, joblib, plotly.express as px
from pathlib import Path

ROOT=Path(__file__).parent
st.title("E-commerce Demand Forecasting")
st.caption("Synthetic daily demand series.")

model_path=ROOT/"model.joblib"
if not model_path.exists():
    st.warning("Run: python src/train.py")
    st.stop()

pred=pd.read_csv(ROOT/"data/test_predictions.csv",parse_dates=["date"])
st.plotly_chart(px.line(pred.tail(180),x="date",y=["demand","prediction"],title="Actual vs predicted demand"),use_container_width=True)

mae=(pred.demand-pred.prediction).abs().mean()
rmse=((pred.demand-pred.prediction)**2).mean()**0.5
c1,c2=st.columns(2)
c1.metric("Test MAE",f"{mae:.2f}")
c2.metric("Test RMSE",f"{rmse:.2f}")

st.info("Business use: translate the forecast into reorder points, safety stock, and purchase recommendations rather than stopping at model accuracy.")
