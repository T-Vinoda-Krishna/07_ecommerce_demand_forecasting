# E-commerce Demand Forecasting

## Business problem
A retailer needs better forecasts to reduce stockouts and excess inventory.

## Objective
Forecast daily demand for a product/category using historical sales and calendar features.

## Methods
- Baseline: seasonal naive
- Statistical: Exponential Smoothing
- ML: Random Forest on lag/rolling features

## Evaluation
Use a time-ordered split. Report MAE, RMSE, and MAPE.

## Demo
```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```
