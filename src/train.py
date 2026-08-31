import pandas as pd, numpy as np, joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/"data/demand.csv",parse_dates=["date"]).sort_values("date")
for lag in [1,7,14,28]:
    df[f"lag_{lag}"]=df.demand.shift(lag)
for w in [7,28]:
    df[f"roll_{w}"]=df.demand.shift(1).rolling(w).mean()
df=df.dropna().reset_index(drop=True)

features=[c for c in df.columns if c not in ["date","demand"]]
split=int(len(df)*.8)
tr=df.iloc[:split]; te=df.iloc[split:]
model=RandomForestRegressor(n_estimators=400,random_state=42,max_depth=12)
model.fit(tr[features],tr.demand)
pred=model.predict(te[features])
mae=mean_absolute_error(te.demand,pred)
rmse=mean_squared_error(te.demand,pred)**0.5
mape=np.mean(np.abs((te.demand-pred)/te.demand))*100
print({"MAE":round(mae,2),"RMSE":round(rmse,2),"MAPE":round(mape,2)})
joblib.dump((model,features),ROOT/"model.joblib")
out=te[["date","demand"]].copy()
out["prediction"]=pred
out.to_csv(ROOT/"data/test_predictions.csv",index=False)
