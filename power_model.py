import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

dcgm = pd.read_csv("dcgm.csv", dtype={"id_job": "int64"})
sched = pd.read_csv("scheduler_data.csv", usecols=["id_job", "id_user"], dtype={"id_job": "int64"})

dup = sched.loc[sched.duplicated("id_job", keep=False), "id_job"].unique()
sched = sched[~sched["id_job"].isin(dup)]
df = dcgm.merge(sched, on="id_job", how="inner")
df = df[(df["totalexecutiontime_sec"] >= 60) & (df["powerusage_watts_avg"] > 0)].copy()
print("GPU rows used:", len(df), "| jobs:", df["id_job"].nunique(), "| users:", df["id_user"].nunique())

# Intuition: how does GPU power change with SM utilization?
bins = pd.cut(df["smutilization_pct_avg"], [-1, 0, 10, 30, 60, 100])
print("\nmedian power (W) by average SM utilization:")
print(df.groupby(bins, observed=True)["powerusage_watts_avg"].agg(["count", "median"]).round(1))

feats = ["smutilization_pct_avg", "smutilization_pct_max", "smutilization_pct_min", "avgsmutilization_pct",
         "avgmemoryutilization_pct", "memoryutilization_pct_avg", "memoryutilization_pct_max",
         "memoryutilization_pct_min", "maxgpumemoryused_bytes"]
X, y, groups = df[feats], df["powerusage_watts_avg"].values, df["id_user"].values

oof, base = np.zeros(len(df)), np.zeros(len(df))
for tr, te in GroupKFold(n_splits=5).split(X, y, groups):      # test users are unseen in training
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=20,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1)
    m.fit(X.iloc[tr], y[tr])
    oof[te] = m.predict(X.iloc[te])
    base[te] = np.median(y[tr])

print(f"\nconstant (train median): R2={r2_score(y, base):6.3f}  MAE={np.mean(np.abs(y - base)):5.1f} W")
print(f"XGBoost power model    : R2={r2_score(y, oof):6.3f}  MAE={np.mean(np.abs(y - oof)):5.1f} W")
print("\nfeature importance:\n", pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False).round(3))
print("\nenergy for a job = predicted power (W) x duration (s) / 3.6e6 -> kWh")