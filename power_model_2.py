import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

dcgm = pd.read_csv("dcgm.csv", dtype={"id_job": "int64"})
sched = pd.read_csv("scheduler_data.csv", usecols=["id_job", "id_user"], dtype={"id_job": "int64"})
dup = sched.loc[sched.duplicated("id_job", keep=False), "id_job"].unique()
sched = sched[~sched["id_job"].isin(dup)]
df = dcgm.merge(sched, on="id_job", how="inner")
df = df[(df["totalexecutiontime_sec"] >= 60) & (df["powerusage_watts_avg"] > 0)].reset_index(drop=True)
print("GPU rows used:", len(df))

y, groups = df["powerusage_watts_avg"].values, df["id_user"].values
sets = {
    "A_sm_only": ["smutilization_pct_avg"],
    "B_sm_plus_mem": ["smutilization_pct_avg", "memoryutilization_pct_avg"],
    "C_all": ["smutilization_pct_avg", "smutilization_pct_max", "smutilization_pct_min", "avgsmutilization_pct",
              "avgmemoryutilization_pct", "memoryutilization_pct_avg", "memoryutilization_pct_max",
              "memoryutilization_pct_min", "maxgpumemoryused_bytes"],
}

def make_model():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=20,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1)

print("\nCross-validated, test users unseen in training:")
for name, cols in sets.items():
    oof = np.zeros(len(df))
    for tr, te in GroupKFold(n_splits=5).split(df, y, groups):
        m = make_model()
        m.fit(df[cols].iloc[tr], y[tr])
        oof[te] = m.predict(df[cols].iloc[te])
    print(f"{name:15s} R2={r2_score(y, oof):6.3f}  MAE={np.mean(np.abs(y - oof)):5.1f} W")

# Train on all rows and save each model for later use in EcoOrchestrator
for name, cols in sets.items():
    m = make_model()
    m.fit(df[cols], y)
    m.save_model(f"power_{name}.json")
print("\nsaved: power_A_sm_only.json, power_B_sm_plus_mem.json, power_C_all.json")

# Example of how the orchestrator would use model B
mB = XGBRegressor()
mB.load_model("power_B_sm_plus_mem.json")

def estimate_energy(sm_util_pct, mem_util_pct, seconds):
    X = pd.DataFrame({"smutilization_pct_avg": [sm_util_pct], "memoryutilization_pct_avg": [mem_util_pct]})
    watts = float(mB.predict(X)[0])
    return watts, watts * seconds / 3.6e6      # watts, kWh

for sm, mem, secs in [(0, 0, 600), (40, 10, 600), (90, 40, 600)]:
    w, kwh = estimate_energy(sm, mem, secs)
    print(f"SM {sm:3d}%  mem {mem:3d}%  {secs}s  ->  {w:6.1f} W  {kwh:.4f} kWh")