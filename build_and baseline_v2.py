import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

pd.set_option("display.max_columns", None, "display.width", 200)
ONLY_COMPLETED = False   # try True later: keep only state == 3

dcgm = pd.read_csv("dcgm.csv", dtype={"id_job": "int64"})
sched = pd.read_csv("scheduler_data.csv", dtype={"id_job": "int64"})

# ---- 1. Clean the scheduler table
dup_ids = sched.loc[sched.duplicated("id_job", keep=False), "id_job"].unique()
print("dropping requeued/duplicated job ids:", len(dup_ids))
sched = sched[~sched["id_job"].isin(dup_ids)].copy()
for c in ["time_submit", "time_eligible", "time_start", "time_end"]:
    sched[c] = sched[c].where((sched[c] >= 0) & (sched[c] < 4e7))
sched["mem_req"] = sched["mem_req"].where(sched["mem_req"] < 1e18)
sched["timelimit"] = sched["timelimit"].where(sched["timelimit"] < 1e8)
sched["gpus_req"] = pd.to_numeric(sched["gres_req"].astype(str).str.extract(r"(\d+)\s*$")[0], errors="coerce")

# ---- 2. One row per job from DCGM (each dcgm row = one GPU)
dcgm["energy_pw_j"] = dcgm["powerusage_watts_avg"] * dcgm["totalexecutiontime_sec"]
per_job = dcgm.groupby("id_job").agg(
    energy_j=("energyconsumed_joules", "sum"),
    energy_pw_j=("energy_pw_j", "sum"),
    n_gpus_seen=("gpu_id", "size"),
    runtime_s=("totalexecutiontime_sec", "max"),
    avg_power_w=("powerusage_watts_avg", "mean"),
).reset_index()
per_job["ratio"] = (per_job["energy_j"] / per_job["energy_pw_j"]).replace([np.inf, -np.inf], np.nan)

# ---- 3. Join, filter, and diagnose the recorded-energy problem
df = per_job.merge(sched, on="id_job", how="inner")
df = df[(df["runtime_s"] >= 60) & (df["avg_power_w"] > 0)].copy()
print("jobs after join + runtime >= 60 s + power > 0:", len(df))

df["ok"] = df["ratio"].between(0.5, 1.5)
print("share of jobs whose recorded energy matches power x runtime:", round(df["ok"].mean(), 3))
print("\nmedians, split by whether recorded energy is consistent (ok):")
print(df.groupby("ok")[["runtime_s", "n_gpus_seen", "avg_power_w", "energy_j", "energy_pw_j"]].median().round(1))
wall = df["time_end"] - df["time_start"]
print("\nmedian of runtime_s / scheduler wall time:", round((df["runtime_s"] / wall).median(), 3))

# Target: power x runtime, summed over GPUs (matches recorded energy where the counter is consistent)
df["target_j"] = df["energy_pw_j"]

# ---- 4. User-history features (previous jobs by the same user, in submit order)
df = df.dropna(subset=["time_submit"]).sort_values("time_submit").reset_index(drop=True)
if ONLY_COMPLETED:
    df = df[df["state"] == 3].reset_index(drop=True)
    print("completed only:", len(df))
df["y"] = np.log1p(df["target_j"])
df["log_rt"] = np.log1p(df["runtime_s"])
g = df.groupby("id_user")
df["user_prev_n"] = g.cumcount()
df["user_prev_y"] = g["y"].transform(lambda s: s.shift(1).expanding().mean())
df["user_prev_logrt"] = g["log_rt"].transform(lambda s: s.shift(1).expanding().mean())

# ---- 5. Time-ordered split: train on the earliest 80%, test on the latest 20%
num = ["cpus_req", "gpus_req", "mem_req", "timelimit"]
cat = ["partition", "job_type", "constraints"]
hist = ["user_prev_n", "user_prev_y", "user_prev_logrt"]
cut = int(len(df) * 0.8)
y_tr, y_te = df["y"].iloc[:cut].values, df["y"].iloc[cut:].values
print(f"\ntrain jobs: {cut}, test jobs: {len(df) - cut}")

def report(name, pred):
    true_j, pred_j = np.expm1(y_te), np.expm1(pred)
    print(f"{name:32s} R2(log)={r2_score(y_te, pred):6.3f}  "
          f"MAE={np.mean(np.abs(pred_j - true_j)) / 3600:7.2f} Wh  "
          f"median %err={np.median(np.abs(pred_j - true_j) / true_j) * 100:6.1f}")

def run(name, cols):
    X = df[cols].copy()
    for c in cat:
        if c in cols:
            X[c] = X[c].astype("category")
    m = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=20,
                     subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
                     tree_method="hist", n_jobs=-1)
    m.fit(X.iloc[:cut], y_tr)
    report(name, m.predict(X.iloc[cut:]))
    return m, X

report("constant (train median)", np.full(len(y_te), np.median(y_tr)))
run("XGBoost, job request only", num + cat)
m, X = run("XGBoost, request + user history", num + cat + hist)
print("\nfeature importance (last model):\n",
      pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False).round(3))