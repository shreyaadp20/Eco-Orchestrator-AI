"""
host_power_model.py - server (host) power from CPU usage and fan speed, trained on BMCDATA (PSU1 + PSU2 power).
The twin uses it for the CPU/host side of every GPU node. Saves host_power.json.
Run from the folder that contains the bmcdata folder:  python host_power_model.py
"""
import glob, io, os, warnings
import numpy as np, pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

DATA_DIR = "bmcdata/data"


def load_influx(path):
    """Read an InfluxDB export; return {device_id: DataFrame}."""
    blocks, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#group"):
                if cur: blocks.append(cur)
                cur = []
            elif line.startswith("#") or not line.strip():
                continue
            else:
                cur.append(line)
    if cur: blocks.append(cur)
    out = {}
    for b in blocks:
        df = pd.read_csv(io.StringIO("".join(b)))
        df = df.drop(columns=[c for c in ["Unnamed: 0", "result", "table", "_start", "_stop"] if c in df.columns])
        df["_time"] = pd.to_datetime(df["_time"])
        out.setdefault(str(df["device_id"].iloc[0]), []).append(df)
    return {k: pd.concat(v, ignore_index=True) for k, v in out.items()}


frames = []
for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
    d = load_influx(p)
    b, h = d.get("bmc"), d.get("host")
    if b is None or h is None or "cpu_usage" not in h.columns:
        continue
    b = b.dropna(subset=["_time"])[["_time", "FAN1", "FAN2", "FAN3", "FAN4", "PSU1_Total_Power", "PSU2_Total_Power"]].sort_values("_time")
    h = h.dropna(subset=["_time"])[["_time", "cpu_usage"]].sort_values("_time")
    m = pd.merge_asof(b, h, on="_time", direction="nearest", tolerance=pd.Timedelta("15s"))
    m["file"] = os.path.basename(p)
    frames.append(m)
df = pd.concat(frames, ignore_index=True)

for c in ("PSU1_Total_Power", "PSU2_Total_Power"):                     # some rows store microwatts
    df[c] = df[c].where(df[c] <= 1e6, df[c] / 1e6)
df["fan_mean"] = df[["FAN1", "FAN2", "FAN3", "FAN4"]].mean(axis=1)
df["psu_w"] = df["PSU1_Total_Power"] + df["PSU2_Total_Power"]
df = df.dropna(subset=["cpu_usage", "fan_mean", "psu_w"]).reset_index(drop=True)
print("rows:", len(df), "| files:", df["file"].nunique())

cols = ["cpu_usage", "fan_mean"]
def make():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=20, subsample=0.8,
                        colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=0,
                        monotone_constraints=(1, 1))      # more CPU load or faster fans never lowers power

y, files = df["psu_w"].values, df["file"].values
oof, base = np.zeros(len(df)), np.zeros(len(df))
for tr, te in GroupKFold(n_splits=5).split(df, y, files):
    oof[te] = make().fit(df[cols].iloc[tr], y[tr]).predict(df[cols].iloc[te])
    base[te] = np.median(y[tr])
print(f"held-out runs: constant R2={r2_score(y, base):.3f} MAE={np.mean(np.abs(y - base)):.1f} W | "
      f"model R2={r2_score(y, oof):.3f} MAE={np.mean(np.abs(y - oof)):.1f} W")

m = make().fit(df[cols], y)
m.save_model("host_power.json")
P = lambda u, f: round(float(m.predict(pd.DataFrame({"cpu_usage": [u], "fan_mean": [f]}))[0]), 0)
print("host power (W): idle (cpu 2%, fans 1500):", P(2, 1500), "| busy (cpu 95%, fans 6000):", P(95, 6000))
print("saved host_power.json")