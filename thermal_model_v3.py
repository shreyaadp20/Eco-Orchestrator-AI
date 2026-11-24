import glob, io, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

DATA_DIR = "bmcdata/data"          # folder created by: git clone https://github.com/arealuser/bmcdata.git

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

BMC = ["Cpu1_Temp", "Cpu2_Temp", "Inlet_Temp", "Outlet_Temp", "Chipset_Temp",
       "FAN1", "FAN2", "FAN3", "FAN4", "PSU1_Total_Power", "PSU2_Total_Power"]
HOST = ["cpu_usage", "load_1", "mem_usage"]

# ---- 1. Build one table: BMC readings + nearest host reading, per file
frames = []
for p in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
    d = load_influx(p)
    b, h = d.get("bmc"), d.get("host")
    if b is None or h is None:
        continue
    b = b.dropna(subset=["_time"])[["_time"] + BMC].sort_values("_time")
    h = h.dropna(subset=["_time"])[["_time"] + [c for c in HOST if c in h.columns]].sort_values("_time")
    m = pd.merge_asof(b, h, on="_time", direction="nearest", tolerance=pd.Timedelta("15s"))
    m["file"] = os.path.basename(p)
    frames.append(m)
df = pd.concat(frames, ignore_index=True)
for c in HOST:
    if c not in df.columns:
        df[c] = np.nan
print("files:", df["file"].nunique(), "| BMC rows:", len(df))

# ---- 1b. Unit fix: some rows store temperatures x1000 (millidegrees) and PSU power x1e6 (microwatts)
TEMPS = ["Cpu1_Temp", "Cpu2_Temp", "Inlet_Temp", "Outlet_Temp", "Chipset_Temp"]
PSUS = ["PSU1_Total_Power", "PSU2_Total_Power"]
scaled = df[TEMPS[0]] > 1000
print("rows stored in milli-units:", int(scaled.sum()), "in", df.loc[scaled, "file"].nunique(), "files")
for c in TEMPS:
    df[c] = df[c].where(df[c] <= 1000, df[c] / 1000)
for c in PSUS:
    df[c] = df[c].where(df[c] <= 1e6, df[c] / 1e6)
print(df[TEMPS + PSUS].describe().loc[["min", "50%", "max"]].round(1).to_string())


# ---- 2. Recent-change features (computed inside each file, never across files)
g = df.groupby("file")
lag_cols = ["Cpu1_Temp", "Cpu2_Temp", "cpu_usage", "PSU1_Total_Power"]
for c in lag_cols:
    for k in (1, 2):
        df[f"{c}_chg{k}"] = df[c] - g[c].shift(k)
feats = BMC + HOST + [f"{c}_chg{k}" for c in lag_cols for k in (1, 2)]

# ---- 3. Predict the temperature CHANGE H readings ahead; the baseline "no change" is persistence
def static_model(psu):
    """Physically sensible: hotter with more CPU usage, load, inlet temperature and PSU power; cooler with more fan speed."""
    mono = (1, 1, 1, -1) + ((1, 1) if psu else ())
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=20,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1,
                        random_state=0, monotone_constraints=mono)


def model():
    return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, min_child_weight=20,
                        subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1)

print("\nHeld-out FILES (unseen runs). MAE in degrees C; 'persistence' = predict no change.")
for H in (6, 30, 90):
    gap = g["_time"].shift(-H) - df["_time"]
    for T in ("Cpu1_Temp", "Cpu2_Temp"):
        y = g[T].shift(-H) - df[T]
        mask = y.notna() & (gap <= pd.Timedelta(seconds=40 * H))
        X, yy, files = df.loc[mask, feats], y[mask].values, df.loc[mask, "file"].values
        oof = np.zeros(len(yy))
        for tr, te in GroupKFold(n_splits=5).split(X, yy, files):
            m = model().fit(X.iloc[tr], yy[tr])
            oof[te] = m.predict(X.iloc[te])
        mae_m, mae_p = np.mean(np.abs(yy - oof)), np.mean(np.abs(yy))
        secs = gap[mask].dt.total_seconds().median()
        print(f"{T} +{H} step (~{secs:4.0f} s ahead): n={len(yy):6d}  persistence={mae_p:5.2f}  model={mae_m:5.2f}  "
              f"improvement={100 * (1 - mae_m / mae_p):5.1f}%")

# ---- 4. Static thermal model: temperature LEVEL from current load, inlet air and fans (held-out files)
from sklearn.metrics import r2_score
print("\nStatic model: CPU temperature from load, inlet temperature and fans (held-out files)")
df["fan_mean"] = df[["FAN1", "FAN2", "FAN3", "FAN4"]].mean(axis=1)
sets = {"load+inlet+fans": ["cpu_usage", "load_1", "Inlet_Temp", "fan_mean"],
        "+PSU power": ["cpu_usage", "load_1", "Inlet_Temp", "fan_mean", "PSU1_Total_Power", "PSU2_Total_Power"]}
for T in ("Cpu1_Temp", "Cpu2_Temp"):
    sub = df.dropna(subset=[T])
    y, files = sub[T].values, sub["file"].values
    base = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(sub, y, files):
        base[te] = np.median(y[tr])
    print(f"{T}  {'constant':16s} R2={r2_score(y, base):6.3f}  MAE={np.mean(np.abs(y - base)):5.2f} C")
    for name, cols in sets.items():
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(sub, y, files):
            m = static_model('PSU' in name).fit(sub[cols].iloc[tr], y[tr])
            oof[te] = m.predict(sub[cols].iloc[te])
        print(f"{T}  {name:16s} R2={r2_score(y, oof):6.3f}  MAE={np.mean(np.abs(y - oof)):5.2f} C")
        tag = "psu" if "PSU" in name else "load"
        static_model('PSU' in name).fit(sub[cols], y).save_model(f"thermal_{T[:4].lower()}_{tag}.json")   # trained on all rows
print("\nsaved: thermal_cpu1_load.json, thermal_cpu1_psu.json, thermal_cpu2_load.json, thermal_cpu2_psu.json")
print("inputs, in this order -> load models: cpu_usage, load_1, Inlet_Temp, fan_mean")
print("                         psu models : cpu_usage, load_1, Inlet_Temp, fan_mean, PSU1_Total_Power, PSU2_Total_Power")