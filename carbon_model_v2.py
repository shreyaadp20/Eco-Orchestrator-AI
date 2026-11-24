import numpy as np, pandas as pd
from xgboost import XGBRegressor

CSV = "df_fuel_ckan.csv"
TRAIN_FROM = "2023-01-01"      # the grid decarbonised a lot since 2009, so old years mislead the model
TEST_DAYS = 180                # the final 180 days are the test period
H = 48                         # forecast up to 48 half-hours (24 h) ahead

cols = ["DATETIME", "CARBON_INTENSITY", "GAS_perc", "COAL_perc", "NUCLEAR_perc",
        "WIND_perc", "WIND_EMB_perc", "SOLAR_perc", "IMPORTS_perc"]
raw = pd.read_csv(CSV, usecols=cols)
raw["DATETIME"] = pd.to_datetime(raw["DATETIME"])
raw = raw.sort_values("DATETIME")

# ---- 1. Data checks
print("range:", raw["DATETIME"].min(), "->", raw["DATETIME"].max(), "| rows:", len(raw))
print("duplicate timestamps:", int(raw["DATETIME"].duplicated().sum()))
step = raw["DATETIME"].diff().value_counts().head(4)
print("most common time steps:\n", step.to_string())
print("\nmean carbon intensity by year (gCO2/kWh):")
print(raw.groupby(raw["DATETIME"].dt.year)["CARBON_INTENSITY"].mean().round(0).to_string())

# ---- 2. Clean: unique timestamps, regular 30-minute grid, recent years only
df = raw.drop_duplicates("DATETIME").set_index("DATETIME").asfreq("30min")
df = df[df.index >= TRAIN_FROM]
print("\nmissing rows on the 30-min grid since", TRAIN_FROM, ":", int(df["CARBON_INTENSITY"].isna().sum()))
df = df.interpolate(limit=4)

ci = df["CARBON_INTENSITY"].values
roll48 = df["CARBON_INTENSITY"].rolling(48).mean().values
wind = (df["WIND_perc"] + df["WIND_EMB_perc"]).values
solar, gas = df["SOLAR_perc"].values, df["GAS_perc"].values
imports, nuclear = df["IMPORTS_perc"].values, df["NUCLEAR_perc"].values
hour = (df.index.hour + df.index.minute / 60).values
dow, month = df.index.dayofweek.values, df.index.month.values

# ---- 3. Stack one row per (forecast origin, horizon). Origins every hour.
def build(origins):
    X, y, meta = [], [], []
    for h in range(1, H + 1):
        t = origins + h
        X.append(pd.DataFrame({
            "h": h, "ci_now": ci[origins], "ci_1h": ci[origins - 2], "ci_2h": ci[origins - 4],
            "ci_6h": ci[origins - 12], "ci_24h": ci[origins - 48], "ci_mean_24h": roll48[origins],
            "wind": wind[origins], "solar": solar[origins], "gas": gas[origins],
            "imports": imports[origins], "nuclear": nuclear[origins],
            "ci_yday_same_time": ci[t - 48], "ci_lastweek_same_time": ci[t - 336],
            "tgt_hour": hour[t], "tgt_dow": dow[t], "tgt_month": month[t]}))
        y.append(ci[t])
        meta.append(pd.DataFrame({"o": origins, "h": h}))
    return pd.concat(X, ignore_index=True), np.concatenate(y), pd.concat(meta, ignore_index=True)

cut = np.searchsorted(df.index.values, df.index.max() - np.timedelta64(TEST_DAYS, "D"))
origins = np.arange(336, len(df) - H, 2)
tr_o = origins[origins + H < cut]
te_o = origins[origins >= cut]
Xtr, ytr, _ = build(tr_o)
Xte, yte, mte = build(te_o)
ok = ~np.isnan(ytr) & ~np.isnan(Xtr["ci_now"].values)
print(f"\ntrain rows: {ok.sum():,} | test origins: {len(te_o):,} (last {TEST_DAYS} days)")

model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                     colsample_bytree=0.8, min_child_weight=50, tree_method="hist", n_jobs=-1)
# Predict the CHANGE from the current intensity, so the model can never do much worse than "no change"
model.fit(Xtr[ok], (ytr - Xtr["ci_now"].values)[ok])
pred = model.predict(Xte) + Xte["ci_now"].values

# ---- 4. Accuracy by horizon, against two simple baselines
res = pd.DataFrame({"o": mte["o"], "h": mte["h"], "actual": yte, "model": pred,
                    "persist": Xte["ci_now"].values, "yesterday": Xte["ci_yday_same_time"].values}).dropna()
print("\nMAE in gCO2/kWh on the test period:")
rows = []
for h in (1, 2, 4, 12, 24, 48):
    r = res[res["h"] == h]
    rows.append({"horizon": f"{h * 30} min", "persistence": (r.actual - r.persist).abs().mean(),
                 "same time yesterday": (r.actual - r.yesterday).abs().mean(),
                 "XGBoost": (r.actual - r.model).abs().mean()})
print(pd.DataFrame(rows).round(1).to_string(index=False))

# ---- 5. Does it help scheduling? Each test day at 00:00, a job can run in any of the next 24 hours.
res["is_midnight"] = hour[res["o"].values] == 0
day = res[res["is_midnight"]]
pick = day.loc[day.groupby("o")["model"].idxmin()]
now = day[day["h"] == 1].set_index("o")["actual"]
oracle = day.groupby("o")["actual"].min()
avg = day.groupby("o")["actual"].mean()
chosen = pick.set_index("o")["actual"]
print(f"\nScheduling, {len(chosen)} test days (gCO2/kWh, mean over days):")
print(f"  run immediately        : {now.mean():6.1f}")
print(f"  average hour in window : {avg.mean():6.1f}")
print(f"  model-picked hour      : {chosen.mean():6.1f}   ({100 * (1 - chosen.mean() / now.mean()):.1f}% below immediate)")
print(f"  best possible (oracle) : {oracle.mean():6.1f}   ({100 * (1 - oracle.mean() / now.mean()):.1f}% below immediate)")

# ---- 6. Save a model trained on ALL data (including the test period) for use in EcoOrchestrator
SAVE_FINAL = True
if SAVE_FINAL:
    all_o = origins
    Xall, yall, _ = build(all_o)
    ok2 = ~np.isnan(yall) & ~np.isnan(Xall["ci_now"].values)
    final = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=50, tree_method="hist", n_jobs=-1)
    final.fit(Xall[ok2], (yall - Xall["ci_now"].values)[ok2])
    final.save_model("carbon_model.json")
    print("\nsaved carbon_model.json (predicts change in gCO2/kWh; add it to the current intensity)")