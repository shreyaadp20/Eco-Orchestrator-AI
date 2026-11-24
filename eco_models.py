"""
eco_models.py - loads the three trained models and answers three questions:
  energy : how many watts / kWh does a GPU use at a given load?
  thermal: what CPU temperature does a node settle at for a given load?
  carbon : what will grid carbon intensity (gCO2/kWh) be over the next 24 hours?

Put this file in the same folder as: power_B_sm_plus_mem.json, thermal_cpu1_load.json, thermal_cpu1_psu.json,
thermal_cpu2_load.json, thermal_cpu2_psu.json, carbon_model.json and df_fuel_ckan.csv.
"""
import os
import numpy as np, pandas as pd
from xgboost import XGBRegressor

CARBON_COLS = ["h", "ci_now", "ci_1h", "ci_2h", "ci_6h", "ci_24h", "ci_mean_24h", "wind", "solar", "gas",
               "imports", "nuclear", "ci_yday_same_time", "ci_lastweek_same_time", "tgt_hour", "tgt_dow", "tgt_month"]
THERMAL_LOAD = ["cpu_usage", "load_1", "Inlet_Temp", "fan_mean"]
THERMAL_PSU = THERMAL_LOAD + ["PSU1_Total_Power", "PSU2_Total_Power"]


class EcoModels:
    def __init__(self, folder=".", neso_csv="df_fuel_ckan.csv", grid_from="2023-01-01", carbon_file="carbon_model.json"):
        def load(name):
            m = XGBRegressor()
            m.load_model(os.path.join(folder, name))
            return m
        self.power = load("power_B_sm_plus_mem.json")
        self.thermal = {(cpu, kind): load(f"thermal_{cpu}_{kind}.json")
                        for cpu in ("cpu1", "cpu2") for kind in ("load", "psu")}
        self.carbon = load(carbon_file)
        self._carbon_cols = self.carbon.get_booster().feature_names or CARBON_COLS
        self._load_grid(os.path.join(folder, neso_csv), grid_from)

    # ---------------- energy ----------------
    def power_w(self, sm_util_pct, mem_util_pct):
        """GPU power in watts (V100-class data-centre GPU) from SM and memory utilization, both 0-100."""
        X = pd.DataFrame({"smutilization_pct_avg": np.atleast_1d(sm_util_pct).astype(float),
                          "memoryutilization_pct_avg": np.atleast_1d(mem_util_pct).astype(float)})
        w = np.clip(self.power.predict(X), 0, None)
        return float(w[0]) if np.ndim(sm_util_pct) == 0 and np.ndim(mem_util_pct) == 0 else w

    def energy_kwh(self, sm_util_pct, mem_util_pct, seconds, n_gpus=1):
        return self.power_w(sm_util_pct, mem_util_pct) * seconds * n_gpus / 3.6e6

    # ---------------- thermal ----------------
    def cpu_temps(self, cpu_usage_pct, load_1, inlet_c, fan_rpm, psu1_w=None, psu2_w=None):
        """CPU1 and CPU2 temperatures (deg C). Uses the PSU-power models when both PSU powers are given."""
        row = {"cpu_usage": cpu_usage_pct, "load_1": load_1, "Inlet_Temp": inlet_c, "fan_mean": fan_rpm}
        kind, cols = "load", THERMAL_LOAD
        if psu1_w is not None and psu2_w is not None:
            row.update({"PSU1_Total_Power": psu1_w, "PSU2_Total_Power": psu2_w})
            kind, cols = "psu", THERMAL_PSU
        X = pd.DataFrame([row])[cols]
        return (float(self.thermal[("cpu1", kind)].predict(X)[0]),
                float(self.thermal[("cpu2", kind)].predict(X)[0]))


    def cpu_temp_batch(self, cpu_usage_pct, load_1, inlet_c, fan_rpm):
        """Vectorised: hotter of CPU1/CPU2 (deg C) for arrays of inputs, using the models without PSU power."""
        X = pd.DataFrame({"cpu_usage": np.asarray(cpu_usage_pct, float), "load_1": np.asarray(load_1, float),
                          "Inlet_Temp": np.asarray(inlet_c, float), "fan_mean": np.asarray(fan_rpm, float)})[THERMAL_LOAD]
        return np.maximum(self.thermal[("cpu1", "load")].predict(X), self.thermal[("cpu2", "load")].predict(X))

    # ---------------- carbon ----------------
    def _load_grid(self, path, grid_from):
        cols = ["DATETIME", "CARBON_INTENSITY", "GAS_perc", "NUCLEAR_perc", "WIND_perc",
                "WIND_EMB_perc", "SOLAR_perc", "IMPORTS_perc"]
        raw = pd.read_csv(path, usecols=cols)
        raw["DATETIME"] = pd.to_datetime(raw["DATETIME"])
        g = raw.drop_duplicates("DATETIME").sort_values("DATETIME").set_index("DATETIME").asfreq("30min")
        g = g[g.index >= grid_from].interpolate(limit=4)
        self.grid = g
        self.ci = g["CARBON_INTENSITY"].values
        self.roll48 = g["CARBON_INTENSITY"].rolling(48).mean().values
        self.wind = (g["WIND_perc"] + g["WIND_EMB_perc"]).values
        self.solar, self.gas = g["SOLAR_perc"].values, g["GAS_perc"].values
        self.imports, self.nuclear = g["IMPORTS_perc"].values, g["NUCLEAR_perc"].values

    def _pos(self, ts):
        return self.grid.index.get_loc(pd.Timestamp(ts).floor("30min"))

    def carbon_actual(self, ts):
        """Recorded carbon intensity (gCO2/kWh) at a timestamp - the simulation's ground truth."""
        return float(self.ci[self._pos(ts)])

    def carbon_forecast(self, now, steps=48, persistence_steps=2):
        """Forecast for the next `steps` half-hours (max 48), using only data up to `now`.
        The first `persistence_steps` half-hours use 'no change', which beat the model at short range."""
        if steps > 48:
            raise ValueError("steps must be <= 48 (the model was trained for up to 24 hours ahead)")
        o = self._pos(now)
        if o < 336:
            raise ValueError("need at least 7 days of grid history before `now`")
        hs = np.arange(1, steps + 1)
        t = o + hs
        times = self.grid.index[o] + pd.to_timedelta(hs * 30, unit="min")
        X = pd.DataFrame({
            "h": hs, "ci_now": self.ci[o], "ci_1h": self.ci[o - 2], "ci_2h": self.ci[o - 4],
            "ci_6h": self.ci[o - 12], "ci_24h": self.ci[o - 48], "ci_mean_24h": self.roll48[o],
            "wind": self.wind[o], "solar": self.solar[o], "gas": self.gas[o],
            "imports": self.imports[o], "nuclear": self.nuclear[o],
            "ci_yday_same_time": self.ci[t - 48], "ci_lastweek_same_time": self.ci[t - 336],
            "tgt_hour": np.asarray(times.hour + times.minute / 60), "tgt_dow": np.asarray(times.dayofweek),
            "tgt_month": np.asarray(times.month)})
        pred = self.ci[o] + self.carbon.predict(X[self._carbon_cols])
        pred[:persistence_steps] = self.ci[o]
        return pd.Series(np.clip(pred, 0, None), index=times, name="forecast_gCO2_per_kWh")

    def best_start(self, now, steps=48):
        """Cleanest half-hour to start within the next `steps` half-hours -> (timestamp, forecast gCO2/kWh)."""
        f = self.carbon_forecast(now, steps)
        return f.idxmin(), float(f.min())


if __name__ == "__main__":
    em = EcoModels(".")
    print("GPU power (W) at SM/mem 0/0, 40/10, 90/40:",
          [round(em.power_w(s, m), 1) for s, m in [(0, 0), (40, 10), (90, 40)]])
    print("energy for 10 min at SM 90 / mem 40:", round(em.energy_kwh(90, 40, 600), 4), "kWh")
    print("CPU temps, light load  (cpu 10%, load 5,  inlet 25, fans 2000 rpm):",
          [round(x, 1) for x in em.cpu_temps(10, 5, 25, 2000)])
    print("CPU temps, heavy load  (cpu 95%, load 100, inlet 30, fans 2000 rpm):",
          [round(x, 1) for x in em.cpu_temps(95, 100, 30, 2000)])
    print("same, more fans (8000 rpm):", [round(x, 1) for x in em.cpu_temps(95, 100, 30, 8000)])
    now = em.grid.index[-48 * 3]                       # three days before the end, so the future is known
    f = em.carbon_forecast(now)
    o = em._pos(now)
    actual = pd.Series(em.ci[o + 1:o + 49], index=f.index)
    when, val = em.best_start(now)
    print(f"\nnow = {now}  | forecast MAE over next 24 h: {(f - actual).abs().mean():.1f} gCO2/kWh")
    print(f"model picks {when} (forecast {val:.0f}); actual there {actual[when]:.0f} | "
          f"run now {actual.iloc[0]:.0f} | best possible {actual.min():.0f}")