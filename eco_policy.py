"""
eco_policy.py - simulated device classes, per-round energy / thermal / carbon accounting,
and the EcoOrchestrator client-selection policy. No torch needed; fl_sim.run() calls the policy as `selector`.

ASSUMPTIONS (edit the tables below; state them in your report):
  * The SuperCloud GPU power curve gives the SHAPE of power vs load; each device class rescales it between its own idle and peak watts.
  * The BMCDATA thermal model is used as a generic node thermal model (load fraction -> cpu_usage; load_1 = 1.3 x cpu_usage, as in the data).
  * Telemetry (utilization, ambient temperature) is synthetic. All clients sit on the Great Britain grid.
"""
import numpy as np, pandas as pd

CLASSES = {   # idle/peak watts, training speed (samples per second), fan rpm, ambient offset (deg C)
    "phone":  dict(idle_w=1.0, peak_w=5.0,  speed=60,  fan=0,    amb=8.0),
    "laptop": dict(idle_w=6.0, peak_w=35.0, speed=250, fan=3000, amb=0.0),
    "edge":   dict(idle_w=8.0, peak_w=50.0, speed=600, fan=2000, amb=2.0),
}
NAMES = list(CLASSES)
PROBS = [0.5, 0.3, 0.2]
TELEMETRY = {"phone": (85, 35), "laptop": (70, 25), "edge": (80, 30)}   # mean SM util %, mean memory util % during training


class EcoPolicy:
    def __init__(self, em, epochs, seed=0, start="2026-05-01 00:00", round_interval_h=1.0,
                 use_energy=False, use_timing=False, use_thermal=False,
                 lam=0.5, eps=0.1, max_delay_h=6, temp_limit_c=44.0):
        self.em, self.epochs, self.seed = em, epochs, seed
        self.t0, self.interval = pd.Timestamp(start), pd.Timedelta(hours=round_interval_h)
        self.use_energy, self.use_timing, self.use_thermal = use_energy, use_timing, use_thermal
        self.lam, self.eps, self.max_delay_h, self.limit = lam, eps, max_delay_h, temp_limit_c
        self.p0, self.pref = em.power_w(0, 0), em.power_w(100, 50)
        self.log, self.prev_start, self._ready = [], None, False

    # ---- helpers ----
    def _device_w(self, sm, mem, idle, peak):
        u = np.clip((self.em.power_w(sm, mem) - self.p0) / max(self.pref - self.p0, 1.0), 0, 1.2)
        return idle + (peak - idle) * u

    @staticmethod
    def _ambient(ts):
        h = ts.hour + ts.minute / 60
        return 24 + 5 * np.sin(2 * np.pi * (h - 9) / 24)

    def _init(self, ctx):
        n = ctx["num_clients"]
        cls = np.random.default_rng(1000 + self.seed).choice(len(NAMES), size=n, p=PROBS)   # same for every policy
        g = lambda key: np.array([CLASSES[NAMES[c]][key] for c in cls], float)
        self.cls, self.idle, self.peak, self.fan, self.amb_off = cls, g("idle_w"), g("peak_w"), g("fan"), g("amb")
        self.sm_mean = np.array([TELEMETRY[NAMES[c]][0] for c in cls], float)
        self.mem_mean = np.array([TELEMETRY[NAMES[c]][1] for c in cls], float)
        samples = np.array([len(c["y"]) for c in ctx["clients"]], float)
        self.time_s = samples * self.epochs / g("speed") + 1.0
        self.e_exp_kwh = self._device_w(self.sm_mean, self.mem_mean, self.idle, self.peak) * self.time_s / 3.6e6
        self._ready = True

    def _choose_start(self, ready):
        if not self.use_timing:
            return ready, 0.0
        steps = int(min(48, 2 * self.max_delay_h))
        f = self.em.carbon_forecast(ready, steps=steps)
        times = [ready] + list(f.index)
        vals = [self.em.carbon_actual(ready)] + list(f.values)       # the current value is known, the rest are forecasts
        start = times[int(np.argmin(vals))]
        return start, (start - ready) / pd.Timedelta(hours=1)

    # ---- the selector called by fl_sim.run ----
    def __call__(self, ctx):
        if not self._ready:
            self._init(ctx)
        r, k, n, rng = ctx["round"], ctx["k"], ctx["num_clients"], ctx["rng"]
        ready = self.t0 if self.prev_start is None else self.prev_start + self.interval
        start, wait_h = self._choose_start(ready)
        self.prev_start = start
        amb = self._ambient(start) + self.amb_off

        # what the orchestrator can see: expected telemetry
        temp_exp = self.em.cpu_temp_batch(self.sm_mean, 1.3 * self.sm_mean, amb, self.fan)
        if not (self.use_energy or self.use_thermal):
            chosen = rng.choice(n, size=k, replace=False)
        else:
            loss = ctx["last_loss"]
            utility = np.ones(n)                                       # never-selected clients get the maximum (explore)
            known = ~np.isnan(loss)
            if known.any():
                utility[known] = loss[known] / np.nanmax(loss)
            score = utility - (self.lam * self.e_exp_kwh / self.e_exp_kwh.max() if self.use_energy else 0.0)
            ok = temp_exp <= self.limit if self.use_thermal else np.ones(n, bool)
            cand = np.where(ok)[0]
            if len(cand) < k:                                           # too few cool clients: add the coolest others
                rest = np.where(~ok)[0]
                cand = np.concatenate([cand, rest[np.argsort(temp_exp[rest])][:k - len(cand)]])
            chosen = list(cand[np.argsort(-score[cand])][:k])
            n_rand = int(rng.binomial(k, self.eps))
            if n_rand:
                pool = [i for i in cand if i not in chosen]
                if pool:
                    for j, i in enumerate(rng.choice(pool, size=min(n_rand, len(pool)), replace=False)):
                        chosen[-1 - j] = int(i)
        chosen = np.array([int(i) for i in chosen])

        # what actually happens: realised telemetry, identical for every policy (same seed + round)
        rr = np.random.default_rng(10_000 + 7919 * self.seed + r)
        sm = np.clip(rr.normal(self.sm_mean, 8), 30, 100)
        mem = np.clip(rr.normal(self.mem_mean, 6), 5, 80)
        amb_real = amb + rr.normal(0, 1.0, n)
        watts = self._device_w(sm, mem, self.idle, self.peak)
        e_kwh = watts[chosen] * self.time_s[chosen] / 3.6e6
        temp_real = self.em.cpu_temp_batch(sm[chosen], 1.3 * sm[chosen], amb_real[chosen], self.fan[chosen])
        ci = self.em.carbon_actual(start)
        self.log.append({"round": r, "start": start, "wait_h": wait_h, "ci_gco2_kwh": ci,
                         "energy_wh": float(e_kwh.sum() * 1000), "carbon_g": float(e_kwh.sum() * ci),
                         "violations": int((temp_real > self.limit).sum()), "max_temp_c": float(temp_real.max()),
                         "round_s": float(self.time_s[chosen].max())})
        return chosen

    def log_df(self):
        return pd.DataFrame(self.log)


def summarize(acc, pol, target=0.70):
    """One row of results for a run: accuracy, energy, carbon, thermal violations, cost to reach `target` accuracy."""
    df = acc[["round", "test_acc"]].merge(pol, on="round")
    df["cum_wh"], df["cum_g"] = df["energy_wh"].cumsum(), df["carbon_g"].cumsum()
    df["acc_smooth"] = df["test_acc"].rolling(3, min_periods=1).mean()      # smoothed, so one lucky round does not count
    hit = df[df["acc_smooth"] >= target]
    first = hit.iloc[0] if len(hit) else None
    return {"final_acc": float(df["test_acc"].tail(10).mean()), "total_wh": float(df["energy_wh"].sum()),
            "total_gco2": float(df["carbon_g"].sum()), "violations": int(df["violations"].sum()),
            "wall_hours": float((df["start"].iloc[-1] - df["start"].iloc[0]) / pd.Timedelta(hours=1)),
            "rounds_to_target": float(first["round"]) if first is not None else np.nan,
            "wh_to_target": float(first["cum_wh"]) if first is not None else np.nan,
            "gco2_to_target": float(first["cum_g"]) if first is not None else np.nan}