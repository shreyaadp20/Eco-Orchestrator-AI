"""
dc_twin.py - digital twin of a GPU data center, driven by a real job trace (SuperCloud).

Each simulated minute the twin tracks: which jobs run where, GPU power (SuperCloud power model), host power (BMCDATA model),
CPU temperature (BMCDATA thermal model), facility power (PUE) and carbon (NESO grid intensity and forecast).
Scheduling policies are plug-in flags, so the same workload can be replayed under each policy and compared.

ASSUMPTIONS (the CFG table; state them in your report):
  * Nodes have 2 GPUs and 40 cores. The BMCDATA server stands in for the GPU nodes' host (CPU, fans, power supplies).
  * Jobs run at their average measured GPU utilisation for their whole runtime (no within-job variation).
  * Node CPU usage = cpu_activity x (CPUs requested / cores). Fans run at a fixed fan_rpm (set fan_rpm=0 to make them follow CPU usage).
    Inlet air = supply air + daily swing + fixed rack offset. In BMCDATA the CPU temperature is driven mostly by inlet air, only weakly by load.
  * Facility power = IT power x PUE. Powered-off nodes draw off_w. Waking a node takes wake_min minutes.
  * Trace timestamps are offsets, not dates; the trace is replayed from a start date you choose (all on the Great Britain grid).
"""
import re
import numpy as np, pandas as pd
from xgboost import XGBRegressor

CFG = dict(gpus_per_node=2, cores_per_node=40, pue=1.4, off_w=15.0, wake_min=3, idle_off_min=10,
           supply_c=26.0, supply_swing_c=3.0, fan_rpm=3000, rack_offset_max_c=6.0, nodes_per_rack=8, temp_limit_c=40.0,
           cpu_activity=0.5, max_defer_h=6.0, backfill_depth=100, green_window_h=12, thermal_every_min=5)

POLICIES = {
    "baseline": dict(place="first_fit", defer=False, poweroff=False),   # first come first served, always-on nodes
    "thermal":  dict(place="cool",      defer=False, poweroff=False),   # place jobs on the coolest nodes
    "carbon":   dict(place="first_fit", defer=True,  poweroff=False),   # hold deferrable jobs for cleaner hours
    "poweroff": dict(place="pack",      defer=False, poweroff=True),    # pack jobs, switch idle nodes off
    "eco":      dict(place="pack",      defer=True,  poweroff=True),    # all of the above except cool placement
}


# ---------------------------------------------------------------- workload
def load_trace(sched_csv, dcgm_csv, days=7, offset_days=100, seed=0, max_hours=72, warmup_days=1, defer_mode="all_but_interactive"):
    """Real SuperCloud GPU jobs submitted in a window of `days` days (plus `warmup_days` before it, so the cluster starts busy).
    Utilisation comes from DCGM where available. defer_mode: 'all_but_interactive' or 'batch_only' (only LLSUB:BATCH and LLMAPREDUCE:MAP)."""
    rng = np.random.default_rng(seed)
    cols = ["id_job", "id_user", "cpus_req", "gres_req", "time_submit", "time_start", "time_end", "job_type"]
    s = pd.read_csv(sched_csv, usecols=cols, dtype={"id_job": "int64"})
    dup = s.loc[s.duplicated("id_job", keep=False), "id_job"].unique()
    s = s[~s["id_job"].isin(dup)].copy()
    for c in ("time_submit", "time_start", "time_end"):
        s[c] = s[c].where((s[c] >= 0) & (s[c] < 4e7))                  # -1 and huge values are sentinels
    s["gpus"] = pd.to_numeric(s["gres_req"].astype(str).str.extract(r"(\d+)\s*$")[0], errors="coerce")
    s = s[s["gres_req"].astype(str).str.contains("gpu") & (s["gpus"] >= 1)]
    s["dur_s"] = s["time_end"] - s["time_start"]
    s = s.dropna(subset=["time_submit", "dur_s"])
    s = s[(s["dur_s"] >= 30) & (s["dur_s"] <= max_hours * 3600)]
    t0 = s["time_submit"].min() + offset_days * 86400
    w = s[(s["time_submit"] >= t0 - warmup_days * 86400) & (s["time_submit"] < t0 + days * 86400)].copy()
    w["submit_s"] = w["time_submit"] - t0

    d = pd.read_csv(dcgm_csv, usecols=["id_job", "smutilization_pct_avg", "memoryutilization_pct_avg", "powerusage_watts_avg"],
                    dtype={"id_job": "int64"})
    d = d.groupby("id_job").agg(sm=("smutilization_pct_avg", "mean"), mem=("memoryutilization_pct_avg", "mean"),
                                meas_w=("powerusage_watts_avg", "mean")).reset_index()
    w = w.merge(d, on="id_job", how="left")
    miss = w["sm"].isna().to_numpy()
    if miss.any():                                                      # jobs without DCGM data: borrow a random measured profile
        pool = d[["sm", "mem"]].to_numpy()
        pick = rng.integers(0, len(pool), int(miss.sum()))
        w.loc[miss, "sm"], w.loc[miss, "mem"] = pool[pick, 0], pool[pick, 1]
    if defer_mode == "batch_only":
        w["deferrable"] = w["job_type"].isin(["LLSUB:BATCH", "LLMAPREDUCE:MAP"])
    else:
        w["deferrable"] = w["job_type"] != "LLSUB:INTERACTIVE"          # interactive jobs cannot wait
    w["cpus"] = w["cpus_req"].fillna(1).clip(lower=1)
    cols_out = ["id_job", "id_user", "submit_s", "dur_s", "gpus", "cpus", "sm", "mem", "meas_w", "deferrable"]
    return w.sort_values("submit_s").reset_index(drop=True)[cols_out]


def suggest_nodes(jobs, days, target_util=0.5, gpus_per_node=2):
    """Cluster size that would keep GPUs about `target_util` busy on average for this workload."""
    jobs = jobs[jobs["submit_s"] >= 0]
    gpu_hours = float((jobs["gpus"] * jobs["dur_s"]).sum() / 3600)
    need = gpu_hours / (days * 24 * gpus_per_node * target_util)
    return int(max(np.ceil(need), np.ceil(jobs["gpus"].quantile(0.99) / gpus_per_node)))


def check_job_energy(jobs, em):
    """Pipeline check: the twin's GPU power estimate vs the measured average power, for jobs that have DCGM data."""
    j = jobs.dropna(subset=["meas_w"])
    pred = np.atleast_1d(em.power_w(j["sm"].to_numpy(), j["mem"].to_numpy()))
    err = np.abs(pred - j["meas_w"].to_numpy()) / np.maximum(j["meas_w"].to_numpy(), 1)
    w = (j["gpus"] * j["dur_s"]).to_numpy()
    return {"jobs_checked": len(j), "median_abs_pct_err": float(np.median(err) * 100),
            "energy_ratio_pred_over_measured": float((pred * w).sum() / (j["meas_w"].to_numpy() * w).sum())}


# ---------------------------------------------------------------- the twin
class DataCenterTwin:
    def __init__(self, em, host_model="host_power.json", cfg=None):
        self.em = em
        self.cfg = {**CFG, **(cfg or {})}
        self.host = XGBRegressor()
        self.host.load_model(host_model)

    def _host_w(self, cpu_use, fan):
        X = pd.DataFrame({"cpu_usage": cpu_use, "fan_mean": fan})
        return np.clip(self.host.predict(X), 0, None)

    def run(self, jobs, n_nodes, policy, start="2026-05-04 00:00", horizon_days=7, warmup_days=1, seed=0, site=None, detail=False):
        site = site or {}
        c, em = {**self.cfg, **site.get("cfg", {})}, self.em      # per-site overrides: pue, supply_c, gpu_scale, ...
        ci_act = site.get("ci_actual") or em.carbon_actual
        ci_fc = site.get("ci_forecast") or (lambda ts, steps: em.carbon_forecast(ts, steps=steps).to_numpy())
        pol = POLICIES[policy] if isinstance(policy, str) else policy
        g, N = c["gpus_per_node"], int(n_nodes)
        S = N * g
        jobs = jobs[jobs["gpus"] <= S].reset_index(drop=True)
        J = len(jobs)
        sub, dur = jobs["submit_s"].to_numpy(), jobs["dur_s"].to_numpy()
        gp, cpus = jobs["gpus"].to_numpy().astype(int), jobs["cpus"].to_numpy()
        jsm, jmem, defer_ok = jobs["sm"].to_numpy(), jobs["mem"].to_numpy(), jobs["deferrable"].to_numpy()
        t_start = pd.Timestamp(start)
        steps = int(horizon_days * 24 * 60)
        warm = int(warmup_days * 24 * 60)
        wake_s, off_after_s = c["wake_min"] * 60, c["idle_off_min"] * 60
        max_defer_s = c["max_defer_h"] * 3600

        rng = np.random.default_rng(seed)
        rack = np.arange(N) // c["nodes_per_rack"]
        rack_off = rng.uniform(0, c["rack_offset_max_c"], rack.max() + 1)[rack]   # same for every policy (same seed)

        on = np.ones(N, bool)
        ready_at = np.zeros(N)
        idle_since = np.zeros(N)
        busy = np.zeros(S, bool)
        slot_sm, slot_mem, slot_cpu = np.zeros(S), np.zeros(S), np.zeros(S)
        temp = np.full(N, c["supply_c"])
        start_at, end_at = np.full(J, np.nan), np.full(J, np.nan)
        started, done = np.zeros(J, bool), np.zeros(J, bool)
        slots_of = [None] * J
        queue, pending, running = [], [], []
        nxt, dirty, green = 0, True, True
        ci, node_w, gpu_w_node = 0.0, np.zeros(N), np.zeros(N)
        cpu_use, fan = np.zeros(N), np.full(N, float(c["fan_rpm"]) if c["fan_rpm"] > 0 else 1500.0)

        kwh_fac = kwh_it = carbon = 0.0
        viol_node_min, peak_kw = 0.0, 0.0
        ts_rows, acc_kw, acc_n = [], 0.0, 0
        snaps = {"time": [], "power_w": [], "temp_c": [], "gpus_busy": [], "on": []}

        def choose(gneed):
            free = g - busy.reshape(N, g).sum(1)
            cand = np.where(free > 0)[0]
            if free[cand].sum() < gneed:
                return None
            if pol["place"] == "first_fit":
                key = cand.astype(float)
            elif pol["place"] == "cool":
                key = temp[cand] + cand * 1e-6
            else:                                                       # pack: busy-but-not-full nodes, then idle on, then off
                rank = np.where(~on[cand], 2, np.where(free[cand] < g, 0, 1)).astype(float)
                hot = (temp[cand] > c["temp_limit_c"] - 1.0).astype(float)
                key = rank * 1e6 + hot * 1e5 + cand
            order = cand[np.argsort(key, kind="stable")]
            k = int(np.searchsorted(np.cumsum(free[order]), gneed))
            out, rem = [], gneed
            for n_ in order[:k + 1]:
                take = int(min(free[n_], rem))
                out += [n_ * g + i for i in range(g) if not busy[n_ * g + i]][:take]
                rem -= take
            return out

        def begin(j):
            started[j] = True
            for s_ in slots_of[j]:
                slot_sm[s_], slot_mem[s_], slot_cpu[s_] = jsm[j], jmem[j], cpus[j] / gp[j]
            running.append(j)

        for t in range(-warm, steps):                                   # t < 0 is warm-up: simulated, not measured
            now = t * 60.0
            ts = t_start + pd.Timedelta(seconds=now)
            while nxt < J and sub[nxt] <= now:
                queue.append(nxt); nxt += 1
            for j in [j for j in pending if start_at[j] <= now]:        # jobs whose nodes have finished waking
                pending.remove(j); begin(j); dirty = True
            for j in [j for j in running if end_at[j] <= now]:          # completions
                running.remove(j); done[j] = True
                for s_ in slots_of[j]:
                    busy[s_] = False; slot_sm[s_] = slot_mem[s_] = slot_cpu[s_] = 0.0
                    if not busy[(s_ // g) * g:(s_ // g) * g + g].any():
                        idle_since[s_ // g] = now
                dirty = True
            if t % 30 == 0:
                ci = ci_act(ts)
                if pol["defer"]:
                    f = np.asarray(ci_fc(ts, int(c["green_window_h"] * 2)))
                    green = bool(ci <= f.min() * 1.02)

            # scheduling: first come first served with backfill
            for j in list(queue[:c["backfill_depth"]]):
                if pol["defer"] and defer_ok[j] and not green and (now - sub[j]) < max_defer_s:
                    continue
                slots = choose(gp[j])
                if slots is None:
                    continue
                queue.remove(j)
                nodes = sorted({s_ // g for s_ in slots})
                st = now
                for n_ in nodes:
                    if not on[n_]:
                        on[n_] = True; ready_at[n_] = now + wake_s
                    st = max(st, ready_at[n_])
                for s_ in slots:
                    busy[s_] = True
                slots_of[j], start_at[j], end_at[j] = slots, st, st + dur[j]
                if st <= now:
                    begin(j)
                else:
                    pending.append(j)
                dirty = True

            if pol["poweroff"]:
                nb = busy.reshape(N, g).sum(1)
                off_now = on & (nb == 0) & (now - idle_since >= off_after_s) & (ready_at <= now)
                if off_now.any():
                    on[off_now] = False; dirty = True

            # power and thermal state
            if dirty:
                cpu_use = np.clip(100 * c["cpu_activity"] * slot_cpu.reshape(N, g).sum(1) / c["cores_per_node"], 0, 100)
                fan = np.full(N, float(c["fan_rpm"])) if c["fan_rpm"] > 0 else np.clip(1500 + 50 * cpu_use, 0, 9000)
                gpu_w_node = np.asarray(em.power_w(slot_sm, slot_mem)).reshape(N, g).sum(1) * c.get("gpu_scale", 1.0)
                node_w = np.where(on, self._host_w(cpu_use, fan) + gpu_w_node, c["off_w"])
                dirty = False
            if t % c["thermal_every_min"] == 0:
                hour = ts.hour + ts.minute / 60
                inlet = c["supply_c"] + c["supply_swing_c"] * np.sin(2 * np.pi * (hour - 9) / 24) + rack_off
                temp = np.where(on, em.cpu_temp_batch(cpu_use, 1.3 * cpu_use, inlet, fan), inlet)
            if t < 0:
                continue
            viol_node_min += float((on & (temp > c["temp_limit_c"])).sum())

            it_kw = node_w.sum() / 1000
            fac_kw = it_kw * c["pue"]
            kwh_it += it_kw / 60; kwh_fac += fac_kw / 60; carbon += fac_kw / 60 * ci
            peak_kw = max(peak_kw, fac_kw)
            acc_kw += fac_kw; acc_n += 1
            if detail and t % 10 == 0:
                snaps["time"].append(ts); snaps["power_w"].append(node_w.copy()); snaps["temp_c"].append(temp.copy())
                snaps["gpus_busy"].append(busy.reshape(N, g).sum(1)); snaps["on"].append(on.copy())
            if (t + 1) % 30 == 0:
                ts_rows.append({"time": ts, "facility_kw": acc_kw / acc_n, "ci_gco2_kwh": ci,
                                "nodes_on": int(on.sum()), "busy_gpus": int(busy.sum()), "queued": len(queue),
                                "max_temp_c": float(temp[on].max()) if on.any() else np.nan})
                acc_kw, acc_n = 0.0, 0

        H = steps * 60.0
        inwin = sub >= 0                                                # only jobs submitted in the measured window
        st_ok = started & ~np.isnan(start_at)
        sel = st_ok & inwin
        wait = (start_at[sel] - sub[sel]) / 60
        bsld = np.maximum((start_at[sel] - sub[sel] + dur[sel]) / np.maximum(dur[sel], 600), 1)
        overlap = np.clip(np.minimum(end_at[st_ok], H) - np.maximum(start_at[st_ok], 0.0), 0, None)
        delivered = float((gp[st_ok] * overlap).sum() / 3600)           # GPU-hours actually delivered inside the window
        metrics = {"policy": policy if isinstance(policy, str) else "custom", "nodes": N, "jobs": int(inwin.sum()),
                   "jobs_done": int((done & inwin).sum()), "jobs_unfinished": int((~done & inwin).sum()),
                   "facility_kwh": kwh_fac, "it_kwh": kwh_it, "carbon_kg": carbon / 1000,
                   "avg_ci_gco2_kwh": carbon / max(kwh_fac, 1e-9), "peak_kw": peak_kw,
                   "gpu_hours": delivered, "kwh_per_gpu_hour": kwh_fac / max(delivered, 1e-9),
                   "thermal_node_hours_over_limit": viol_node_min / 60,
                   "mean_wait_min": float(wait.mean()) if len(wait) else np.nan,
                   "p95_wait_min": float(np.percentile(wait, 95)) if len(wait) else np.nan,
                   "mean_bounded_slowdown": float(bsld.mean()) if len(bsld) else np.nan}
        if not detail:
            return metrics, pd.DataFrame(ts_rows)
        jt = jobs[["id_job", "submit_s", "dur_s", "gpus"]].copy()
        jt["start_s"], jt["end_s"] = start_at, end_at
        jt["wait_min"] = (start_at - sub) / 60
        jt["in_window"] = inwin
        jt["done"] = done
        jt["nodes"] = ["|".join(str(n_) for n_ in sorted({s_ // g for s_ in sl})) if sl is not None else "" for sl in slots_of]
        snap = {k: (np.array(v) if k != "time" else list(v)) for k, v in snaps.items()}
        return metrics, pd.DataFrame(ts_rows), {"jobs": jt, "snapshots": snap, "n_nodes": N, "rack_offset_c": rack_off}