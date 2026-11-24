"""
eco_router.py - decides which site (group of servers) each incoming workload is sent to.

For every workload and every site it estimates energy (GPU power x hardware efficiency + host share, x cooling overhead PUE),
carbon (energy x the site's forecast grid intensity while the job would run) and CPU temperature, then sends the workload to the
site with the best combined score, skipping sites that are nearly full. Every decision is logged with its per-site numbers.

ASSUMPTIONS (synthetic stand-ins for three regions; replace with real data if you have it):
  * Each site's grid is the NESO series scaled by ci_scale and delayed by lag_h hours (so its forecast can be built from the real forecast).
  * Sites differ in cooling overhead (pue), supply air temperature, and GPU efficiency (gpu_scale, lower = more efficient).
"""
import heapq
import numpy as np, pandas as pd

SITES = {
    "North":   dict(share=0.4, pue=1.15, supply_c=22.0, gpu_scale=0.85, ci_scale=0.55, lag_h=6.0),   # cold climate, newer GPUs, cleaner grid
    "Central": dict(share=0.4, pue=1.40, supply_c=26.0, gpu_scale=1.00, ci_scale=1.00, lag_h=0.0),   # the NESO grid as it is
    "South":   dict(share=0.2, pue=1.60, supply_c=29.0, gpu_scale=1.15, ci_scale=1.30, lag_h=3.0),   # hot climate, older GPUs, dirtier grid
}
HOST_MARGINAL_W = 60.0        # extra host power per node that a job wakes up (assumption)


def split_nodes(total, sites=SITES):
    n = {k: max(1, int(round(total * v["share"]))) for k, v in sites.items()}
    return n


def make_site(em, name, spec):
    """Settings and carbon functions for one site, in the form DataCenterTwin.run(site=...) expects."""
    lag, scale, cache = pd.Timedelta(hours=spec["lag_h"]), spec["ci_scale"], {}

    def actual(ts):
        return scale * em.carbon_actual(pd.Timestamp(ts) - lag)

    def forecast(ts, steps=48):
        ts0 = pd.Timestamp(ts).floor("30min")
        if ts0 not in cache:
            base = em.carbon_forecast(ts0, steps=48).to_numpy()
            out = np.empty(48)
            for k in range(1, 49):
                src = ts0 + pd.Timedelta(minutes=30 * k) - lag
                out[k - 1] = em.carbon_actual(src) if src <= ts0 else base[int((src - ts0) / pd.Timedelta(minutes=30)) - 1]
            cache[ts0] = scale * out
            if len(cache) > 400:
                cache.pop(next(iter(cache)))
        return cache[ts0][:steps]

    return {"name": name, "cfg": {"pue": spec["pue"], "supply_c": spec["supply_c"], "gpu_scale": spec["gpu_scale"]},
            "ci_actual": actual, "ci_forecast": forecast}


class EcoRouter:
    def __init__(self, em, sites=SITES, w_carbon=0.6, w_energy=0.4, util_cap=0.85, temp_limit_c=40.0, w_load=0.0):
        self.em, self.sites = em, sites
        self.w_c, self.w_e, self.cap, self.limit, self.w_load = w_carbon, w_energy, util_cap, temp_limit_c, w_load
        self.hooks = {k: make_site(em, k, v) for k, v in sites.items()}
        self._temp_cache = {}

    def _temp(self, name, ts, cpu_use):
        key = (name, ts.floor("h"), int(cpu_use // 10))
        if key not in self._temp_cache:
            sp = self.sites[name]
            hour = ts.hour + ts.minute / 60
            inlet = sp["supply_c"] + 3.0 * np.sin(2 * np.pi * (hour - 9) / 24) + 3.0      # +3 C = average rack offset
            self._temp_cache[key] = float(self.em.cpu_temp_batch([cpu_use], [1.3 * cpu_use], [inlet], [3000.0])[0])
        return self._temp_cache[key]

    def route(self, jobs, site_nodes, start, mode="eco", seed=0):
        """Returns (site name for every job, decision log). mode: 'eco' (scored) or 'random' (proportional to site size)."""
        names = list(self.sites)
        cap_gpus = {n: 2 * site_nodes[n] for n in names}
        active, occ = {n: [] for n in names}, {n: 0 for n in names}
        rng = np.random.default_rng(seed)
        t0 = pd.Timestamp(start)
        gw = np.atleast_1d(self.em.power_w(jobs["sm"].to_numpy(), jobs["mem"].to_numpy()))
        chosen, log = [], []
        for i, r in enumerate(jobs.itertuples()):
            now = float(r.submit_s)
            for n in names:                                              # jobs assumed to start on submission, to estimate load
                while active[n] and active[n][0][0] <= now:
                    occ[n] -= heapq.heappop(active[n])[1]
            g = int(r.gpus)
            fits = [n for n in names if cap_gpus[n] >= g]
            elig = [n for n in fits if occ[n] + g <= self.cap * cap_gpus[n]] or \
                   ([min(fits, key=lambda n: occ[n] / cap_gpus[n])] if fits else [names[1]])
            ts = t0 + pd.Timedelta(seconds=now)
            row = {"idx": i, "id_job": r.id_job, "gpus": g, "dur_h": r.dur_s / 3600}
            if mode == "random":
                w = np.array([self.sites[n]["share"] for n in elig], float)
                pick = elig[int(rng.choice(len(elig), p=w / w.sum()))]
            else:
                steps = int(min(48, max(1, np.ceil(r.dur_s / 1800))))
                cpu_use = float(np.clip(100 * 0.5 * r.cpus / 40, 0, 100))
                en, ca, pen = {}, {}, {}
                for n in elig:
                    sp = self.sites[n]
                    en[n] = (gw[i] * sp["gpu_scale"] * g + HOST_MARGINAL_W * int(np.ceil(g / 2))) * r.dur_s / 3.6e6 * sp["pue"]
                    ci = float(np.mean(self.hooks[n]["ci_forecast"](ts, steps)))
                    ca[n] = en[n] * ci / 1000                                    # kg CO2
                    pen[n] = 1.0 if self._temp(n, ts, cpu_use) > self.limit else 0.0
                me, mc = np.mean(list(en.values())), np.mean(list(ca.values()))
                score = {n: self.w_c * ca[n] / max(mc, 1e-12) + self.w_e * en[n] / max(me, 1e-12) + pen[n]
                         + self.w_load * (occ[n] + g) / cap_gpus[n] for n in elig}      # w_load > 0 spreads work to keep queues short
                pick = min(score, key=score.get)
                for n in elig:
                    row[f"kwh_{n}"], row[f"kg_{n}"], row[f"score_{n}"] = en[n], ca[n], score[n]
            for n in names:
                row[f"occ_{n}"] = occ[n] / cap_gpus[n]
            row["site"] = pick
            occ[pick] += g
            heapq.heappush(active[pick], (now + r.dur_s, g))
            chosen.append(pick); log.append(row)
        return np.array(chosen), pd.DataFrame(log)