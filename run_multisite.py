"""
run_multisite.py - compares routing workloads across three sites against the single-site data center.

Quick check:  python run_multisite.py --days 2 --offset-days 356
Needs everything run_twin.py needs, plus eco_router.py.
"""
import argparse
import numpy as np, pandas as pd
from eco_models import EcoModels
from dc_twin import DataCenterTwin, load_trace, suggest_nodes, check_job_energy
from eco_router import EcoRouter, SITES, split_nodes, make_site

p = argparse.ArgumentParser()
p.add_argument("--sched", default="scheduler_data.csv")
p.add_argument("--dcgm", default="dcgm.csv")
p.add_argument("--days", type=float, default=7)
p.add_argument("--offset-days", type=float, default=100)
p.add_argument("--nodes", type=int, default=0, help="total nodes across all sites; 0 = size for about 50%% GPU utilisation")
p.add_argument("--start", default="2026-05-04 00:00")
p.add_argument("--warmup-days", type=float, default=1)
p.add_argument("--defer-mode", default="all_but_interactive", choices=["all_but_interactive", "batch_only"])
p.add_argument("--w-carbon", type=float, default=0.6)
p.add_argument("--w-energy", type=float, default=0.4)
p.add_argument("--util-cap", type=float, default=0.85, help="router skips a site whose estimated GPU load would exceed this share")
p.add_argument("--w-load", type=float, default=0.0, help="router penalty for loaded sites (0 = off); try 0.5 to shorten queues")
p.add_argument("--scenarios", nargs="+", default=None, help="subset of scenarios to run (default: all five)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

em = EcoModels(".", carbon_file="carbon_model_holdout.json")
jobs = load_trace(args.sched, args.dcgm, days=args.days, offset_days=args.offset_days, seed=args.seed,
                  warmup_days=args.warmup_days, defer_mode=args.defer_mode)
total = args.nodes or suggest_nodes(jobs, args.days)
site_nodes = split_nodes(total)
win = jobs[jobs["submit_s"] >= 0]
print(f"trace: {len(win)} GPU jobs in window | {(win['gpus'] * win['dur_s']).sum() / 3600:.0f} GPU-hours | total nodes {total} -> {site_nodes}")
print("pipeline check:", {k: round(v, 3) for k, v in check_job_energy(jobs, em).items()})

twin = DataCenterTwin(em)
router = EcoRouter(em, SITES, w_carbon=args.w_carbon, w_energy=args.w_energy, util_cap=args.util_cap, w_load=args.w_load)
kw = dict(start=args.start, horizon_days=args.days, warmup_days=args.warmup_days, seed=args.seed, detail=True)


def run_single(within):
    m, ts, d = twin.run(jobs, total, within, **kw)
    return [(m, d)], None


def run_multi(mode, within):
    assign, log = router.route(jobs, site_nodes, args.start, mode=mode, seed=args.seed)
    out = []
    for k, name in enumerate(SITES):
        sub = jobs[assign == name].reset_index(drop=True)
        m, ts, d = twin.run(sub, site_nodes[name], within, site=router.hooks[name], **{**kw, "seed": args.seed + k})
        m["policy"] = name
        d["jobs"]["site"] = name
        out.append((m, d))
    return out, (log, assign)


SCENARIOS = {
    "single_site_baseline": lambda: run_single("baseline"),
    "single_site_eco":      lambda: run_single("eco"),
    "three_sites_random":   lambda: run_multi("random", "baseline"),
    "three_sites_routed":   lambda: run_multi("eco", "baseline"),
    "three_sites_full":     lambda: run_multi("eco", "eco"),
}

rows, log_saved = [], None
for name, fn in SCENARIOS.items():
    if args.scenarios and name not in args.scenarios:
        continue
    print(f"running {name} ...")
    parts, extra = fn()
    if name in ("single_site_baseline", "three_sites_full"):                  # node states every 10 min, for the UI
        arrs = {}
        for mm, dd in parts:
            label = mm["policy"] if name.startswith("three") else "All"
            sn = dd["snapshots"]
            arrs[f"{label}_power"], arrs[f"{label}_temp"] = sn["power_w"], sn["temp_c"]
            arrs[f"{label}_gpus"], arrs[f"{label}_on"] = sn["gpus_busy"], sn["on"]
            arrs["times"] = np.array([str(t) for t in sn["time"]])
        np.savez_compressed(f"ms_snap_{name}.npz", **arrs)
    ms = [m for m, _ in parts]
    waits = pd.concat([d["jobs"].loc[d["jobs"]["in_window"] & d["jobs"]["start_s"].notna(), "wait_min"] for _, d in parts])
    kwh, kg = sum(m["facility_kwh"] for m in ms), sum(m["carbon_kg"] for m in ms)
    gh = sum(m["gpu_hours"] for m in ms)
    rows.append({"scenario": name, "facility_kwh": kwh, "carbon_kg": kg, "avg_ci_gco2_kwh": 1000 * kg / kwh,
                 "kwh_per_gpu_hour": kwh / gh, "g_co2_per_gpu_hour": 1000 * kg / gh, "thermal_node_hours_over_limit": sum(m["thermal_node_hours_over_limit"] for m in ms),
                 "mean_wait_min": waits.mean(), "p95_wait_min": waits.quantile(0.95), "gpu_hours": gh,
                 "jobs_unfinished": sum(m["jobs_unfinished"] for m in ms)})
    if extra is not None and name == "three_sites_full":
        extra[0].to_csv("ms_decisions.csv", index=False)
        log_saved = extra[1]
        pd.concat([d["jobs"] for _, d in parts]).to_csv("ms_jobs_full.csv", index=False)

res = pd.DataFrame(rows).set_index("scenario")
res.to_csv("multisite_results.csv")
pd.set_option("display.width", 250, "display.max_columns", None)
print("\n", res.round(3).to_string())
if "single_site_baseline" in res.index:
    rel = (res / res.loc["single_site_baseline"] - 1) * 100
    print("\nchange vs single_site_baseline (%):")
    print(rel.round(1).to_string())
if log_saved is not None and len(log_saved):
    print("\nshare of jobs sent to each site by the router:", pd.Series(log_saved).value_counts(normalize=True).round(2).to_dict())
print("\nsaved multisite_results.csv, ms_decisions.csv (per-job scores per site) and ms_jobs_full.csv")