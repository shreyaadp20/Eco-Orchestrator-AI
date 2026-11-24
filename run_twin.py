"""
run_twin.py - replays a real SuperCloud job trace through the data-center twin under several scheduling policies.

Quick check:  python run_twin.py --days 2 --policies baseline eco
Full run:     python run_twin.py
Needs: scheduler_data.csv, dcgm.csv, df_fuel_ckan.csv, power_B_sm_plus_mem.json, thermal_cpu*_load.json, host_power.json,
       carbon_model_holdout.json, and eco_models.py / dc_twin.py in this folder.
"""
import argparse
from pathlib import Path

import pandas as pd
from eco_models import EcoModels
from dc_twin import DataCenterTwin, POLICIES, load_trace, suggest_nodes, check_job_energy

ROOT = Path(__file__).resolve().parent


def input_path(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    return cwd_path if cwd_path.exists() else ROOT / path


p = argparse.ArgumentParser()
p.add_argument("--sched", default=str(ROOT / "scheduler_data.csv"))
p.add_argument("--dcgm", default=str(ROOT / "dcgm.csv"))
p.add_argument("--host-model", default=str(ROOT / "host_power.json"), help="XGBoost host power model")
p.add_argument("--carbon-model", default=str(ROOT / "carbon_model_holdout.json"), help="hold-out carbon model")
p.add_argument("--days", type=float, default=7, help="length of the replayed window")
p.add_argument("--offset-days", type=float, default=100, help="where in the trace the window starts")
p.add_argument("--nodes", type=int, default=0, help="0 = size the cluster for about 50%% average GPU utilisation")
p.add_argument("--start", default="2026-05-04 00:00", help="calendar date the window is replayed on (must be after 2026-04-17)")
p.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
p.add_argument("--warmup-days", type=float, default=1, help="arrivals simulated before the measured window")
p.add_argument("--defer-mode", default="all_but_interactive", choices=["all_but_interactive", "batch_only"])
p.add_argument("--set", nargs="*", default=[], help="override twin settings, e.g. --set supply_c=28 temp_limit_c=45 pue=1.3")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

inputs = {
    "scheduler trace": input_path(args.sched),
    "DCGM trace": input_path(args.dcgm),
    "host power model": input_path(args.host_model),
    "carbon model": input_path(args.carbon_model),
    "GPU power model": ROOT / "power_B_sm_plus_mem.json",
    "CPU thermal models": [
        ROOT / f"thermal_{cpu}_{kind}.json"
        for cpu in ("cpu1", "cpu2")
        for kind in ("load", "psu")
    ],
    "grid carbon data": ROOT / "df_fuel_ckan.csv",
}
missing = [
    f"{label}: {path}"
    for label, paths in inputs.items()
    for path in (paths if isinstance(paths, list) else [paths])
    if not path.is_file()
]
if missing:
    p.error("required input file(s) not found:\n  " + "\n  ".join(missing))

cfg = {}
for item in args.set:
    key, separator, value = item.partition("=")
    if not separator or not key:
        p.error(f"invalid --set value {item!r}; expected key=value")
    try:
        cfg[key] = float(value)
    except ValueError:
        p.error(f"invalid numeric value in --set {item!r}")

em = EcoModels(str(ROOT), carbon_file=str(inputs["carbon model"]))  # hold-out model: replay dates are unseen
jobs = load_trace(str(inputs["scheduler trace"]), str(inputs["DCGM trace"]),
                  days=args.days, offset_days=args.offset_days, seed=args.seed,
                  warmup_days=args.warmup_days, defer_mode=args.defer_mode)
n_nodes = args.nodes or suggest_nodes(jobs, args.days)
win = jobs[jobs["submit_s"] >= 0]
print(f"trace: {len(win)} GPU jobs in the measured window ({len(jobs) - len(win)} more in warm-up) | "
      f"{(win['gpus'] * win['dur_s']).sum() / 3600:.0f} GPU-hours requested | deferrable jobs: {100 * win['deferrable'].mean():.0f}% "
      f"| cluster: {n_nodes} nodes x 2 GPUs")
if jobs["meas_w"].notna().any():
    print("pipeline check (twin GPU power vs measured):",
          {k: round(v, 3) for k, v in check_job_energy(jobs, em).items()})
else:
    print("pipeline check: no matched DCGM measurements in this trace window")

twin = DataCenterTwin(em, host_model=str(inputs["host power model"]), cfg=cfg)
rows = []
for name in args.policies:
    print(f"running {name} ...")
    metrics, ts = twin.run(jobs, n_nodes, name, start=args.start, horizon_days=args.days,
                           warmup_days=args.warmup_days, seed=args.seed)
    ts.to_csv(f"twin_ts_{name}.csv", index=False)
    rows.append(metrics)

res = pd.DataFrame(rows).set_index("policy")
res.to_csv("twin_results.csv")
pd.set_option("display.width", 240, "display.max_columns", None)
show = ["facility_kwh", "carbon_kg", "avg_ci_gco2_kwh", "kwh_per_gpu_hour", "peak_kw",
        "thermal_node_hours_over_limit", "mean_wait_min", "p95_wait_min", "gpu_hours", "jobs_unfinished"]
print("\n", res[show].round(3).to_string())
if "baseline" in res.index:
    print("\nchange vs baseline (%):")
    rel = (res[show[:-1]] / res.loc["baseline", show[:-1]] - 1) * 100
    print(rel.round(1).to_string())
print("\nsaved twin_results.csv and twin_ts_<policy>.csv")