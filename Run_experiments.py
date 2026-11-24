"""
run_experiment.py - runs the FL simulation under several client-selection policies and seeds, then compares them.

Quick check (~4 min):  python run_experiment.py --rounds 20 --seeds 0 --policies random full
Full comparison:       python run_experiment.py            (random, energy, full; seeds 0 1 2; about 10 min per run)

Needs in this folder: fl_data.py, fl_sim.py, eco_models.py, eco_policy.py, df_fuel_ckan.csv, power_B_sm_plus_mem.json,
thermal_cpu*_load.json / thermal_cpu*_psu.json, carbon_model_holdout.json (made by carbon_model_v3.py).
"""
import argparse
import pandas as pd
from datasets import load_dataset

import fl_sim
from eco_models import EcoModels
from eco_policy import EcoPolicy, summarize

POLICIES = {
    "random":  dict(),                                                     # baseline
    "timing":  dict(use_timing=True),                                      # random clients, rounds shifted to cleaner hours
    "energy":  dict(use_energy=True),                                      # loss-based utility minus energy cost
    "thermal": dict(use_thermal=True),                                     # random among clients predicted below the temperature limit
    "full":    dict(use_energy=True, use_timing=True, use_thermal=True),   # everything together
}

p = argparse.ArgumentParser()
p.add_argument("--policies", nargs="+", default=["random", "energy", "full"], choices=list(POLICIES))
p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--rounds", type=int, default=100)
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--clients", type=int, default=100)
p.add_argument("--per-round", type=int, default=10)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--min-samples", type=int, default=50)
p.add_argument("--target", type=float, default=0.70, help="accuracy used for the 'cost to reach target' columns")
args = p.parse_args()

em = EcoModels(".", carbon_file="carbon_model_holdout.json")      # hold-out carbon model: the simulated dates are unseen by it
ds = load_dataset("flwrlabs/femnist", split="train")

rows = []
for seed in args.seeds:
    for name in args.policies:
        print(f"\n=== policy {name} | seed {seed} ===")
        a = argparse.Namespace(clients=args.clients, per_round=args.per_round, rounds=args.rounds, epochs=args.epochs,
                               lr=args.lr, batch=args.batch, min_samples=args.min_samples, seed=seed)
        start = pd.Timestamp("2026-05-01") + pd.Timedelta(days=10 * seed)     # different weather for each seed
        policy = EcoPolicy(em, args.epochs, seed=seed, start=start, **POLICIES[name])
        acc = fl_sim.run(ds, a, selector=policy, tag=f"{name}_s{seed}")
        pol_log = policy.log_df()
        pol_log.to_csv(f"policy_log_{name}_s{seed}.csv", index=False)
        row = summarize(acc, pol_log, args.target)
        row.update(policy=name, seed=seed)
        rows.append(row)
        pd.DataFrame(rows).to_csv("results_summary.csv", index=False)       # saved after every run

res = pd.DataFrame(rows)
cols = ["final_acc", "total_wh", "total_gco2", "violations", "wall_hours", "wh_to_target", "gco2_to_target"]
pd.set_option("display.width", 200, "display.max_columns", None)
print("\n=== mean over seeds ===")
print(res.groupby("policy")[cols].mean().round(3).to_string())
print("\nfull per-run results are in results_summary.csv")