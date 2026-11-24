"""rebuild_summary.py - rebuilds the results table from the per-run log files, so nothing depends on results_summary.csv.
Only valid runs are used: finished runs made with the fixed fl_sim.py (they have a 'skipped' column) and no NaN losses."""
import glob, re
import pandas as pd
from eco_policy import summarize

rows = []
for f in sorted(glob.glob("fl_log_*_s*.csv")):
    m = re.match(r"fl_log_(.+)_s(\d+)\.csv$", f)
    if not m:
        continue
    name, seed = m.group(1), int(m.group(2))
    acc = pd.read_csv(f)
    try:
        pol = pd.read_csv(f"policy_log_{name}_s{seed}.csv", parse_dates=["start"])
    except FileNotFoundError:
        print("skipping", f, "(no matching policy log)")
        continue
    if "skipped" not in acc.columns or acc["test_loss"].isna().any() or len(acc) != len(pol):
        print("skipping", f, "(old, diverged or incomplete run)")
        continue
    row = summarize(acc, pol, 0.70)
    row.update(policy=name, seed=seed, dropped=int(acc["skipped"].sum()))
    rows.append(row)

res = pd.DataFrame(rows).sort_values(["policy", "seed"])
res.to_csv("results_all.csv", index=False)
cols = ["final_acc", "total_wh", "total_gco2", "violations", "wall_hours", "wh_to_target", "gco2_to_target"]
pd.set_option("display.width", 220, "display.max_columns", None)
print("\nruns used per policy:", res.groupby("policy")["seed"].apply(list).to_dict())
print("dropped diverged updates in total:", int(res["dropped"].sum()))
print("\n=== mean over seeds ===")
print(res.groupby("policy")[cols].mean().round(3).to_string())
print("\n=== standard deviation over seeds ===")
print(res.groupby("policy")[cols].std().round(3).to_string())
print("\nsaved results_all.csv")