import pandas as pd
r = pd.concat([pd.read_csv("results_summary_seed0.csv"), pd.read_csv("results_summary.csv")])
r.to_csv("results_all.csv", index=False)
cols = ["final_acc", "total_wh", "total_gco2", "violations", "wall_hours", "wh_to_target", "gco2_to_target"]
print(r.groupby("policy")[cols].agg(["mean", "std"]).round(3).T)