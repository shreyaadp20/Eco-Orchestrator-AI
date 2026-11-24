"""find_window.py - finds days of the SuperCloud trace where GPU jobs have DCGM measurements, so the twin replays a window with real data.
Run:  python find_window.py        then use the printed --offset-days value with run_twin.py"""
import numpy as np, pandas as pd

s = pd.read_csv("scheduler_data.csv", usecols=["id_job", "gres_req", "time_submit", "time_start", "time_end"], dtype={"id_job": "int64"})
dup = s.loc[s.duplicated("id_job", keep=False), "id_job"].unique()
s = s[~s["id_job"].isin(dup)].copy()
for c in ("time_submit", "time_start", "time_end"):
    s[c] = s[c].where((s[c] >= 0) & (s[c] < 4e7))
s = s[s["gres_req"].astype(str).str.contains("gpu")].dropna(subset=["time_submit", "time_start", "time_end"])
s = s[(s["time_end"] - s["time_start"] >= 30) & (s["time_end"] - s["time_start"] <= 72 * 3600)]
d_ids = set(pd.read_csv("dcgm.csv", usecols=["id_job"], dtype={"id_job": "int64"})["id_job"])
s["has_dcgm"] = s["id_job"].isin(d_ids)
t0 = s["time_submit"].min()
s["day"] = ((s["time_submit"] - t0) // 86400).astype(int)
daily = s.groupby("day").agg(gpu_jobs=("id_job", "size"), with_dcgm=("has_dcgm", "sum"))
daily["share_pct"] = (100 * daily["with_dcgm"] / daily["gpu_jobs"]).round(1)
print("GPU jobs:", len(s), "| with DCGM data:", int(s["has_dcgm"].sum()), f"({100 * s['has_dcgm'].mean():.1f}%)")
print("days in trace that have GPU jobs:", len(daily))
for span in (2, 7):
    roll = daily.reindex(range(daily.index.max() + 1), fill_value=0).rolling(span).sum()
    roll["share_pct"] = (100 * roll["with_dcgm"] / roll["gpu_jobs"].replace(0, np.nan)).round(1)
    roll = roll.dropna()
    best = roll[roll["gpu_jobs"] >= 300].sort_values("share_pct", ascending=False).head(5)
    print(f"\nbest {span}-day windows (at least 300 GPU jobs). 'day' is the LAST day of the window:")
    for day, r in best.iterrows():
        print(f"  use --offset-days {int(day) - span + 1}  -> {int(r.gpu_jobs)} GPU jobs, {r.share_pct}% with DCGM data")