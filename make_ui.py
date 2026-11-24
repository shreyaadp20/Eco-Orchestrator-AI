"""
make_ui.py - builds a self-contained dashboard (one HTML file) from the outputs of run_multisite.py.

Run run_multisite.py first (it saves ms_snap_*.npz, ms_decisions.csv, ms_jobs_full.csv, multisite_results.csv), then:
    python make_ui.py
and double-click eco_twin_dashboard.html. Nothing else is needed: the data is embedded in the file.
"""
import argparse, json
import numpy as np, pandas as pd
from eco_models import EcoModels
from eco_router import SITES, make_site
from dc_twin import CFG

TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EcoOrchestrator AI - Data Center Digital Twin</title>
<style>
:root{--bg:#f4f6f8;--card:#ffffff;--ink:#1b2430;--muted:#5f6b7a;--line:#dde2e8;--accent:#1a7f4b;--accent-soft:#e3f4ea;--bad:#b42318;
--t-off:#c9ced6;--t-idle:#d8ecdf;--t-1:#79c496;--t-2:#1a7f4b;--hot:#d92d20;--before:#8a94a3;--after:#1a7f4b}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#171d27;--ink:#e7ebf1;--muted:#9aa5b5;--line:#2a3343;--accent:#58d08a;--accent-soft:#1c3327;--bad:#f97066;
--t-off:#2b3342;--t-idle:#24382d;--t-1:#3c9a63;--t-2:#58d08a;--hot:#f97066;--before:#7d8797;--after:#58d08a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1280px;margin:0 auto;padding:16px}
h1{font-size:20px;margin:0}
.sub{color:var(--muted);margin:2px 0 12px}
.banner{background:#fff4d6;color:#6b4e00;border:1px solid #f0d98a;border-radius:8px;padding:8px 12px;margin-bottom:12px;font-weight:600}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:12px}
button,select{font:inherit;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff;min-width:84px}
input[type=range]{flex:1;min-width:200px;accent-color:var(--accent)}
.clock{font-variant-numeric:tabular-nums;font-weight:600;min-width:150px}
.row{display:grid;gap:12px;margin-bottom:12px}
.row.two{grid-template-columns:1fr 1fr}
@media (max-width:900px){.row.two{grid-template-columns:1fr}}
.title{font-weight:700;margin:0 0 8px;font-size:14px}
.muted{color:var(--muted)}
.sites{display:grid;gap:10px;grid-template-columns:repeat(3,1fr)}
@media (max-width:1100px){.sites{grid-template-columns:1fr}}
.panel{border:1px solid var(--line);border-radius:10px;padding:10px;transition:box-shadow .3s,border-color .3s}
.panel.flash{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
.panel h3{margin:0 0 2px;font-size:14px}
.facts{font-size:12px;color:var(--muted);margin-bottom:8px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(16px,1fr));gap:3px}
.tile{height:16px;border-radius:3px;background:var(--t-idle);border:2px solid transparent}
.tile.s-1{background:var(--t-off)} .tile.s0{background:var(--t-idle)} .tile.s1{background:var(--t-1)} .tile.s2{background:var(--t-2)}
.tile.hot{border-color:var(--hot)}
.tile.pulse{animation:pulse .9s ease-out}
@keyframes pulse{0%{transform:scale(1.9);box-shadow:0 0 0 4px var(--accent)}100%{transform:scale(1);box-shadow:0 0 0 0 transparent}}
.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}
.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px;border:2px solid transparent}
.opt{display:grid;grid-template-columns:70px 1fr 130px;gap:8px;align-items:center;margin:5px 0;font-size:13px}
.opt.win .name{font-weight:700;color:var(--accent)}
.bar{height:12px;border-radius:6px;background:var(--line);overflow:hidden}
.bar>b{display:block;height:100%;background:var(--before);border-radius:6px}
.opt.win .bar>b{background:var(--accent)}
.opt .val{font-size:12px;color:var(--muted);text-align:right;font-variant-numeric:tabular-nums}
.why{margin-top:8px;padding:8px 10px;border-radius:8px;background:var(--accent-soft);font-size:13px}
.big{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}
.kpi .v{font-size:28px;font-weight:800;color:var(--accent);font-variant-numeric:tabular-nums}
.kpi .l{font-size:12px;color:var(--muted)}
.kpi .d{font-size:12px;font-variant-numeric:tabular-nums}
.feed{list-style:none;margin:0;padding:0;font-size:13px}
.feed li{padding:5px 0;border-bottom:1px solid var(--line);display:flex;justify-content:space-between;gap:8px}
.feed li:last-child{border-bottom:0}
.tag{font-weight:700;color:var(--accent)}
svg{width:100%;height:auto;display:block}
.foot{font-size:12px;color:var(--muted);margin-top:6px}
</style>
</head>
<body>
<div class="wrap">
  <h1>EcoOrchestrator AI - Data Center Digital Twin</h1>
  <div class="sub" id="sub"></div>
  <div class="banner" id="banner" style="display:none">PREVIEW: this page was built from stand-in data, not from the final experiment. Rebuild it with make_ui.py on your own run.</div>

  <div class="card controls">
    <button class="primary" id="play">Play</button>
    <label>Speed <select id="speed">
      <option value="10">10 min / second</option><option value="30" selected>30 min / second</option>
      <option value="60">1 hour / second</option><option value="180">3 hours / second</option></select></label>
    <input type="range" id="slider" min="0" value="0">
    <span class="clock" id="clock"></span>
  </div>

  <div class="row two">
    <div class="card" id="workload"></div>
    <div class="card">
      <p class="title">Savings so far (EcoOrchestrator vs normal scheduling)</p>
      <div class="big" id="kpis"></div>
      <div class="foot" id="kpifoot"></div>
    </div>
  </div>

  <div class="row two" style="grid-template-columns:1fr 2fr">
    <div class="card">
      <p class="title">Before: one data center, first come first served</p>
      <div class="panel" id="p-before"><div class="facts" id="f-before"></div><div class="grid" id="g-before"></div></div>
    </div>
    <div class="card">
      <p class="title">After: three sites, each workload sent to the best one</p>
      <div class="sites" id="sites"></div>
    </div>
  </div>
  <div class="card legend" style="margin-bottom:12px">
    <span><i style="background:var(--t-off)"></i>switched off</span>
    <span><i style="background:var(--t-idle)"></i>on, idle</span>
    <span><i style="background:var(--t-1)"></i>1 GPU busy</span>
    <span><i style="background:var(--t-2)"></i>2 GPUs busy</span>
    <span><i style="border-color:var(--hot)"></i>CPU above temperature limit</span>
  </div>

  <div class="row two">
    <div class="card">
      <p class="title">Cumulative carbon emitted (kg CO2)</p>
      <svg viewBox="0 0 600 200" id="chart"></svg>
      <div class="legend"><span><i style="background:var(--before)"></i>before</span><span><i style="background:var(--after)"></i>after</span></div>
    </div>
    <div class="card">
      <p class="title">Latest decisions</p>
      <ul class="feed" id="feed"></ul>
    </div>
  </div>
  <div class="foot" id="foot"></div>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
const D = JSON.parse(document.getElementById('data').textContent);
const T = D.times.length, STEP = D.step_min, END = (T - 1) * STEP;
const $ = id => document.getElementById(id);
const fmt = (x, d) => (x === null || x === undefined || isNaN(x)) ? '-' : Number(x).toFixed(d === undefined ? 1 : d);
const co2 = kg => kg >= 1 ? fmt(kg, 2) + ' kg' : fmt(kg * 1000, kg < 0.01 ? 1 : 0) + ' g';
const en = kwh => kwh >= 1 ? fmt(kwh, 2) + ' kWh' : fmt(kwh * 1000, 0) + ' Wh';
let t = 0, playing = false, speed = 30, last = null, evPtr = 0, stPtr = 0, lastIdx = -1;
const tiles = {};
const starts = D.events.filter(e => e.start !== null && e.nodes.length).map(e => ({t: e.start, site: e.site, nodes: e.nodes})).sort((a, b) => a.t - b.t);

function makeGrid(el, key, n){ el.innerHTML = ''; tiles[key] = [];
  for (let i = 0; i < n; i++){ const d = document.createElement('div'); d.className = 'tile s0'; el.appendChild(d); tiles[key].push(d); } }

function build(){
  $('sub').textContent = D.events.length + ' real GPU jobs replayed on simulated servers - power, temperature and grid carbon are computed every minute';
  if (D.preview) $('banner').style.display = 'block';
  $('slider').max = END; $('slider').step = STEP;
  makeGrid($('g-before'), 'before', D.before.nodes);
  const sites = $('sites'); sites.innerHTML = '';
  D.site_names.forEach(n => {
    const s = D.after[n], p = document.createElement('div'); p.className = 'panel'; p.id = 'p-' + n;
    p.innerHTML = '<h3>' + n + '</h3><div class="facts" id="f-' + n + '"></div><div class="grid" id="g-' + n + '"></div>';
    sites.appendChild(p); makeGrid($('g-' + n), n, s.nodes);
  });
  $('foot').textContent = 'Simulation notes: workloads are real SuperCloud jobs; power, thermal and carbon come from models trained on real data. ' +
    'The three sites and their cooling, GPU efficiency and grid carbon differences are assumed, so the savings show the method, not a real facility.';
}

function paint(key, blk, i){
  const st = blk.state[i], tp = blk.temp[i], arr = tiles[key];
  for (let k = 0; k < arr.length; k++){
    const hot = st[k] >= 0 && tp[k] > D.limit_c;
    arr[k].className = 'tile s' + st[k] + (hot ? ' hot' : '');
    arr[k].title = 'node ' + k + ': ' + (st[k] < 0 ? 'off' : st[k] + ' GPU(s) busy') + ', CPU ' + fmt(tp[k], 0) + ' C';
  }
}
function facts(blk, i){ const on = blk.state[i].filter(x => x >= 0).length;
  return 'grid now ' + fmt(blk.ci[i], 0) + ' gCO2/kWh | cooling overhead x' + fmt(blk.pue, 2) + ' | nodes on ' + on + '/' + blk.nodes + ' | ' + fmt(blk.kw[i], 1) + ' kW'; }

function chart(i){
  const W = 600, H = 200, mx = Math.max(D.cum.before_kg[T - 1], D.cum.after_kg[T - 1], 1e-9);
  const pts = arr => { let s = ''; for (let k = 0; k <= i; k++) s += (k / (T - 1) * W).toFixed(1) + ',' + (H - 8 - arr[k] / mx * (H - 16)).toFixed(1) + ' '; return s; };
  $('chart').innerHTML = '<line x1="0" y1="' + (H - 8) + '" x2="' + W + '" y2="' + (H - 8) + '" stroke="var(--line)"/>' +
    '<polyline fill="none" stroke="var(--before)" stroke-width="2.5" points="' + pts(D.cum.before_kg) + '"/>' +
    '<polyline fill="none" stroke="var(--after)" stroke-width="2.5" points="' + pts(D.cum.after_kg) + '"/>';
}

function kpis(i){
  const bk = D.cum.before_kwh[i], ak = D.cum.after_kwh[i], bc = D.cum.before_kg[i], ac = D.cum.after_kg[i];
  const pe = bk > 0 ? 100 * (1 - ak / bk) : 0, pc = bc > 0 ? 100 * (1 - ac / bc) : 0;
  $('kpis').innerHTML =
    '<div class="kpi"><div class="v">' + fmt(pe, 0) + '%</div><div class="l">less energy</div><div class="d">' + fmt(bk, 0) + ' kWh before, ' + fmt(ak, 0) + ' kWh after</div></div>' +
    '<div class="kpi"><div class="v">' + fmt(pc, 0) + '%</div><div class="l">less carbon</div><div class="d">' + fmt(bc, 1) + ' kg before, ' + fmt(ac, 1) + ' kg after</div></div>';
  const h = D.headline;
  $('kpifoot').textContent = (h.before && h.after && h.before.mean_wait_min !== undefined)
    ? 'Trade-off: average job wait was ' + fmt(h.before.mean_wait_min, 0) + ' min before and ' + fmt(h.after.mean_wait_min, 0) + ' min after, because flexible jobs are held for cleaner hours.' : '';
}

function showEvent(e){
  const names = D.site_names, vals = names.map(n => e.est[n] ? e.est[n][1] : null);
  const mx = Math.max.apply(null, vals.filter(v => v !== null).concat([1e-12]));
  let h = '<p class="title">Incoming workload #' + e.id + ' <span class="muted">needs ' + e.gpus + ' GPU' + (e.gpus > 1 ? 's' : '') + ' for ' + fmt(e.dur_min, 0) + ' min</span></p>';
  names.forEach((n, k) => {
    const est = e.est[n], win = n === e.site;
    h += '<div class="opt' + (win ? ' win' : '') + '"><span class="name">' + n + '</span><div class="bar"><b style="width:' + (est ? Math.max(3, 100 * est[1] / mx) : 0) + '%"></b></div>' +
      '<span class="val">' + (est ? co2(est[1]) + ' CO2, ' + en(est[0]) : 'site full') + (win ? ' &larr;' : '') + '</span></div>';
  });
  const others = names.filter(n => n !== e.site && e.est[n]).map(n => e.est[n][1]);
  const mine = e.est[e.site] ? e.est[e.site][1] : null;
  let why;
  if (mine !== null && others.length){ const avg = others.reduce((a, b) => a + b, 0) / others.length, pct = 100 * (1 - mine / avg);
    why = 'Sent to ' + e.site + ': ' + (pct >= 0 ? fmt(pct, 0) + '% less' : fmt(-pct, 0) + '% more') + ' carbon than the other options on average (' + co2(mine) + ' vs ' + co2(avg) + '). Other sites are weighed on energy, heat and how full they are.'; }
  else why = 'Sent to ' + e.site + ': the other sites were full or too small.';
  h += '<div class="why">' + why + '</div>';
  $('workload').innerHTML = h;
  const li = document.createElement('li'); li.innerHTML = '<span>#' + e.id + ' &middot; ' + e.gpus + ' GPU &middot; ' + fmt(e.dur_min, 0) + ' min</span><span class="tag">' + e.site + '</span>';
  const feed = $('feed'); feed.insertBefore(li, feed.firstChild); while (feed.children.length > 8) feed.removeChild(feed.lastChild);
  const p = $('p-' + e.site); if (p){ p.classList.add('flash'); setTimeout(() => p.classList.remove('flash'), 700); }
}

function render(i){
  if (i !== lastIdx){ lastIdx = i;
    paint('before', D.before, i); $('f-before').textContent = facts(D.before, i);
    D.site_names.forEach(n => { paint(n, D.after[n], i); $('f-' + n).textContent = facts(D.after[n], i); });
    kpis(i); chart(i); }
  $('clock').textContent = D.times[i].replace('T', ' ');
  $('slider').value = t;
}
function pulseNodes(s){ const arr = tiles[s.site]; if (!arr) return;
  s.nodes.forEach(k => { const el = arr[k]; if (el){ el.classList.remove('pulse'); void el.offsetWidth; el.classList.add('pulse'); setTimeout(() => el.classList.remove('pulse'), 950); } }); }

function advanceEvents(quiet){
  while (evPtr < D.events.length && D.events[evPtr].t <= t){ if (!quiet || D.events[evPtr].t > t - 30) showEvent(D.events[evPtr]); evPtr++; }
  while (stPtr < starts.length && starts[stPtr].t <= t){ if (!quiet) pulseNodes(starts[stPtr]); stPtr++; }
}
function seek(newT){
  t = Math.max(0, Math.min(END, newT)); evPtr = 0; stPtr = 0; $('feed').innerHTML = '';
  while (evPtr < D.events.length && D.events[evPtr].t <= t) evPtr++;
  while (stPtr < starts.length && starts[stPtr].t <= t) stPtr++;
  const recent = D.events.slice(Math.max(0, evPtr - 8), evPtr); recent.forEach(showEvent);
  if (!recent.length) $('workload').innerHTML = '<p class="title">Incoming workload</p><p class="muted">Press Play. Each time a workload arrives, the twin scores every site and sends it to the best one.</p>';
  lastIdx = -1; render(Math.min(T - 1, Math.floor(t / STEP)));
}
function step(dtMin){ t = Math.min(END, t + dtMin); advanceEvents(false); render(Math.min(T - 1, Math.floor(t / STEP))); if (t >= END) setPlaying(false); }
function setPlaying(p){ playing = p; $('play').textContent = p ? 'Pause' : (t >= END ? 'Replay' : 'Play'); last = null; }
function loop(ts){ if (playing){ if (last !== null) step(speed * Math.min(0.25, (ts - last) / 1000)); last = ts; } requestAnimationFrame(loop); }

build();
$('play').onclick = () => { if (t >= END) seek(0); setPlaying(!playing); };
$('speed').onchange = e => { speed = Number(e.target.value); };
$('slider').oninput = e => seek(Number(e.target.value));
seek(0);
requestAnimationFrame(loop);
window.__eco = {seek: seek, step: step, data: D};
})();
</script>
</body>
</html>
'''

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="eco_twin_dashboard.html")
ap.add_argument("--preview", action="store_true", help="adds a banner saying the data is stand-in data")
args = ap.parse_args()

em = EcoModels(".", carbon_file="carbon_model_holdout.json")
hooks = {k: make_site(em, k, v) for k, v in SITES.items()}
full, base = np.load("ms_snap_three_sites_full.npz"), np.load("ms_snap_single_site_baseline.npz")
times = pd.to_datetime(full["times"])
T = len(times)
STEP_H = 10 / 60


def block(npz, label, pue, ci_fn):
    power, temp, gpus, on = npz[f"{label}_power"], npz[f"{label}_temp"], npz[f"{label}_gpus"], npz[f"{label}_on"]
    kw = power.sum(1) / 1000 * pue
    ci = np.array([ci_fn(t) for t in times])
    state = np.where(on, gpus, -1).astype(int)                                  # -1 = switched off, otherwise GPUs busy (0-2)
    kwh = np.cumsum(kw * STEP_H)
    kg = np.cumsum(kw * STEP_H * ci / 1000)
    return {"nodes": int(power.shape[1]), "pue": pue, "state": state.tolist(), "temp": np.round(temp, 1).tolist(),
            "kw": np.round(kw, 2).tolist(), "ci": np.round(ci, 0).tolist()}, kwh, kg


after_sites, kwh_a, kg_a = {}, np.zeros(T), np.zeros(T)
for name, spec in SITES.items():
    after_sites[name], k1, k2 = block(full, name, spec["pue"], hooks[name]["ci_actual"])
    kwh_a += k1; kg_a += k2
before, kwh_b, kg_b = block(base, "All", CFG["pue"], em.carbon_actual)

# ---- decisions: one event per workload that arrived inside the measured window
dec, jobs = pd.read_csv("ms_decisions.csv"), pd.read_csv("ms_jobs_full.csv")
m = dec.merge(jobs[["id_job", "submit_s", "dur_s", "start_s", "end_s", "wait_min", "in_window", "nodes"]], on="id_job", how="inner")
m = m[m["in_window"]].sort_values("submit_s")
names = list(SITES)
events = []
for r in m.itertuples(index=False):
    row = r._asdict()
    est = {n: ([round(float(row[f"kwh_{n}"]), 4), round(float(row[f"kg_{n}"]), 5)] if pd.notna(row.get(f"kwh_{n}")) else None) for n in names}
    nodes = [int(x) for x in str(row["nodes"]).split("|") if x.strip().isdigit()] if pd.notna(row["nodes"]) else []
    events.append({"t": round(row["submit_s"] / 60, 2), "id": int(row["id_job"]), "gpus": int(row["gpus"]),
                   "dur_min": round(row["dur_s"] / 60, 1), "site": row["site"], "est": est,
                   "start": round(row["start_s"] / 60, 2) if pd.notna(row["start_s"]) else None,
                   "wait": round(float(row["wait_min"]), 1) if pd.notna(row["wait_min"]) else None, "nodes": nodes})

res = pd.read_csv("multisite_results.csv").set_index("scenario")
head = {}
for sc, key in (("single_site_baseline", "before"), ("three_sites_full", "after")):
    if sc in res.index:
        r = res.loc[sc]
        head[key] = {c: round(float(r[c]), 3) for c in ("facility_kwh", "carbon_kg", "mean_wait_min", "g_co2_per_gpu_hour") if c in r.index}
print(f"dashboard sampling check (10-minute samples vs the per-minute totals in multisite_results.csv):")
if head:
    print(f"  energy before: {kwh_b[-1]:.0f} vs {head['before']['facility_kwh']:.0f} kWh | after: {kwh_a[-1]:.0f} vs {head['after']['facility_kwh']:.0f} kWh")
    print(f"  carbon before: {kg_b[-1]:.1f} vs {head['before']['carbon_kg']:.1f} kg | after: {kg_a[-1]:.1f} vs {head['after']['carbon_kg']:.1f} kg")

data = {"preview": bool(args.preview), "times": [str(t)[:16] for t in times], "step_min": 10, "limit_c": CFG["temp_limit_c"],
        "site_names": names, "before": before, "after": after_sites, "events": events, "headline": head,
        "cum": {"before_kwh": np.round(kwh_b, 2).tolist(), "before_kg": np.round(kg_b, 3).tolist(),
                "after_kwh": np.round(kwh_a, 2).tolist(), "after_kg": np.round(kg_a, 3).tolist()}}
html = TEMPLATE
payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
open(args.out, "w", encoding="utf-8").write(html.replace("__DATA__", payload))
print(f"wrote {args.out} ({len(events)} workloads, {T} time steps). Double-click it to open.")
