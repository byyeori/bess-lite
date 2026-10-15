import json, glob, numpy as np, pandas as pd
R = [json.load(open(f)) for f in sorted(glob.glob("results2/metrics_seed*.json"))]
print("seeds:", [r["seed"] for r in R], "| test samples:", R[0]["n_test"])
rows = []
for r in R:
    for k in ["pv", "load", "wind"]:
        for m in ["baseline", "baseline9", "ours", "baseline_phys", "baseline9_phys", "ours_phys"]:
            rows.append(dict(seed=r["seed"], target=k, model=m, **r["metrics"][k][m]))
df = pd.DataFrame(rows); df.to_csv("results2/metrics_all_seeds.csv", index=False)
def fmt(g, col, nd): return f"{g[col].mean():.{nd}f} ± {g[col].std(ddof=1):.{nd}f}"
for unit, suf, nd in [("normalized", "", 4), ("original units", "_phys", 1)]:
    print(f"\n--- {unit} (mean ± sd over seeds) ---")
    for k in ["pv", "load", "wind"]:
        for m in ["baseline", "baseline9", "ours"]:
            g = df[(df.target == k) & (df.model == m + suf)]
            print(f"{k:5s} {m:10s} MSE {fmt(g,'MSE',nd+ (2 if suf=='' else 0))}  RMSE {fmt(g,'RMSE',nd)}  MAE {fmt(g,'MAE',nd)}")
print("\n--- paired per-seed RMSE: ours better than ... in how many seeds ---")
for k in ["pv", "load", "wind"]:
    o = df[(df.target == k) & (df.model == "ours")].set_index("seed").RMSE
    for m in ["baseline", "baseline9"]:
        b = df[(df.target == k) & (df.model == m)].set_index("seed").RMSE
        red = (1 - o / b) * 100
        print(f"{k:5s} vs {m:10s}: {int((o < b).sum())}/{len(o)} seeds, RMSE reduction {red.mean():.1f}% ± {red.std(ddof=1):.1f}%")
