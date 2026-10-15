"""Train baseline (CNN-LSTM, base_model.py) and proposed (MemoryModel, main.py) under one
evaluation protocol and export test-set predictions.

Protocol (same for both):
  - data/data.csv, chronological 70/15/15 split, scalers fit on train only
  - input 48 h, target t+1
  - each model keeps its own training recipe from its script (optimizer/lr/schedule/epochs=50)
  - checkpoint = lowest validation loss (MSE, normalized)
  - test metrics computed over the whole test set (global MSE/RMSE/MAE, normalized 0-1)
  - complexity: thop MACs at T=48, batch 1; latency: CPU, 1 thread, perf_counter, 1000 runs
Usage: python run_compare.py SEED
"""
import sys, copy, json, time, math, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from thop import profile
import main as M
import base_model as B

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
torch.set_num_threads(1)
T, H, EPOCHS = 48, 1, 50

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

raw = pd.read_csv("data/data.csv")
ts_all = pd.to_datetime(raw["timestamp"], format="%m/%d/%Y %H")
n = len(raw); tr, va = int(n * 0.7), int(n * 0.85)
test_ts = ts_all.values[va:]

def metrics(p, y):
    e = p - y
    return dict(MSE=float(np.mean(e**2)), RMSE=float(np.sqrt(np.mean(e**2))), MAE=float(np.mean(np.abs(e))))

def latency_ms(fn, runs=1000):
    with torch.no_grad():
        for _ in range(50): fn()
        t0 = time.perf_counter()
        for _ in range(runs): fn()
    return (time.perf_counter() - t0) / runs * 1000

# ---------------- proposed ----------------
def run_ours():
    seed_all(SEED)
    df = raw.rename(columns={"pv_kw": "pv_kW", "load_kw": "net_load", "wind_kw": "wind_kW"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H")
    df["hour"] = df["timestamp"].dt.hour; df["day_of_week"] = df["timestamp"].dt.dayofweek
    trd, vad, ted = df[:tr].copy(), df[tr:va].copy(), df[va:].copy()
    tr_ds = M.BESSDataset(trd, seq_len=T, pred_len=H)
    sc = (tr_ds.scaler_weather, tr_ds.scaler_signal, tr_ds.scaler_pv_y, tr_ds.scaler_net_y, tr_ds.scaler_wind_y)
    va_ds = M.BESSDataset(vad, seq_len=T, pred_len=H, scalers=sc)
    te_ds = M.BESSDataset(ted, seq_len=T, pred_len=H, scalers=sc)
    trl = DataLoader(tr_ds, batch_size=128, shuffle=True)
    model = M.MemoryModel(input_features=9, pred_len=H)
    opt = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.003, total_steps=len(trl) * EPOCHS)

    def predict(ds):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=256, shuffle=False):
                P.append(torch.cat(model(b["x"], b["hour_idx"], b["day_idx"]), -1).numpy())
                Y.append(torch.cat([b["pv_y"], b["net_y"], b["wind_y"]], -1).numpy())
        return np.concatenate(P), np.concatenate(Y)

    best, best_state = 1e9, None
    for ep in range(EPOCHS):
        M.train_epoch(model, trl, opt, sch, "cpu")
        p, y = predict(va_ds); v = float(np.mean((p - y) ** 2))
        if v < best: best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    P, Y = predict(te_ds)
    tidx = np.arange(len(P)) + T  # target row within test split
    xb = te_ds[0]
    x, h, d = xb["x"][None], xb["hour_idx"][None], xb["day_idx"][None]
    macs, params = profile(model, inputs=(x, h, d), verbose=False)
    model.eval(); lat = latency_ms(lambda: model(x, h, d))
    return P, Y, tidx, sc[2:], dict(params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                                    macs=float(macs), latency_ms=lat)

# ---------------- baseline ----------------
def run_base():
    seed_all(SEED)
    df = raw.drop(columns=["timestamp"]).copy()
    frames = [B.add_rolling_lag_features(df, c, {"pv_kw": "pv_", "load_kw": "net_", "wind_kw": "wind_"}[c], lags=[1, 2, 24, 168])
              for c in ["pv_kw", "load_kw", "wind_kw"]]
    df = pd.concat([df] + frames, axis=1)
    trd, vad, ted = df[:tr].copy(), df[tr:va].copy(), df[va:].copy()
    tc = ["pv_kw", "load_kw", "wind_kw"]
    tr_ds = B.MultitaskDataset(trd, seq_len=T, pred_horizons=[1], target_cols=tc)
    sc = (tr_ds.scaler_signal, tr_ds.scaler_history, tr_ds.scaler_pv_y, tr_ds.scaler_net_y, tr_ds.scaler_wind_y)
    va_ds = B.MultitaskDataset(vad, seq_len=T, pred_horizons=[1], scalers=sc, target_cols=tc)
    te_ds = B.MultitaskDataset(ted, seq_len=T, pred_horizons=[1], scalers=sc, target_cols=tc)
    trl = DataLoader(tr_ds, batch_size=128, shuffle=True)
    in_dim = tr_ds[0][0].shape[-1]
    model = B.CNNLSTMModel(in_dim, 3)
    opt = optim.Adam(model.parameters(), lr=0.001); crit = nn.MSELoss()

    def predict(ds):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for x, y in DataLoader(ds, batch_size=256, shuffle=False):
                P.append(model(x).numpy()); Y.append(y.numpy())
        return np.concatenate(P), np.concatenate(Y)

    best, best_state = 1e9, None
    for ep in range(EPOCHS):
        model.train()
        for x, y in trl:
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        p, y = predict(va_ds); v = float(np.mean((p - y) ** 2))
        if v < best: best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    P, Y = predict(te_ds)
    tidx = np.arange(len(P)) + T
    x = te_ds[0][0][None]
    macs, params = profile(model, inputs=(x,), verbose=False)
    model.eval(); lat = latency_ms(lambda: model(x))
    return P, Y, tidx, sc[2:], dict(params=int(sum(p.numel() for p in model.parameters())), macs=float(macs),
                                    latency_ms=lat, input_dim=int(in_dim))


# ---------------- baseline, same 9 inputs as proposed ----------------
def run_base9():
    seed_all(SEED)
    df = raw.rename(columns={"pv_kw": "pv_kW", "load_kw": "net_load", "wind_kw": "wind_kW"}).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H")
    df["hour"] = df["timestamp"].dt.hour; df["day_of_week"] = df["timestamp"].dt.dayofweek
    trd, vad, ted = df[:tr].copy(), df[tr:va].copy(), df[va:].copy()
    tr_ds = M.BESSDataset(trd, seq_len=T, pred_len=H)
    sc = (tr_ds.scaler_weather, tr_ds.scaler_signal, tr_ds.scaler_pv_y, tr_ds.scaler_net_y, tr_ds.scaler_wind_y)
    va_ds = M.BESSDataset(vad, seq_len=T, pred_len=H, scalers=sc)
    te_ds = M.BESSDataset(ted, seq_len=T, pred_len=H, scalers=sc)
    trl = DataLoader(tr_ds, batch_size=128, shuffle=True)
    model = B.CNNLSTMModel(9, 3)
    opt = optim.Adam(model.parameters(), lr=0.001); crit = nn.MSELoss()
    def predict(ds):
        model.eval(); P, Y = [], []
        with torch.no_grad():
            for b in DataLoader(ds, batch_size=256, shuffle=False):
                P.append(model(b["x"]).numpy()); Y.append(torch.cat([b["pv_y"], b["net_y"], b["wind_y"]], -1).numpy())
        return np.concatenate(P), np.concatenate(Y)
    best, best_state = 1e9, None
    for ep in range(EPOCHS):
        model.train()
        for b in trl:
            y = torch.cat([b["pv_y"], b["net_y"], b["wind_y"]], -1)
            opt.zero_grad(); loss = crit(model(b["x"]), y); loss.backward(); opt.step()
        p, y = predict(va_ds); v = float(np.mean((p - y) ** 2))
        if v < best: best, best_state = v, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    P, Y = predict(te_ds)
    return P, Y, np.arange(len(P)) + T

Po, Yo, io, sco, co = run_ours()
Pb, Yb, ib, scb, cb = run_base()
P9, Y9, i9 = run_base9()

# align on common target rows
common = np.intersect1d(np.intersect1d(io, ib), i9)
mo, mb = np.isin(io, common), np.isin(ib, common)
Po, Yo, Pb, Yb = Po[mo], Yo[mo], Pb[mb], Yb[mb]
P9 = P9[np.isin(i9, common)]
assert np.allclose(Yo, Yb, atol=1e-5), "targets differ between pipelines"

out = pd.DataFrame({"timestamp": test_ts[common]})
res = {"seed": SEED, "n_test": int(len(common)), "ours": co, "baseline": cb, "metrics": {}}
for i, k in enumerate(["pv", "load", "wind"]):
    inv = sco[i]
    tru = inv.inverse_transform(Yo[:, [i]]).ravel()
    phys = lambda P: inv.inverse_transform(P[:, [i]]).ravel()
    res["metrics"][k] = {"ours": metrics(Po[:, i], Yo[:, i]), "baseline": metrics(Pb[:, i], Yb[:, i]), "baseline9": metrics(P9[:, i], Yo[:, i]),
                         "ours_phys": metrics(phys(Po), tru), "baseline_phys": metrics(scb[i].inverse_transform(Pb[:, [i]]).ravel(), tru),
                         "baseline9_phys": metrics(phys(P9), tru)}
    out[f"{k}_true_n"], out[f"{k}_ours_n"], out[f"{k}_base_n"], out[f"{k}_base9_n"] = Yo[:, i], Po[:, i], Pb[:, i], P9[:, i]
    inv = sco[i]
    out[f"{k}_true"] = inv.inverse_transform(Yo[:, [i]]).ravel()
    out[f"{k}_ours"] = inv.inverse_transform(Po[:, [i]]).ravel()
    out[f"{k}_base"] = scb[i].inverse_transform(Pb[:, [i]]).ravel()
out.to_csv(f"results2/pred_seed{SEED}.csv", index=False)
json.dump(res, open(f"results2/metrics_seed{SEED}.json", "w"), indent=2)
print(json.dumps(res, indent=2))
