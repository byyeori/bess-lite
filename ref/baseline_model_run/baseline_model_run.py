# -*- coding: utf-8 -*-
import os, sys, time, json, argparse, statistics, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
try:
    import psutil
except Exception:
    psutil = None

# ---------- utils ----------
def set_seed(seed=11, deterministic=True, device="cpu"):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

class SeqDataset(Dataset):
    def __init__(self, X, y, ts=None):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32)).reshape(-1,1)
        self.ts = ts
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        if self.ts is None: return self.X[i], self.y[i]
        return self.X[i], self.y[i], self.ts[i]

def load_csv(csv_path, time_col):
    if not os.path.exists(csv_path): raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    need = ["load_kW","price","pv_kW","net_load"]
    miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"Missing columns: {miss}")
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            df = df.sort_values(time_col).reset_index(drop=True)
        except Exception:
            df = df.reset_index(drop=True)
    for c in need: df[c] = pd.to_numeric(df[c], errors="coerce")
    before=len(df)
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=need)
    print(f">> cleaned rows: kept {len(df)}/{before} (dropped {before-len(df)})")
    if len(df)==0: raise ValueError("All rows dropped after cleaning.")
    return df

def make_supervised(df, feats, tgt, seq_len, horizon):
    vals = df[feats+[tgt]].values.astype(np.float32)
    N = len(vals) - seq_len - horizon + 1
    if N <= 0: raise ValueError("Not enough rows.")
    X = np.stack([vals[i:i+seq_len, :len(feats)] for i in range(N)])          # [N,T,F]
    y = np.array([vals[i+seq_len+horizon-1, -1] for i in range(N)], np.float32)
    mask = np.isfinite(X).all((1,2)) & np.isfinite(y)
    if mask.sum()==0: raise ValueError("All sequences non-finite.")
    if mask.sum()<len(X): print(f">> filtered sequences: kept {mask.sum()}/{len(X)}")
    return X[mask], y[mask]

def split_indices(n, val_ratio=0.1, test_ratio=0.1, seed=11):
    rng = np.random.default_rng(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_test = int(n*test_ratio); n_val = int(n*val_ratio); n_train = n - n_val - n_test
    if min(n_train,n_val,n_test)<=0: raise ValueError(f"Split invalid: N={n}")
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def fit_std_feats(X, idx):
    Xt = X[idx].reshape(len(idx), -1, X.shape[-1])  # [Ntr,T,F]
    mu = Xt.mean(axis=(0,1)); sd = Xt.std(axis=(0,1)); sd[sd==0]=1.0
    return mu, sd
def apply_std_feats(X, mu, sd): return (X - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)
def fit_std_target(y, idx):
    mu = y[idx].mean(); sd = y[idx].std(); sd = 1.0 if sd==0 else sd
    return float(mu), float(sd)

# ---------- model (Ozan-style: Conv×3 + BN + LSTM + FC) ----------
class OzanCNNLSTM(nn.Module):
    def __init__(self, in_features, hidden_size=64, num_layers=1):
        super().__init__()
        # conv stack
        self.conv1 = nn.Conv1d(in_features, 64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=1, stride=1, padding=1)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=1, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        # LSTM expects feature size = 32 (conv channels)
        self.lstm  = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc    = nn.Linear(hidden_size, 1)

    def forward(self, x):          # x: [B,T,F]
        x = x.transpose(1,2)       # [B,F,T]
        x = torch.selu(self.conv1(x))
        x = self.conv2(x)
        x = torch.selu(self.bn1(x))
        x = self.conv3(x)
        x = torch.selu(self.bn2(x))  # [B,32,T]
        x = x.transpose(1,2)         # [B,T,32]
        out, _ = self.lstm(x)        # [B,T,H]
        out = self.fc(out[:,-1,:])   # [B,1]  (마지막 타임스텝)
        return out

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------- train/eval ----------
def train(model, trL, vaL, epochs=20, lr=1e-3, early=6, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best=float("inf"); keep=None; patience=early
    for ep in range(1, epochs+1):
        model.train(); run=0.0; nobs=0
        for xb,yb,*_ in trL:
            xb=xb.to(device); yb=yb.to(device)
            pred=model(xb); loss=F.mse_loss(pred,yb)
            if not torch.isfinite(loss): continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()*xb.size(0); nobs += xb.size(0)
        tr = run/max(nobs,1)
        model.eval(); v=0.0; nobs=0
        with torch.no_grad():
            for batch in vaL:
                if len(batch)==2: xb,yb=batch
                else: xb,yb,_=batch
                xb=xb.to(device); yb=yb.to(device)
                l = F.mse_loss(model(xb), yb)
                if torch.isfinite(l): v += l.item()*xb.size(0); nobs+=xb.size(0)
        vl = v/max(nobs,1)
        print(f"epoch {ep:02d}  trainMSE {tr:.6f}  valMSE {vl:.6f}")
        if vl < best - 1e-9:
            best=vl; keep={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; patience=early
        else:
            patience -= 1
            if patience<=0: print("early stopping."); break
    if keep: model.load_state_dict(keep)
    return model

def evaluate(model, loader, device="cpu"):
    model.eval(); yps=[]; yts=[]; tss=[]
    with torch.no_grad():
        for batch in loader:
            if len(batch)==2: xb,yb=batch; ts=None
            else: xb,yb,ts=batch
            yp = model(xb.to(device)).detach().cpu()
            yps.append(yp); yts.append(yb); tss.append(ts)
    yp=torch.cat(yps,0).numpy().flatten(); yt=torch.cat(yts,0).numpy().flatten()
    ts=None
    if all(t is not None for t in tss): ts = np.concatenate(tss,0)
    return yt, yp, ts

# ---------- benchmark ----------
def bench_infer(model, sample_np, device="cpu", runs=1000, save_path="bench.json"):
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc: gc.collect(); rss_before = proc.memory_info().rss
    torch.save(model.state_dict(), f"baseline_fp32_{device}.pt")
    model_bytes = os.path.getsize(f"baseline_fp32_{device}.pt")
    x = torch.from_numpy(sample_np.astype(np.float32)).to(device)
    model.eval().to(device)
    with torch.no_grad():
        for _ in range(20): _=model(x)
        if device.startswith("cuda"): torch.cuda.synchronize()
        lat=[]
        peak = rss_before if proc else 0
        for _ in range(runs):
            t0=time.perf_counter(); _=model(x)
            if device.startswith("cuda"): torch.cuda.synchronize()
            t1=time.perf_counter()
            lat.append((t1-t0)*1000.0)
            if proc: peak = max(peak, psutil.Process(os.getpid()).memory_info().rss)
    stats = {
        "device": device, "runs": runs,
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_p50": float(np.percentile(lat,50)),
        "latency_ms_p95": float(np.percentile(lat,95)),
        "latency_ms_p99": float(np.percentile(lat,99)),
        "model_bytes": model_bytes,
    }
    if proc:
        stats.update({
            "rss_before_bytes": int(rss_before),
            "rss_peak_bytes": int(peak),
            "rss_increase_bytes": int(peak - rss_before),
        })
    with open(save_path,"w",encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("Benchmark:", stats)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time-col", default="timestamp")
    ap.add_argument("--seq-len", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--device", default="cpu")           # cpu|cuda
    ap.add_argument("--bench-runs", type=int, default=1000)
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(">> CUDA not available; falling back to CPU"); args.device="cpu"

    set_seed(11, deterministic=True, device=args.device)
    print(">> start"); print("cwd:", os.getcwd()); print("argv:", sys.argv); print("device:", args.device)

    df = load_csv(args.csv, args.time_col)
    feats = ["load_kW","price","pv_kW"]; tgt="net_load"
    X_all, y_all = make_supervised(df, feats, tgt, args.seq_len, args.horizon)

    ts=None
    if args.time_col in df.columns:
        ts_all = df[args.time_col].iloc[args.seq_len+args.horizon-1:].reset_index(drop=True).astype(str).values
        ts = ts_all[:len(y_all)]

    # split
    idx_tr, idx_va, idx_te = split_indices(len(X_all), args.val_ratio, args.test_ratio, seed=11)

    # standardize (train-only)
    mu_x, sd_x = fit_std_feats(X_all, idx_tr); Xn = apply_std_feats(X_all, mu_x, sd_x)
    mu_y, sd_y = fit_std_target(y_all, idx_tr); yn = (y_all - mu_y) / sd_y

    pack = lambda I: (Xn[I], yn[I], None if ts is None else ts[I])
    X_tr, y_tr, ts_tr = pack(idx_tr); X_va, y_va, ts_va = pack(idx_va); X_te, y_te, ts_te = pack(idx_te)

    train_loader = DataLoader(SeqDataset(X_tr, y_tr, ts_tr), batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(SeqDataset(X_va, y_va, ts_va), batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(SeqDataset(X_te, y_te, ts_te), batch_size=args.batch, shuffle=False, num_workers=0)

    # --- Ozan-style model ---
    model = OzanCNNLSTM(in_features=len(feats), hidden_size=64, num_layers=1)
    print(f"params: {count_params(model):,}")

    model = train(model, train_loader, val_loader, epochs=args.epochs, lr=1e-3, early=6, device=args.device)

    # eval (inverse scaling → kW)
    yt_n, yp_n, ts_out = evaluate(model, test_loader, device=args.device)
    yt = (yt_n * sd_y + mu_y).astype(np.float64); yp = (yp_n * sd_y + mu_y).astype(np.float64)
    err = yp - yt
    mae = float(np.mean(np.abs(err))); rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(yt), 1e-6))) * 100.0)
    metrics = {"MAE_kW": mae, "RMSE_kW": rmse, "MAPE_%": mape}
    print("test metrics (kW):", metrics)

    # bench
    bench_infer(model, Xn[:1], device=args.device, runs=args.bench_runs, save_path=f"bench_{args.device}.json")

    # save results
    payload = {
        "metrics_kW": metrics,
        "params": count_params(model),
        "seq_len": args.seq_len, "horizon": args.horizon, "epochs": args.epochs,
        "feature_mu": mu_x.tolist(), "feature_sd": sd_x.tolist(),
        "target_mu": mu_y, "target_sd": sd_y,
        "device": args.device,
        "model_type": "OzanCNNLSTM(Conv×3+BN+LSTM+FC)"
    }
    with open("results_github_baseline.json","w",encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # preds CSV
    rows=[]; 
    if ts_out is None: ts_out = np.array([None]*len(yt))
    for t, yti, ypi in zip(ts_out, yt, yp):
        rows.append({"timestamp": t, "y_true_kW": float(yti), "y_pred_kW": float(ypi)})
    pd.DataFrame(rows).to_csv("preds_github_baseline.csv", index=False)

    # scatter
    plt.figure(); plt.scatter(yt, yp, s=6)
    lo=float(min(yt.min(), yp.min())); hi=float(max(yt.max(), yp.max()))
    plt.plot([lo,hi],[lo,hi]); plt.xlabel("True (kW)"); plt.ylabel("Pred (kW)")
    plt.title("Ozan-style CNN-LSTM: True vs Pred"); plt.tight_layout()
    plt.savefig("scatter_github_baseline.png"); plt.close()

    print(f"Artifacts: results_github_baseline.json, bench_{args.device}.json, preds_github_baseline.csv, scatter_github_baseline.png")

if __name__ == "__main__":
    main()
