# -*- coding: utf-8 -*-
"""
Student FP32 (DWConv + LSTM32) baseline runner
- CPU only
- Train μ/σ standardization (features & target)
- kW metrics (MAE, RMSE, MAPE)
- Simple latency/model-size bench (batch=1)
"""
import os, sys, time, json, argparse, gc
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

try:
    import psutil
except Exception:
    psutil = None

# -------- Utils --------
def set_seed(seed=11):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

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
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
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
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=need)
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
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"Split invalid: N={n}")
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def fit_std_feats(X, idx):
    Xt = X[idx].reshape(len(idx), -1, X.shape[-1])  # [Ntr,T,F]
    mu = Xt.mean(axis=(0,1))
    sd = Xt.std(axis=(0,1)); sd[sd==0] = 1.0
    return mu, sd

def apply_std_feats(X, mu, sd):
    return (X - mu.reshape(1,1,-1)) / sd.reshape(1,1,-1)

def fit_std_target(y, idx):
    mu = y[idx].mean(); sd = y[idx].std()
    if sd == 0: sd = 1.0
    return float(mu), float(sd)

# -------- Model: DWConv + LSTM(32) --------
class DWConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, padding=1):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, k, stride=stride, padding=padding, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
    def forward(self, x):           # [B,Cin,T]
        return F.relu(self.bn(self.pw(self.dw(x))))

class LightCNNLSTM(nn.Module):
    def __init__(self, in_features, lstm_hidden=32):
        super().__init__()
        self.block1 = DWConv1d(in_features, 24, k=3, padding=1)  # 3 -> 24
        self.block2 = DWConv1d(24, 24, k=3, padding=1)           # 24 -> 24
        self.lstm   = nn.LSTM(input_size=24, hidden_size=lstm_hidden, batch_first=True)
        self.fc     = nn.Linear(lstm_hidden, 1)

    def forward(self, x):           # x: [B,T,F]
        x = x.transpose(1,2)        # [B,F,T]
        x = self.block1(x)          # [B,24,T]
        x = self.block2(x)          # [B,24,T]
        x = x.transpose(1,2)        # [B,T,24]
        out,_ = self.lstm(x)        # [B,T,32]
        return self.fc(out[:,-1,:]) # [B,1]

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# -------- Train / Eval --------
def train(model, trL, vaL, epochs=20, lr=1e-3, early=6, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best = float("inf"); keep=None; patience=early
    for ep in range(1, epochs+1):
        model.train(); run=0.0; nobs=0
        for batch in trL:
            xb, yb = batch[:2]
            xb=xb.to(device); yb=yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item()*xb.size(0); nobs += xb.size(0)
        tr = run/max(nobs,1)

        model.eval(); v=0.0; nobs=0
        with torch.no_grad():
            for batch in vaL:
                xb, yb = batch[:2]
                xb=xb.to(device); yb=yb.to(device)
                l = F.mse_loss(model(xb), yb)
                v += l.item()*xb.size(0); nobs += xb.size(0)
        vl = v/max(nobs,1)
        print(f"epoch {ep:02d}  trainMSE {tr:.6f}  valMSE {vl:.6f}")

        if vl < best - 1e-9:
            best = vl
            keep = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            patience = early
        else:
            patience -= 1
            if patience <= 0:
                print("early stopping."); break
    if keep: model.load_state_dict(keep)
    return model

def evaluate(model, loader, device="cpu"):
    model.eval(); yps=[]; yts=[]; tss=[]
    with torch.no_grad():
        for batch in loader:
            if len(batch)==2: xb,yb = batch; ts=None
            else: xb,yb,ts = batch
            yp = model(xb.to(device)).detach().cpu()
            yps.append(yp); yts.append(yb); tss.append(ts)
    yp = torch.cat(yps,0).numpy().flatten()
    yt = torch.cat(yts,0).numpy().flatten()
    ts = None
    if all(t is not None for t in tss): ts = np.concatenate(tss,0)
    return yt, yp, ts

# -------- Bench --------
def bench_infer(model, sample_np, runs=1000, save_path="bench_student_fp32.json"):
    proc = psutil.Process(os.getpid()) if psutil else None
    if proc: gc.collect(); rss_before = proc.memory_info().rss
    torch.save(model.state_dict(), "student_fp32.pt")
    model_bytes = os.path.getsize("student_fp32.pt")

    x = torch.from_numpy(sample_np.astype(np.float32))
    model.eval()
    with torch.no_grad():
        for _ in range(20): _ = model(x)   # warmup
        lat=[]
        peak = rss_before if proc else 0
        for _ in range(runs):
            t0=time.perf_counter(); _=model(x); t1=time.perf_counter()
            lat.append((t1-t0)*1000.0)
            if proc:
                peak = max(peak, proc.memory_info().rss)
    stats = {
        "runs": runs,
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

# -------- Main --------
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
    ap.add_argument("--bench-runs", type=int, default=1000)
    args = ap.parse_args()

    # CPU 강제
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    set_seed(11)
    print(">> start (Student FP32, CPU-only)")
    print("cwd:", os.getcwd()); print("argv:", sys.argv)

    # 1) Data
    df = load_csv(args.csv, args.time_col)
    feats = ["load_kW","price","pv_kW"]; tgt = "net_load"
    X_all, y_all = make_supervised(df, feats, tgt, args.seq_len, args.horizon)

    ts=None
    if args.time_col in df.columns:
        ts_all = df[args.time_col].iloc[args.seq_len+args.horizon-1:].reset_index(drop=True).astype(str).values
        ts = ts_all[:len(y_all)]

    # 2) Split
    idx_tr, idx_va, idx_te = split_indices(len(X_all), args.val_ratio, args.test_ratio, seed=11)

    # 3) Standardization (train-only)
    mu_x, sd_x = fit_std_feats(X_all, idx_tr); Xn = apply_std_feats(X_all, mu_x, sd_x)
    mu_y, sd_y = fit_std_target(y_all, idx_tr); yn = (y_all - mu_y) / sd_y

    pack = lambda I: (Xn[I], yn[I], None if ts is None else ts[I])
    X_tr, y_tr, ts_tr = pack(idx_tr); X_va, y_va, ts_va = pack(idx_va); X_te, y_te, ts_te = pack(idx_te)

    # 4) Loaders
    trL = DataLoader(SeqDataset(X_tr, y_tr, ts_tr), batch_size=args.batch, shuffle=True,  num_workers=0)
    vaL = DataLoader(SeqDataset(X_va, y_va, ts_va), batch_size=args.batch, shuffle=False, num_workers=0)
    teL = DataLoader(SeqDataset(X_te, y_te, ts_te), batch_size=args.batch, shuffle=False, num_workers=0)

    # 5) Model
    model = LightCNNLSTM(in_features=len(feats), lstm_hidden=32)
    print(f"params: {count_params(model):,}")

    # 6) Train
    model = train(model, trL, vaL, epochs=args.epochs, lr=1e-3, early=6, device="cpu")

    # 7) Evaluate (kW)
    yt_n, yp_n, ts_out = evaluate(model, teL, device="cpu")
    yt = (yt_n * sd_y + mu_y).astype(np.float64)
    yp = (yp_n * sd_y + mu_y).astype(np.float64)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = float(np.mean(np.abs(err / np.maximum(np.abs(yt), 1e-6))) * 100.0)
    metrics = {"MAE_kW": mae, "RMSE_kW": rmse, "MAPE_%": mape}
    print("student test metrics (kW):", metrics)

    # 8) Bench (batch=1, normalized input)
    bench_infer(model, Xn[:1], runs=args.bench_runs, save_path="bench_student_fp32.json")

    # 9) Save results
    payload = {
        "metrics_kW": metrics,
        "params": count_params(model),
        "seq_len": args.seq_len, "horizon": args.horizon, "epochs": args.epochs,
        "feature_mu": mu_x.tolist(), "feature_sd": sd_x.tolist(),
        "target_mu": mu_y, "target_sd": sd_y,
        "device": "cpu",
        "model_type": "Student_FP32(DWConv+LSTM32)"
    }
    with open("results_student_fp32.json","w",encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 10) Pred CSV
    rows=[]
    if ts_out is None: ts_out = np.array([None]*len(yt))
    for t, yti, ypi in zip(ts_out, yt, yp):
        rows.append({"timestamp": t, "y_true_kW": float(yti), "y_pred_kW": float(ypi)})
    pd.DataFrame(rows).to_csv("preds_student_fp32.csv", index=False)

    # 11) Scatter
    plt.figure(); plt.scatter(yt, yp, s=6)
    lo=float(min(yt.min(), yp.min())); hi=float(max(yt.max(), yp.max()))
    plt.plot([lo,hi],[lo,hi]); plt.xlabel("True (kW)"); plt.ylabel("Pred (kW)")
    plt.title("Student FP32 (DWConv + LSTM32): True vs Pred")
    plt.tight_layout(); plt.savefig("scatter_student_fp32.png"); plt.close()

    print("Artifacts: results_student_fp32.json, bench_student_fp32.json, preds_student_fp32.csv, scatter_student_fp32.png, student_fp32.pt")

if __name__ == "__main__":
    main()
