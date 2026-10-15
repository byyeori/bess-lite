"""Exact operation count, batch 1, T=48, three independent methods:
  (1) analytic per-layer formula (MACs of conv / linear / LSTM matmuls),
  (2) torch.utils.flop_counter.FlopCounterMode (counts matmul-type FLOPs = 2 x MACs),
  (3) thop (MACs, adds some elementwise terms, skips functional ops).
Elementwise ops (ReLU, gates, sin/cos, memory weights) are listed separately.
"""
import json, torch, torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from thop import profile
import main as M, base_model as B
from fast_model import FastMemoryModelConv

T, F, Hd = 48, 9, 32

def analytic_ours(last_only=False):
    rows = [
        ("Depthwise Conv1d k=5 (9 ch)", F * 5 * T),
        ("Pointwise Conv1d 9->32", F * Hd * T),
        ("Dilated Conv1d d=2 (32->32, k=3)", Hd * Hd * 3 * T),
        ("Dilated Conv1d d=4", Hd * Hd * 3 * T),
        ("Dilated Conv1d d=8", Hd * Hd * 3 * T),
        ("Memory aggregation (weighted sum)", Hd * T),
        ("Shared Linear 36->32" + (" (last step only)" if last_only else " (all 48 steps)"), (Hd + 4) * Hd * (1 if last_only else T)),
        ("3 heads: 32->16->1", 3 * (Hd * 16 + 16)),
    ]
    return rows

def analytic_base(in_dim):
    L = [T, T // 2, T // 4, T // 8]  # after each MaxPool(2)
    lstm = lambda i, h, n: 4 * (i * h + h * h) * n
    return [
        (f"Conv1d {in_dim}->64 k=3 (T=48)", in_dim * 64 * 3 * L[0]),
        ("Conv1d 64->128 k=3 (T=24)", 64 * 128 * 3 * L[1]),
        ("Conv1d 128->64 k=3 (T=12)", 128 * 64 * 3 * L[2]),
        ("LSTM 64->64 (6 steps)", lstm(64, 64, L[3])),
        ("LSTM 64->128 (6 steps)", lstm(64, 128, L[3])),
        ("LSTM 128->64 (6 steps)", lstm(128, 64, L[3])),
        ("FC 64->1024", 64 * 1024),
        ("FC 1024->3", 1024 * 3),
    ]

def fcm(model, args):
    with torch.no_grad(), FlopCounterMode(display=False) as c:
        model(*args)
    return c.get_total_flops(), {str(k): v for k, v in c.get_flop_counts()["Global"].items()}

torch.manual_seed(0)
ours = M.MemoryModel(9, pred_len=1).eval()
fast = FastMemoryModelConv(ours).eval()
base33 = B.CNNLSTMModel(33, 3).eval()
base9 = B.CNNLSTMModel(9, 3).eval()
x = torch.randn(1, T, 9); h = torch.randint(0, 24, (1, T)); d = torch.randint(0, 7, (1, T))
xb = torch.randn(1, T, 33)

out = {}
for name, model, args, rows in [
    ("ours (original code)", ours, (x, h, d), analytic_ours(False)),
    ("ours (inference-optimized, same outputs)", fast, (x, h, d), analytic_ours(True)),
    ("baseline CNN-LSTM, 33 inputs", base33, (xb,), analytic_base(33)),
    ("baseline CNN-LSTM, 9 inputs", base9, (x,), analytic_base(9)),
]:
    macs_a = sum(r[1] for r in rows)
    flops_f, detail = fcm(model, args)
    macs_t, _ = profile(model, inputs=args, verbose=False)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) if name.startswith("base") or "original" in name else sum(p.numel() for p in ours.parameters() if p.requires_grad)
    out[name] = dict(params=params, analytic_MACs=macs_a, analytic_FLOPs=2 * macs_a, FlopCounterMode_FLOPs=flops_f,
                     FlopCounterMode_ops=detail, thop_MACs=float(macs_t), layers=rows)
    print(f"\n== {name} ==  params {params:,}")
    for r in rows: print(f"   {r[0]:<48s} {r[1]:>10,} MACs")
    print(f"   analytic: {macs_a:,} MACs = {2*macs_a:,} FLOPs | FlopCounterMode: {flops_f:,} FLOPs {detail} | thop: {int(macs_t):,} MACs")
json.dump(out, open("results2/flops.json", "w"), indent=1, default=str)
