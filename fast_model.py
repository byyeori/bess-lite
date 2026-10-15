"""Inference-only rewrite of MemoryModel (main.py). Same weights, same outputs, fewer/cheaper ops.

Changes (all mathematically equivalent in eval mode):
  1. BatchNorm folded into the depthwise conv.
  2. Dilated conv (k=3) as one matmul over 3 shifted copies -> avoids PyTorch's slow_conv_dilated path.
  3. Memory weights precomputed once (buffer) instead of pow/arange/sum every call.
  4. time_decoder and time encoding computed only for the last step (the only step the heads use).
  5. Three heads fused: one Linear(32->48) + one block-diagonal Linear(48->3).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class FastMemoryModel(nn.Module):
    def __init__(self, m, T=48):
        super().__init__()
        dw, bn, pw = m.temporal_conv[0], m.temporal_conv[1], m.temporal_conv[3]
        s = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        self.dw_w = nn.Parameter((dw.weight * s[:, None, None]).detach(), requires_grad=False)
        self.dw_b = nn.Parameter(((dw.bias - bn.running_mean) * s + bn.bias).detach(), requires_grad=False)
        self.C = dw.in_channels
        self.pw = pw
        self.dil = []
        for conv in [m.temporal_encoder[0], m.temporal_encoder[2], m.temporal_encoder[4]]:
            W = conv.weight.detach()                      # (out, in, 3)
            lin = nn.Linear(W.shape[1] * 3, W.shape[0])
            lin.weight.data = W.permute(0, 2, 1).reshape(W.shape[0], -1).clone()   # [k0 | k1 | k2]
            lin.bias.data = conv.bias.detach().clone()
            self.dil.append((lin, conv.dilation[0]))
        self.dil_layers = nn.ModuleList([l for l, _ in self.dil])
        self.dils = [d for _, d in self.dil]
        lam = float(m.memory_decay)
        w = torch.tensor([lam ** (T - 1 - i) for i in range(T)])
        self.register_buffer("mem_w", (w / w.sum()).view(1, T, 1))
        self.time_decoder = m.time_decoder
        heads = [m.pv_head, m.net_head, m.wind_head]
        H1 = heads[0][0].out_features
        self.h1 = nn.Linear(heads[0][0].in_features, 3 * H1)
        self.h1.weight.data = torch.cat([h[0].weight for h in heads]).detach().clone()
        self.h1.bias.data = torch.cat([h[0].bias for h in heads]).detach().clone()
        self.h2 = nn.Linear(3 * H1, 3)
        W2 = torch.zeros(3, 3 * H1)
        for i, h in enumerate(heads):
            W2[i, i * H1:(i + 1) * H1] = h[2].weight.detach()[0]
        self.h2.weight.data = W2
        self.h2.bias.data = torch.cat([h[2].bias for h in heads]).detach().clone()

    def forward(self, x, hour_idx, day_idx):
        # x: (B, T, F)
        z = F.conv1d(x.transpose(1, 2), self.dw_w, self.dw_b, padding=2, groups=self.C).relu()
        z = self.pw(z).relu().transpose(1, 2)                       # (B, T, 32)
        for lin, d in zip(self.dil_layers, self.dils):
            p = F.pad(z, (0, 0, d, d))                               # pad time axis
            T = z.shape[1]
            z = lin(torch.cat([p[:, 0:T], p[:, d:d + T], p[:, 2 * d:2 * d + T]], dim=-1)).relu()
        last = z[:, -1] + (z * self.mem_w).sum(1)                   # memory added to last step
        h = hour_idx[:, -1].float() * (2 * math.pi / 24.0)
        dd = day_idx[:, -1].float() * (2 * math.pi / 7.0)
        te = torch.stack([h.sin(), h.cos(), dd.sin(), dd.cos()], -1)
        y = self.h2(self.h1(self.time_decoder(torch.cat([last, te], -1))).relu())
        return y[:, 0:1], y[:, 1:2], y[:, 2:3]


class FastMemoryModelConv(FastMemoryModel):
    """Same as FastMemoryModel but keeps the dilated convs as Conv1d (better for ONNX Runtime,
    which has an optimized dilated Conv kernel)."""
    def __init__(self, m, T=48):
        super().__init__(m, T)
        self.enc = m.temporal_encoder

    def forward(self, x, hour_idx, day_idx):
        z = F.conv1d(x.transpose(1, 2), self.dw_w, self.dw_b, padding=2, groups=self.C).relu()
        z = self.enc(self.pw(z).relu())                              # (B, 32, T)
        last = z[:, :, -1] + (z * self.mem_w.transpose(1, 2)).sum(2)
        h = hour_idx[:, -1].float() * (2 * math.pi / 24.0)
        dd = day_idx[:, -1].float() * (2 * math.pi / 7.0)
        te = torch.stack([h.sin(), h.cos(), dd.sin(), dd.cos()], -1)
        y = self.h2(self.h1(self.time_decoder(torch.cat([last, te], -1))).relu())
        return y[:, 0:1], y[:, 1:2], y[:, 2:3]
