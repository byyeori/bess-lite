import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math, time, joblib
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from thop import profile, clever_format

BASE_DIR = "data"
FEATURE_CSV = "data.csv"
DATA_PATH = os.path.join(BASE_DIR, FEATURE_CSV)
EPOCHS = 50
WINDOW = 48
HORIZON = 1

torch.autograd.set_detect_anomaly(False)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def get_sinusoidal_encoding(hour, day):
    hour_rad = 2 * math.pi * hour / 24.0
    day_rad  = 2 * math.pi * day / 7.0
    return torch.stack([
        torch.sin(hour_rad), torch.cos(hour_rad),
        torch.sin(day_rad), torch.cos(day_rad)
    ], dim=-1)

def add_rolling_lag_features(df, target_col, prefix, window_sizes=[6, 12, 24], lags=[1, 2, 24, 168]):
    """Return rolling and lag features for a given signal."""
    features = pd.DataFrame(index=df.index)
    target = df[target_col]
    for w in window_sizes:
        features[f"{prefix}roll_mean_{w}h"] = target.rolling(w).mean()
        features[f"{prefix}roll_std_{w}h"] = target.rolling(w).std()
    for l in lags:
        features[f"{prefix}lag_{l}h"] = target.shift(l)
    return features.bfill().ffill().fillna(0)




class MemoryModel(nn.Module):
    def __init__(self, input_features, pred_len=24, hidden_dim=32): 
        super().__init__()
        self.pred_len = pred_len
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_features, input_features, kernel_size=5, padding=2, groups=input_features),
            nn.BatchNorm1d(input_features),
            nn.ReLU(),
            nn.Conv1d(input_features, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=8, dilation=8),
            nn.ReLU(),
        )
        self.memory_decay = nn.Parameter(torch.tensor(0.9), requires_grad=False)
        self.time_decoder = nn.Linear(hidden_dim + 4, hidden_dim)
        self.pv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.net_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.wind_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )


    def forward(self, x, hour_idx, day_idx):
        B, T, F = x.shape
        time_emb = get_sinusoidal_encoding(hour_idx, day_idx).to(x.device)
        x = self.temporal_conv(x.transpose(1, 2))
        x = self.temporal_encoder(x)
        weights = torch.pow(self.memory_decay, torch.arange(T - 1, -1, -1, device=x.device))
        weights = weights / weights.sum()
        memory = torch.sum(x * weights.view(1, 1, -1), dim=2)
        x = x.transpose(1, 2).contiguous()
        x[:, -1, :] = x[:, -1, :] + memory
        x = torch.cat([x, time_emb], dim=-1)
        x = self.time_decoder(x)
        pred_region = x[:, -self.pred_len:, :]
        # pv_pred = self.pv_head(x).squeeze(-1)
        # net_pred = self.net_head(x).squeeze(-1)
        pv_pred = self.pv_head(pred_region).squeeze(-1)
        net_pred = self.net_head(pred_region).squeeze(-1)
        wind_pred = self.wind_head(pred_region).squeeze(-1)
        return pv_pred, net_pred, wind_pred



# def weighted_multitask_loss(pv_pred, load_pred, pv_true, load_true, daylight_mask):
#     pv_loss = torch.mean(torch.abs(pv_pred - pv_true) * daylight_mask)
#     peak_mask = (daylight_mask == 0).float() * 0.3 + 1.0
#     load_loss = torch.mean(torch.abs(load_pred - load_true) * peak_mask)
#     pv_loss = torch.nan_to_num(pv_loss, nan=0.0, posinf=10.0, neginf=-10.0)
#     load_loss = torch.nan_to_num(load_loss, nan=0.0, posinf=10.0, neginf=-10.0)

#     return pv_loss + 0.5 * load_loss



class BESSDataset(Dataset):
    def __init__(self, df, seq_len=24, pred_len=24, scalers=None):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.weather_cols = ['DHI', 'DNI', 'GHI', 'Wind Speed', 'Temperature', 'Pressure']
        self.signal_cols = ['pv_kW', 'net_load', 'wind_kW']
        history_prefixes = ['pv_', 'net_', 'wind_']
        self.history_cols = sorted([
            c for c in df.columns
            if any(
                c.startswith(f"{prefix}roll_") or c.startswith(f"{prefix}lag_")
                for prefix in history_prefixes
            )
        ])
        self.feature_cols = self.weather_cols + self.signal_cols + self.history_cols

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for dataset: {missing}")

        # df[self.feature_cols] = df[self.feature_cols].interpolate().bfill().ffill()
        df.loc[:, self.feature_cols] = (
            df[self.feature_cols].interpolate().bfill().ffill()
        )

        if scalers is None:
            self.scaler_weather = StandardScaler()
            self.scaler_signal = MinMaxScaler()
            self.scaler_history = StandardScaler()
            self.scaler_pv_y = MinMaxScaler()
            self.scaler_net_y = MinMaxScaler()
            self.scaler_wind_y = MinMaxScaler()

            weather_scaled = self.scaler_weather.fit_transform(df[self.weather_cols])
            signal_scaled = self.scaler_signal.fit_transform(df[self.signal_cols])
            history_scaled = self.scaler_history.fit_transform(df[self.history_cols])
            pv_y_scaled = self.scaler_pv_y.fit_transform(df[['pv_kW']])
            net_y_scaled = self.scaler_net_y.fit_transform(df[['net_load']])
            wind_y_scaled = self.scaler_wind_y.fit_transform(df[['wind_kW']])


        else:
            (
                self.scaler_weather,
                self.scaler_signal,
                self.scaler_history,
                self.scaler_pv_y,
                self.scaler_net_y,
                self.scaler_wind_y,
            ) = scalers
            weather_scaled = self.scaler_weather.transform(df[self.weather_cols])
            signal_scaled = self.scaler_signal.transform(df[self.signal_cols])
            history_scaled = self.scaler_history.transform(df[self.history_cols])
            pv_y_scaled = self.scaler_pv_y.transform(df[['pv_kW']])
            net_y_scaled = self.scaler_net_y.transform(df[['net_load']])
            wind_y_scaled = self.scaler_wind_y.transform(df[['wind_kW']])


        self.features = np.concatenate([weather_scaled, signal_scaled, history_scaled], axis=1)
        self.pv_y = pv_y_scaled.squeeze()
        self.net_y = net_y_scaled.squeeze()
        self.wind_y = wind_y_scaled.squeeze()
        self.hours = df['hour'].values
        self.days = df['day_of_week'].values

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        hour_idx = self.hours[idx:idx+self.seq_len]
        day_idx = self.days[idx:idx+self.seq_len]
        pv_y = self.pv_y[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        net_y = self.net_y[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        wind_y = self.wind_y[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return {
            'x': torch.FloatTensor(x),
            'hour_idx': torch.LongTensor(hour_idx),
            'day_idx': torch.LongTensor(day_idx),
            'pv_y': torch.FloatTensor(pv_y),
            'net_y': torch.FloatTensor(net_y),
            'wind_y': torch.FloatTensor(wind_y)
        }


def multitask_loss(pv_pred, net_pred, wind_pred, pv_true, net_true, wind_true):
    pv_loss = torch.mean(torch.abs(pv_pred - pv_true))
    net_loss = torch.mean(torch.abs(net_pred - net_true))
    wind_loss = torch.mean(torch.abs(wind_pred - wind_true))
    return pv_loss + net_loss + wind_loss

def evaluate(model, loader, device):
    model.eval()
    pv_mse, pv_rmse, pv_mae = [], [], []
    net_mse, net_rmse, net_mae = [], [], []
    wind_mse, wind_rmse, wind_mae = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            hour = batch['hour_idx'].to(device)
            day = batch['day_idx'].to(device)
            pv_y = batch['pv_y'].to(device)
            net_y = batch['net_y'].to(device)
            wind_y = batch['wind_y'].to(device)
            pv_pred, net_pred, wind_pred = model(x, hour, day)
            pv_err = pv_pred - pv_y
            net_err = net_pred - net_y
            wind_err = wind_pred - wind_y
            pv_mse.append(torch.mean(pv_err ** 2).item())
            pv_rmse.append(torch.sqrt(torch.mean(pv_err ** 2)).item())
            pv_mae.append(torch.mean(torch.abs(pv_err)).item())
            net_mse.append(torch.mean(net_err ** 2).item())
            net_rmse.append(torch.sqrt(torch.mean(net_err ** 2)).item())
            net_mae.append(torch.mean(torch.abs(net_err)).item())
            wind_mse.append(torch.mean(wind_err ** 2).item())
            wind_rmse.append(torch.sqrt(torch.mean(wind_err ** 2)).item())
            wind_mae.append(torch.mean(torch.abs(wind_err)).item())
    return (
        np.mean(pv_mse), np.mean(pv_rmse), np.mean(pv_mae),
        np.mean(net_mse), np.mean(net_rmse), np.mean(net_mae),
        np.mean(wind_mse), np.mean(wind_rmse), np.mean(wind_mae),
    )



def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    for batch in loader:
        x, hour_idx, day_idx = batch['x'].to(device), batch['hour_idx'].to(device), batch['day_idx'].to(device)
        pv_y = batch['pv_y'].to(device)
        net_y = batch['net_y'].to(device)
        wind_y = batch['wind_y'].to(device)
        optimizer.zero_grad()
        pv_pred, net_pred, wind_pred = model(x, hour_idx, day_idx)
        loss = multitask_loss(pv_pred, net_pred, wind_pred, pv_y, net_y, wind_y)
        if torch.isnan(loss):
            print("⚠️ NaN detected in loss. Skipping batch.")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        valid_batches += 1
    return total_loss / max(1, valid_batches)


def measure_latency(model, scalers, df_sample, device):
    lat_ds = BESSDataset(df_sample, seq_len=24, pred_len=24, scalers=scalers)
    sample = lat_ds[len(lat_ds) - 1]
    x = sample['x'].unsqueeze(0).to(device)
    hour = sample['hour_idx'].unsqueeze(0).to(device)
    day = sample['day_idx'].unsqueeze(0).to(device)

    # Conv1D(kernel=5) 때문에 길이 검사 필요
    if x.shape[1] < 5:
        print(f"❌ Latency input too short: T={x.shape[1]}. Skipping latency measurement.")
        return None
    for _ in range(10):
        _ = model(x, hour, day)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(x, hour, day)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    return (end - start) / 100 * 1000


def set_seed(seed=11):
    import random, torch, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

if __name__ == "__main__":  
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data file not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df.rename(columns={
        "pv_kw": "pv_kW",
        "load_kw": "net_load",
        "wind_kw": "wind_kW"
    }, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    required_signals = ["pv_kW", "net_load", "wind_kW"]
    missing_signals = [c for c in required_signals if c not in df.columns]
    if missing_signals:
        raise ValueError(f"Missing required columns: {missing_signals}")

    history_specs = [
        ("pv_kW", "pv_"),
        ("net_load", "net_"),
        ("wind_kW", "wind_"),
    ]
    history_frames = [add_rolling_lag_features(df, col, prefix=prefix) for col, prefix in history_specs]
    df = pd.concat([df] + history_frames, axis=1)


    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    # train_df, val_df, test_df = df[:train_end], df[train_end:val_end], df[val_end:]


    train_df = df[:train_end].copy()
    val_df   = df[train_end:val_end].copy()
    test_df  = df[val_end:].copy()

    
    train_ds = BESSDataset(train_df, seq_len=WINDOW, pred_len=HORIZON)
    shared_scalers = (
        train_ds.scaler_weather,
        train_ds.scaler_signal,
        train_ds.scaler_history,
        train_ds.scaler_pv_y,
        train_ds.scaler_net_y,
        train_ds.scaler_wind_y,
    )

    val_ds = BESSDataset(val_df, scalers=shared_scalers, seq_len=WINDOW, pred_len=HORIZON)
    test_ds = BESSDataset(test_df, scalers=shared_scalers, seq_len=WINDOW, pred_len=HORIZON)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = len(train_ds.feature_cols)
    model = MemoryModel(input_features=input_dim, pred_len=train_ds.pred_len).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=len(train_loader) * EPOCHS
    )

    best_val = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        pv_mse, pv_rmse, pv_mae, net_mse, net_rmse, net_mae, wind_mse, wind_rmse, wind_mae = evaluate(model, val_loader, device)
        val_metric = (pv_rmse + net_rmse + wind_rmse) / 3

        print(f"[{epoch+1:02d}/{EPOCHS}] Loss={train_loss:.4f} | "
              f"PV (MSE {pv_mse:.4f}, RMSE {pv_rmse:.4f}, MAE {pv_mae:.4f}) | "
              f"Net (MSE {net_mse:.4f}, RMSE {net_rmse:.4f}, MAE {net_mae:.4f}) | "
              f"Wind (MSE {wind_mse:.4f}, RMSE {wind_rmse:.4f}, MAE {wind_mae:.4f})")

        if val_metric < best_val:
            best_val = val_metric
            torch.save(model.state_dict(), "best_memory_model.pth")

    model.load_state_dict(torch.load("best_memory_model.pth"))
    pv_mse, pv_rmse, pv_mae, net_mse, net_rmse, net_mae, wind_mse, wind_rmse, wind_mae = evaluate(model, test_loader, device)
    print("\n===== Final Test Results =====")
    print(f"PV   -> MSE: {pv_mse:.4f}, RMSE: {pv_rmse:.4f}, MAE: {pv_mae:.4f}")
    print(f"Net  -> MSE: {net_mse:.4f}, RMSE: {net_rmse:.4f}, MAE: {net_mae:.4f}")
    print(f"Wind -> MSE: {wind_mse:.4f}, RMSE: {wind_rmse:.4f}, MAE: {wind_mae:.4f}")

    dummy_x = torch.randn(1, 24, input_dim).to(device)
    dummy_hour = torch.randint(0, 24, (1, 24)).to(device)
    dummy_day = torch.randint(0, 7, (1, 24)).to(device)
    class _PVProfileWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x, hour, day):
            return self.backbone(x, hour, day)[0]

    profiling_model = _PVProfileWrapper(model)
    flops, params = profile(profiling_model, inputs=(dummy_x, dummy_hour, dummy_day), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print("\n===== Model Complexity =====")
    print(f"FLOPs  : {flops}")
    print(f"Params : {params}")
    # latency = measure_latency(model, shared_scalers, df.tail(1000), device)
    latency = measure_latency(model, shared_scalers, df, device)
    print(f"\nInference Latency: {latency:.2f} ms ({'GPU' if device.type=='cuda' else 'CPU'})")
