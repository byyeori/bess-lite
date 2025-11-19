import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math, time, joblib
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from thop import profile, clever_format

BASE_DIR = "data"
FEATURE_CSV = "urop_data.csv"
DATA_PATH = os.path.join(BASE_DIR, FEATURE_CSV)
EPOCHS = 30
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

def add_rolling_lag_features(df, window_sizes=[6, 12, 24], lags=[1, 2, 24, 168]):
    for w in window_sizes:
        df[f"roll_mean_{w}h"] = df["target"].rolling(w).mean()
        df[f"roll_std_{w}h"]  = df["target"].rolling(w).std()
    for l in lags:
        df[f"lag_{l}h"] = df["target"].shift(l)
    df = df.bfill().ffill()
    df = df.fillna(0)
    return df




class MemoryModel(nn.Module):
    def __init__(self, input_features, hidden_dim=32): 
        super().__init__()
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
        pv_pred = self.pv_head(x).squeeze(-1)
        return pv_pred



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
        self.pv_feats = [c for c in df.columns if c.startswith('pv_') and ('roll_' in c or 'lag_' in c)]
        self.feature_cols = self.weather_cols + self.pv_feats
        
        df[self.feature_cols] = df[self.feature_cols].interpolate().bfill().ffill()

        if scalers is None:
            self.scaler_weather = RobustScaler()
            self.scaler_pv_feats = StandardScaler()
            self.scaler_pv_y = MinMaxScaler()

            weather_scaled = self.scaler_weather.fit_transform(df[self.weather_cols])
            pv_scaled = self.scaler_pv_feats.fit_transform(df[self.pv_feats])
            pv_y_scaled = self.scaler_pv_y.fit_transform(df[['pv_kW']])

        else:
            self.scaler_weather, self.scaler_pv_feats, self.scaler_pv_y = scalers
            weather_scaled = self.scaler_weather.transform(df[self.weather_cols])
            pv_scaled = self.scaler_pv_feats.transform(df[self.pv_feats])
            pv_y_scaled = self.scaler_pv_y.transform(df[['pv_kW']])

        self.features = np.concatenate([weather_scaled, pv_scaled], axis=1)
        self.pv_y = pv_y_scaled.squeeze()
        self.hours = df['hour'].values
        self.days = df['day_of_week'].values

    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx+self.seq_len]
        hour_idx = self.hours[idx:idx+self.seq_len]
        day_idx = self.days[idx:idx+self.seq_len]
        pv_y = self.pv_y[idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return {
            'x': torch.FloatTensor(x),
            'hour_idx': torch.LongTensor(hour_idx),
            'day_idx': torch.LongTensor(day_idx),
            'pv_y': torch.FloatTensor(pv_y)
        }


def pv_loss_fn(pred, true):
    return torch.mean(torch.abs(pred - true))

def evaluate(model, loader, device):
    model.eval()
    pv_mae, pv_rmse, pv_mape = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, hour, day, pv_y = batch['x'].to(device), batch['hour_idx'].to(device), batch['day_idx'].to(device), batch['pv_y'].to(device)
            pv_pred = model(x, hour, day)
            pv_mae.append(torch.mean(torch.abs(pv_pred - pv_y)).item())
            pv_rmse.append(torch.sqrt(torch.mean((pv_pred - pv_y)**2)).item())
            pv_mape.append(torch.mean(torch.abs((pv_pred - pv_y) / (pv_y.abs() + 1e-6)) * 100).item())
    return np.mean(pv_mae), np.mean(pv_rmse), np.mean(pv_mape)



def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    valid_batches = 0
    for batch in loader:
        x, hour_idx, day_idx = batch['x'].to(device), batch['hour_idx'].to(device), batch['day_idx'].to(device)
        pv_y = batch['pv_y'].to(device)
        optimizer.zero_grad()
        pv_pred = model(x, hour_idx, day_idx)
        loss = pv_loss_fn(pv_pred, pv_y)
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
    dataset = BESSDataset(df_sample, scalers=scalers)
    sample = dataset[-1]
    x = sample['x'].unsqueeze(0).to(device)
    hour = sample['hour_idx'].unsqueeze(0).to(device)
    day = sample['day_idx'].unsqueeze(0).to(device)
    
    # warm-up
    for _ in range(10):
        _ = model(x, hour, day)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    
    start = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(x, hour, day)
    torch.cuda.synchronize() if device.type == 'cuda' else None
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
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    df_pv = add_rolling_lag_features(df.copy().assign(target=df["pv_kW"]))
    for c in [col for col in df_pv.columns if "roll_" in col or "lag_" in col]:
        df[f"pv_{c}"] = df_pv[c]


    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_df, val_df, test_df = df[:train_end], df[train_end:val_end], df[val_end:]
    
    train_ds = BESSDataset(train_df)
    val_ds = BESSDataset(val_df, scalers=(train_ds.scaler_weather, train_ds.scaler_pv_feats, train_ds.scaler_pv_y))
    test_ds = BESSDataset(test_df, scalers=(train_ds.scaler_weather, train_ds.scaler_pv_feats, train_ds.scaler_pv_y))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = len(train_ds.feature_cols)
    model = MemoryModel(input_features=input_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=len(train_loader) * EPOCHS
    )

    best_val = float("inf")
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        pv_mae, pv_rmse, pv_mape = evaluate(model, val_loader, device)
        val_metric = pv_rmse

        print(f"[{epoch+1:02d}/{EPOCHS}] "
              f"Loss={train_loss:.4f} | RMSE={pv_rmse:.4f}, MAE={pv_mae:.4f}, MAPE={pv_mape:.2f}%")

        if val_metric < best_val:
            best_val = val_metric
            torch.save(model.state_dict(), "best_memory_model.pth")

    model.load_state_dict(torch.load("best_memory_model.pth"))
    pv_mae, pv_rmse, pv_mape = evaluate(model, test_loader, device)
    print("\n===== Final Test Results =====")
    print(f"PV RMSE: {pv_rmse:.4f}, MAE: {pv_mae:.4f}, MAPE: {pv_mape:.2f}%")

    dummy_x = torch.randn(1, 24, input_dim).to(device)
    dummy_hour = torch.randint(0, 24, (1, 24)).to(device)
    dummy_day = torch.randint(0, 7, (1, 24)).to(device)
    flops, params = profile(model, inputs=(dummy_x, dummy_hour, dummy_day), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print("\n===== Model Complexity =====")
    print(f"FLOPs  : {flops}")
    print(f"Params : {params}")
    latency = measure_latency(model, (train_ds.scaler_weather, train_ds.scaler_pv_feats, train_ds.scaler_pv_y), df.tail(300), device)
    print(f"\nInference Latency: {latency:.2f} ms ({'GPU' if device.type=='cuda' else 'CPU'})")

