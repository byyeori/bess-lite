"""
Wind MSE Top1 모델 학습 스크립트 (main.py와 동일한 변인통제)
Trial ID: 20251123104743_4139
Config: hidden_dim=20, depthwise_kernel=9, dilated_layers=3, dilation=[1,2,4], ema_decay=0.75, head_dropout=0.05
Resources: FLOPs=209,238, Params=5,059
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import math, time
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from thop import profile, clever_format

# Hyperparameters (Wind Top1)
WINDOW = 48
HORIZON = 1
BATCH_SIZE = 128
EPOCHS = 50
BASE_LR = 0.001
MAX_LR = 0.003
SEED = 42

# Model architecture
HIDDEN_DIM = 20
DEPTHWISE_KERNEL = 9
DILATED_LAYERS = 3
DILATION_PATTERN = [1, 2, 4]
EMA_DECAY = 0.75
HEAD_DROPOUT = 0.05

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT_DIR / 'data' / 'data.csv'
CKPT_DIR = ROOT_DIR / 'ckpt'


def get_sinusoidal_encoding(hour, day):
    hour_rad = 2 * math.pi * hour / 24.0
    day_rad  = 2 * math.pi * day / 7.0
    return torch.stack([
        torch.sin(hour_rad), torch.cos(hour_rad),
        torch.sin(day_rad), torch.cos(day_rad)
    ], dim=-1)


class BESSDataset(Dataset):
    def __init__(self, df, seq_len=24, pred_len=24, scalers=None):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.weather_cols = ['DHI', 'DNI', 'GHI', 'Wind Speed', 'Temperature', 'Pressure']
        self.signal_cols = ['pv_kW', 'net_load', 'wind_kW']
        self.feature_cols = self.weather_cols + self.signal_cols

        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for dataset: {missing}")

        df.loc[:, self.feature_cols] = (
            df[self.feature_cols].interpolate().bfill().ffill()
        )

        if scalers is None:
            self.scaler_weather = StandardScaler()
            self.scaler_signal = MinMaxScaler()
            self.scaler_pv_y = MinMaxScaler()
            self.scaler_net_y = MinMaxScaler()
            self.scaler_wind_y = MinMaxScaler()

            weather_scaled = self.scaler_weather.fit_transform(df[self.weather_cols])
            signal_scaled = self.scaler_signal.fit_transform(df[self.signal_cols])
            pv_y_scaled = self.scaler_pv_y.fit_transform(df[['pv_kW']])
            net_y_scaled = self.scaler_net_y.fit_transform(df[['net_load']])
            wind_y_scaled = self.scaler_wind_y.fit_transform(df[['wind_kW']])
        else:
            (
                self.scaler_weather,
                self.scaler_signal,
                self.scaler_pv_y,
                self.scaler_net_y,
                self.scaler_wind_y,
            ) = scalers
            weather_scaled = self.scaler_weather.transform(df[self.weather_cols])
            signal_scaled = self.scaler_signal.transform(df[self.signal_cols])
            pv_y_scaled = self.scaler_pv_y.transform(df[['pv_kW']])
            net_y_scaled = self.scaler_net_y.transform(df[['net_load']])
            wind_y_scaled = self.scaler_wind_y.transform(df[['wind_kW']])

        self.features = np.concatenate([weather_scaled, signal_scaled], axis=1)
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


class MemoryModel(nn.Module):
    def __init__(self, input_features, pred_len=24, hidden_dim=32,
                 depthwise_kernel=5, dilations=[2, 4, 8], dropout=0.0): 
        super().__init__()
        self.pred_len = pred_len
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_features, input_features, kernel_size=depthwise_kernel, 
                     padding=depthwise_kernel//2, groups=input_features),
            nn.BatchNorm1d(input_features),
            nn.ReLU(),
            nn.Conv1d(input_features, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )
        blocks = []
        for d in dilations:
            blocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d))
            blocks.append(nn.ReLU())
        self.temporal_encoder = nn.Sequential(*blocks)
        
        self.memory_decay = nn.Parameter(torch.tensor(0.9), requires_grad=False)
        self.time_decoder = nn.Linear(hidden_dim + 4, hidden_dim)
        
        self.pv_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.net_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.wind_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
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
        pv_pred = self.pv_head(pred_region).squeeze(-1)
        net_pred = self.net_head(pred_region).squeeze(-1)
        wind_pred = self.wind_head(pred_region).squeeze(-1)
        return pv_pred, net_pred, wind_pred


criterion = nn.MSELoss()

def multitask_loss(pv_pred, net_pred, wind_pred, pv_true, net_true, wind_true):
    pred = torch.cat([pv_pred, net_pred, wind_pred], dim=-1)
    true = torch.cat([pv_true, net_true, wind_true], dim=-1)
    return criterion(pred, true)


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
    for batch in loader:
        x, hour_idx, day_idx = batch['x'].to(device), batch['hour_idx'].to(device), batch['day_idx'].to(device)
        pv_y = batch['pv_y'].to(device)
        net_y = batch['net_y'].to(device)
        wind_y = batch['wind_y'].to(device)
        optimizer.zero_grad()
        pv_pred, net_pred, wind_pred = model(x, hour_idx, day_idx)
        loss = multitask_loss(pv_pred, net_pred, wind_pred, pv_y, net_y, wind_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def measure_latency(model, dataset, device):
    if len(dataset) == 0:
        return None
    sample = dataset[len(dataset) - 1]
    x = sample['x'].unsqueeze(0).to(device)
    hour = sample['hour_idx'].unsqueeze(0).to(device)
    day = sample['day_idx'].unsqueeze(0).to(device)
    if x.shape[1] < DEPTHWISE_KERNEL:
        return None
    for _ in range(10):
        _ = model(x, hour, day)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x, hour, day)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    return (end - start) / 100 * 1000


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Wind MSE Top1 모델 학습 시작")
    print(f"Device: {device}")
    print(f"Config: hidden_dim={HIDDEN_DIM}, kernel={DEPTHWISE_KERNEL}, layers={DILATED_LAYERS}")
    print(f"        dilation={DILATION_PATTERN}, ema_decay={EMA_DECAY}, dropout={HEAD_DROPOUT}")
    
    # Load data
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

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
    
    # Split data (70% train, 15% val, 15% test)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    train_ds = BESSDataset(train_df, seq_len=WINDOW, pred_len=HORIZON)
    shared_scalers = (
        train_ds.scaler_weather,
        train_ds.scaler_signal,
        train_ds.scaler_pv_y,
        train_ds.scaler_net_y,
        train_ds.scaler_wind_y,
    )
    val_ds = BESSDataset(val_df, scalers=shared_scalers, seq_len=WINDOW, pred_len=HORIZON)
    test_ds = BESSDataset(test_df, scalers=shared_scalers, seq_len=WINDOW, pred_len=HORIZON)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"\n📊 Data split: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    
    # Create model
    input_dim = len(train_ds.feature_cols)
    model = MemoryModel(
        input_features=input_dim,
        pred_len=HORIZON,
        hidden_dim=HIDDEN_DIM,
        depthwise_kernel=DEPTHWISE_KERNEL,
        dilations=DILATION_PATTERN,
        dropout=HEAD_DROPOUT
    ).to(device)
    
    # Profile model
    dummy_x = torch.randn(1, WINDOW, input_dim).to(device)
    dummy_hour = torch.randint(0, 24, (1, WINDOW)).to(device)
    dummy_day = torch.randint(0, 7, (1, WINDOW)).to(device)
    
    class _ProfileWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, x, hour, day):
            return self.backbone(x, hour, day)[0]
    
    profiling_model = _ProfileWrapper(model)
    flops_raw, params_raw = profile(profiling_model, inputs=(dummy_x, dummy_hour, dummy_day), verbose=False)
    flops_str, params_str = clever_format([flops_raw, params_raw], "%.3f")
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=BASE_LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        total_steps=len(train_loader) * EPOCHS
    )
    
    # Training loop
    print(f"\n🏋️  Training for {EPOCHS} epochs...")
    best_val = float("inf")
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        pv_mse, pv_rmse, pv_mae, net_mse, net_rmse, net_mae, wind_mse, wind_rmse, wind_mae = evaluate(model, val_loader, device)
        val_metric = (pv_rmse + net_rmse + wind_rmse) / 3
        
        if val_metric < best_val:
            best_val = val_metric
            best_epoch = epoch
            torch.save(model.state_dict(), CKPT_DIR / 'wind_top1_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f"[{epoch+1:02d}/{EPOCHS}] Loss={train_loss:.4f} | Val Metric={val_metric:.4f}")
    
    print(f"\n✅ Best model saved at epoch {best_epoch+1}")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(CKPT_DIR / 'wind_top1_model.pth', map_location=device))
    pv_mse, pv_rmse, pv_mae, net_mse, net_rmse, net_mae, wind_mse, wind_rmse, wind_mae = evaluate(model, test_loader, device)
    print("\n===== Final Test Results =====")
    print(f"PV   -> MSE: {pv_mse:.4f}, RMSE: {pv_rmse:.4f}, MAE: {pv_mae:.4f}")
    print(f"Net  -> MSE: {net_mse:.4f}, RMSE: {net_rmse:.4f}, MAE: {net_mae:.4f}")
    print(f"Wind -> MSE: {wind_mse:.4f}, RMSE: {wind_rmse:.4f}, MAE: {wind_mae:.4f}")

    print("\n===== Model Complexity =====")
    print(f"FLOPs  : {flops_str}")
    print(f"Params : {params_str}")

    latency = measure_latency(model, train_ds, device)
    if latency is not None:
        print(f"\nInference Latency: {latency:.2f} ms ({'GPU' if device.type=='cuda' else 'CPU'})")
    else:
        print("\nInference Latency: N/A (insufficient sample length)")


if __name__ == '__main__':
    main()
