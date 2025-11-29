import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import time
import math
import random

# FLOPs 계산 라이브러리
try:
    from thop import profile, clever_format
except ImportError:
    profile = None
    clever_format = None

# --- Feature Engineering Functions ---
def add_rolling_lag_features(df, target_col, prefix, window_sizes=[6, 12, 24], lags=[1, 24]):
    features = pd.DataFrame(index=df.index)
    target = df[target_col]
    for w in window_sizes:
        features[f"{prefix}roll_mean_{w}h"] = target.rolling(w).mean()
        features[f"{prefix}roll_std_{w}h"] = target.rolling(w).std()
    for l in lags:
        features[f"{prefix}lag_{l}h"] = target.shift(l)
    return features.bfill().ffill().fillna(0)

# --- Dataset Class ---
class MultitaskDataset(Dataset):
    def __init__(self, df, seq_len=24, pred_horizons=[1, 2, 3], scalers=None, target_cols=['pv_kw', 'load_kw', 'wind_kw']):
        self.seq_len = seq_len
        self.pred_horizons = pred_horizons 
        self.max_horizon = max(pred_horizons)
        self.target_cols = target_cols
        
        # --- [수정] Weather Columns 제거 (Feature Set 제거) ---
        # self.weather_cols = ['DHI', 'DNI', 'GHI', 'Wind Speed', 'Temperature', 'Pressure']
        
        self.history_cols = []
        for col in df.columns:
            if 'roll_' in col or 'lag_' in col:
                self.history_cols.append(col)
        
        # Use cols에서 Weather 제외
        use_cols = self.target_cols + self.history_cols
        existing_cols = [c for c in use_cols if c in df.columns]
        df[existing_cols] = df[existing_cols].interpolate().bfill().ffill()
        
        if scalers is None:
            # --- [수정] Scaler Weather 제거 ---
            # self.scaler_weather = StandardScaler()
            self.scaler_signal = MinMaxScaler() # PV, Load, Wind (Signal)
            self.scaler_history = StandardScaler()
            
            # Target Scalers (개별 적용)
            self.scaler_pv_y = MinMaxScaler()
            self.scaler_net_y = MinMaxScaler() 
            self.scaler_wind_y = MinMaxScaler()
            
            # Fitting
            # weather_scaled = self.scaler_weather.fit_transform(df[self.weather_cols])
            
            # Signal Scaling
            signal_scaled = self.scaler_signal.fit_transform(df[self.target_cols])
            history_scaled = self.scaler_history.fit_transform(df[self.history_cols])
            
            # Target Scalers Fitting
            pv_scaled = self.scaler_pv_y.fit_transform(df[['pv_kw']])
            net_scaled = self.scaler_net_y.fit_transform(df[['load_kw']])
            wind_scaled = self.scaler_wind_y.fit_transform(df[['wind_kw']])
            
            targets_scaled = np.concatenate([pv_scaled, net_scaled, wind_scaled], axis=1)

        else:
            # Load Scalers (Weather 제외)
            (self.scaler_signal, self.scaler_history, 
             self.scaler_pv_y, self.scaler_net_y, self.scaler_wind_y) = scalers
            
            # weather_scaled = self.scaler_weather.transform(df[self.weather_cols])
            signal_scaled = self.scaler_signal.transform(df[self.target_cols])
            history_scaled = self.scaler_history.transform(df[self.history_cols])
            
            pv_scaled = self.scaler_pv_y.transform(df[['pv_kw']])
            net_scaled = self.scaler_net_y.transform(df[['load_kw']])
            wind_scaled = self.scaler_wind_y.transform(df[['wind_kw']])
            
            targets_scaled = np.concatenate([pv_scaled, net_scaled, wind_scaled], axis=1)
            
        # Input Features 구성: Signal + History (Weather 제외)
        self.features = np.concatenate([signal_scaled, history_scaled], axis=1)
        self.targets = targets_scaled
        
    def __len__(self):
        return len(self.features) - self.seq_len - self.max_horizon
    
    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]
        y_list = []
        for horizon in self.pred_horizons:
            target_idx = idx + self.seq_len + horizon - 1
            y_list.append(self.targets[target_idx])
        y = np.array(y_list)
        y = y.flatten() 
        return torch.FloatTensor(x), torch.FloatTensor(y)

# --- Model Architecture ---
class CNNLSTMModel(nn.Module):
    def __init__(self, input_features, output_dim):
        super(CNNLSTMModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_features, out_channels=64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True)
        
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(64, 1024)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(1024, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.relu_fc(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Utilities ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def measure_latency(model, input_sample, device):
    model.eval()
    for _ in range(10): 
        with torch.no_grad(): _ = model(input_sample)
    if device.type == 'cuda': torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        with torch.no_grad(): _ = model(input_sample)
    if device.type == 'cuda': torch.cuda.synchronize()
    end = time.time()
    return (end - start) / 100 * 1000

# --- Main ---
if __name__ == "__main__":
    set_seed(42)
    
    # Settings
    CSV_FILE_PATH = 'data/urop_data_final.csv'
    TARGET_COLS = ['pv_kw', 'load_kw', 'wind_kw']
    
    TIME_STEPS = 48
    PRED_HORIZONS = [1] 
    HORIZON_NAMES = ["1h"]
    
    BATCH_SIZE = 128
    EPOCHS = 50 # --- [수정] Epochs 50으로 설정 ---
    LEARNING_RATE = 0.001
    
    # --- [수정] Lags 설정 (1, 2, 24, 168) ---
    LAGS = [1, 2, 24, 168]
    
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ 데이터 파일 없음: {CSV_FILE_PATH}")
        exit()

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except:
        df = pd.read_csv(CSV_FILE_PATH, encoding='cp949')

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.drop(columns=['timestamp'])

    history_frames = []
    prefix_map = {'pv_kw': 'pv_', 'load_kw': 'net_', 'wind_kw': 'wind_'}
    for col in TARGET_COLS:
        if col not in df.columns: raise ValueError(f"필수 컬럼 누락: {col}")
        prefix = prefix_map.get(col, f"{col}_")
        # --- [수정] lags 파라미터 적용 ---
        feat_df = add_rolling_lag_features(df, col, prefix, lags=LAGS)
        history_frames.append(feat_df)
    df = pd.concat([df] + history_frames, axis=1)

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df[:train_end].copy()
    val_df = df[train_end:val_end].copy()
    test_df = df[val_end:].copy()
    
    # Dataset 생성 (Scaler 5개 전달 - Weather 제외)
    train_ds = MultitaskDataset(train_df, seq_len=TIME_STEPS, pred_horizons=PRED_HORIZONS, target_cols=TARGET_COLS)
    scalers = (train_ds.scaler_signal, train_ds.scaler_history,
               train_ds.scaler_pv_y, train_ds.scaler_net_y, train_ds.scaler_wind_y)
    
    val_ds = MultitaskDataset(val_df, seq_len=TIME_STEPS, pred_horizons=PRED_HORIZONS, scalers=scalers, target_cols=TARGET_COLS)
    test_ds = MultitaskDataset(test_df, seq_len=TIME_STEPS, pred_horizons=PRED_HORIZONS, scalers=scalers, target_cols=TARGET_COLS)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    sample_x, _ = train_ds[0]
    INPUT_DIM = sample_x.shape[-1]
    
    NUM_TARGETS = len(TARGET_COLS)
    NUM_HORIZONS = len(PRED_HORIZONS)
    OUTPUT_DIM = NUM_TARGETS * NUM_HORIZONS
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input Dim: {INPUT_DIM}, Output Dim: {OUTPUT_DIM}")
    
    model = CNNLSTMModel(INPUT_DIM, OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    print("\n모델 학습 시작...")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"[{epoch+1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("\n===== Final Test Results (Normalized 0~1) =====")
    model.eval()
    
    errors_dict = {t: {h: [] for h in HORIZON_NAMES} for t in ['PV', 'Net', 'Wind']}
    target_map = {0: 'PV', 1: 'Net', 2: 'Wind'}
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x) 
            
            pred = pred.view(-1, NUM_HORIZONS, NUM_TARGETS)
            y = y.view(-1, NUM_HORIZONS, NUM_TARGETS)
            
            diff = pred - y
            
            for h_idx, h_name in enumerate(HORIZON_NAMES):
                for t_idx in range(NUM_TARGETS):
                    t_name = target_map[t_idx]
                    errors_dict[t_name][h_name].append(diff[:, h_idx, t_idx])

    for t_name in ['PV', 'Net', 'Wind']:
        print(f"\n--- Target: {t_name} ---")
        for h_name in HORIZON_NAMES:
            all_diff = torch.cat(errors_dict[t_name][h_name], dim=0)
            mse = torch.mean(all_diff ** 2).item()
            rmse = math.sqrt(mse)
            mae = torch.mean(torch.abs(all_diff)).item()
            print(f"  [{h_name}] MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

    print("\n===== Model Complexity =====")
    dummy_input = torch.randn(1, TIME_STEPS, INPUT_DIM).to(device)
    if profile:
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops, params = clever_format([flops, params], "%.3f")
        print(f"FLOPs  : {flops}")
        print(f"Params : {params}")
    else:
        print("thop library not installed.")
        
    latency = measure_latency(model, dummy_input, device)
    dev_str = 'GPU' if device.type == 'cuda' else 'CPU'
    print(f"\nInference Latency: {latency:.2f} ms ({dev_str})")