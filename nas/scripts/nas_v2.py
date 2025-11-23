import os, math, json, argparse, random, time
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from thop import profile
import yaml
try:
    import optuna
except ImportError:
    optuna = None

DATA_PATH = os.path.join('data', 'data.csv')
SPACE_PATH = os.path.join('nas', 'config', 'nas_space_v2.yaml')
RUNS_DIR = os.path.join('nas', 'runs', 'v2')
os.makedirs(RUNS_DIR, exist_ok=True)

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ---------------- Time Encoding ----------------
def get_sinusoidal_encoding(hour, day):
    hour_rad = 2 * math.pi * hour / 24.0
    day_rad  = 2 * math.pi * day / 7.0
    return torch.stack([
        torch.sin(hour_rad), torch.cos(hour_rad),
        torch.sin(day_rad),  torch.cos(day_rad)
    ], dim=-1)

# ---------------- Dataset ----------------
class BESSDataset(Dataset):
    weather_cols = ['Temperature','DHI','DNI','GHI','Pressure','Wind Speed']
    # Raw columns in data.csv are lowercase with _kw suffix: wind_kw, load_kw, pv_kw
    # We will rename them to wind_kW, load_kW, pv_kW for consistency, and create net_load = load_kW - pv_kW + wind_kW if absent.
    signal_cols = ['pv_kW','net_load','wind_kW']
    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int, scalers=None):
        self.seq_len = seq_len; self.pred_len = pred_len
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Rename raw generation/load columns to canonical names
        rename_map = {'wind_kw': 'wind_kW', 'load_kw': 'load_kW', 'pv_kw': 'pv_kW'}
        for k,v in rename_map.items():
            if k in df.columns:
                df.rename(columns={k: v}, inplace=True)
        # Construct net_load if not present (pv + wind - load pattern may differ; here use load_kW - pv_kW adjustment if domain specifies)
        if 'net_load' not in df.columns and all(c in df.columns for c in ['load_kW','pv_kW','wind_kW']):
            # Assuming net load definition similar to load minus generation (pv + wind)
            df['net_load'] = df['load_kW'] - (df['pv_kW'] + df['wind_kW'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        all_cols = self.weather_cols + self.signal_cols
        df[all_cols] = df[all_cols].interpolate().bfill().ffill()
        if scalers is None:
            self.scaler_weather = StandardScaler().fit(df[self.weather_cols])
            # Use separate scalers per target (avoid leakage between distributions)
            self.scaler_signal  = MinMaxScaler().fit(df[self.signal_cols])
            self.scaler_pv_y    = MinMaxScaler().fit(df[['pv_kW']])
            self.scaler_net_y   = MinMaxScaler().fit(df[['net_load']])
            self.scaler_wind_y  = MinMaxScaler().fit(df[['wind_kW']])
        else:
            (self.scaler_weather, self.scaler_signal,
             self.scaler_pv_y, self.scaler_net_y, self.scaler_wind_y) = scalers
        weather = self.scaler_weather.transform(df[self.weather_cols])
        signal  = self.scaler_signal.transform(df[self.signal_cols])
        self.features = np.concatenate([weather, signal], axis=1)
        self.pv_y   = self.scaler_pv_y.transform(df[['pv_kW']]).squeeze()
        self.net_y  = self.scaler_net_y.transform(df[['net_load']]).squeeze()
        self.wind_y = self.scaler_wind_y.transform(df[['wind_kW']]).squeeze()
        self.hours = df['hour'].values; self.days = df['day_of_week'].values
        self.feature_cols = self.weather_cols + self.signal_cols
    def __len__(self):
        return len(self.features) - self.seq_len - self.pred_len + 1
    def __getitem__(self, idx):
        s = idx; e = s + self.seq_len
        pred_s = e; pred_e = pred_s + self.pred_len
        x = torch.tensor(self.features[s:e], dtype=torch.float32)
        hour_idx = torch.tensor(self.hours[s:e], dtype=torch.long)
        day_idx  = torch.tensor(self.days[s:e], dtype=torch.long)
        pv = torch.tensor(self.pv_y[pred_s:pred_e], dtype=torch.float32)
        net= torch.tensor(self.net_y[pred_s:pred_e], dtype=torch.float32)
        wind= torch.tensor(self.wind_y[pred_s:pred_e], dtype=torch.float32)
        return {
            'x': x, 'hour_idx': hour_idx, 'day_idx': day_idx,
            'pv_y': pv, 'net_y': net, 'wind_y': wind
        }

# ---------------- Model ----------------
class MemoryModel(nn.Module):
    def __init__(self, input_features: int, hidden_dim: int, pred_len: int,
                 dilations: List[int], ema_decay: float,
                 head_dropout: float, depthwise_kernel: int):
        super().__init__()
        self.pred_len = pred_len
        # Always learnable EMA (baseline behavior); initial value varies via search (ema_decay)
        self.memory_decay = nn.Parameter(torch.tensor(ema_decay))
        pad = depthwise_kernel // 2
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(input_features, input_features, kernel_size=depthwise_kernel, padding=pad, groups=input_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_features, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        blocks = []
        for d in dilations:
            blocks.append(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=d, dilation=d))
            blocks.append(nn.ReLU(inplace=True))
        self.temporal_encoder = nn.Sequential(*blocks)
        self.time_decoder = nn.Linear(hidden_dim + 4, hidden_dim)
        def head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(p=head_dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        self.pv_head = head(); self.net_head = head(); self.wind_head = head()
    def forward(self, x, hour_idx, day_idx):
        B, T, F = x.shape
        time_emb = get_sinusoidal_encoding(hour_idx, day_idx).to(x.device)
        x = self.temporal_conv(x.transpose(1,2))
        x = self.temporal_encoder(x)
        T2 = x.shape[-1]
        weights = torch.pow(self.memory_decay.to(x.device), torch.arange(T2-1, -1, -1, device=x.device))
        weights = weights / weights.sum()
        memory = torch.sum(x * weights.view(1,1,-1), dim=2)
        x = x.transpose(1,2).contiguous()
        x[:, -1, :] = x[:, -1, :] + memory
        x = torch.cat([x, time_emb], dim=-1)
        x = self.time_decoder(x)
        pred_region = x[:, -self.pred_len:, :]
        pv   = self.pv_head(pred_region).squeeze(-1)
        net  = self.net_head(pred_region).squeeze(-1)
        wind = self.wind_head(pred_region).squeeze(-1)
        return pv, net, wind

# ---------------- Helpers ----------------

def load_space(path: str) -> Dict[str, Any]:
    with open(path,'r',encoding='utf-8') as f: return yaml.safe_load(f)

def sample_random(space: Dict[str, Any]) -> Dict[str, Any]:
    ss = space['search_space']; cfg = {}
    cfg['hidden_dim'] = random.choice(ss['hidden_dim']['choices'])
    cfg['depthwise_kernel'] = random.choice(ss['depthwise_kernel']['choices'])
    cfg['dilated_layers'] = random.choice(ss['dilated_layers']['choices'])
    dil = random.choice(ss['dilation_pattern']['choices'])
    cfg['dilation_pattern'] = dil[:cfg['dilated_layers']]
    cfg['ema_decay'] = random.choice(ss['ema_decay']['choices'])
    cfg['head_dropout'] = random.choice(ss['head_dropout']['choices'])
    return cfg

def build_model(input_dim: int, pred_len: int, trial_cfg: Dict[str, Any]):
    return MemoryModel(
        input_features=input_dim,
        hidden_dim=trial_cfg['hidden_dim'],
        pred_len=pred_len,
        dilations=trial_cfg['dilation_pattern'],
        ema_decay=trial_cfg['ema_decay'],
        head_dropout=trial_cfg['head_dropout'],
        depthwise_kernel=trial_cfg['depthwise_kernel']
    )

@torch.no_grad()
def profile_model(model: nn.Module):
    model.eval()
    dummy_x = torch.randn(1, 48, model.temporal_conv[0].in_channels).to(next(model.parameters()).device)
    dummy_h = torch.randint(0,24,(1,48)).to(next(model.parameters()).device)
    dummy_d = torch.randint(0,7,(1,48)).to(next(model.parameters()).device)
    try:
        flops, params = profile(model, inputs=(dummy_x, dummy_h, dummy_d), verbose=False)
    except Exception:
        # Fallback approximate: count conv MACs ~2* in/out*kernel*length aggregated
        flops = 0
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                Cin = m.in_channels; Cout = m.out_channels; K = m.kernel_size[0]; L = dummy_x.shape[1]
                flops += Cin * Cout * K * L
        params = sum(p.numel() for p in model.parameters())
    return flops, params

# ---------------- Training / Evaluation ----------------
criterion = nn.MSELoss()

def evaluate(model, loader, device):
    model.eval()
    pv_mse=[]; net_mse=[]; wind_mse=[]
    pv_rmse=[]; net_rmse=[]; wind_rmse=[]
    pv_mae=[]; net_mae=[]; wind_mae=[]
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device); h=batch['hour_idx'].to(device); d=batch['day_idx'].to(device)
            pv_y = batch['pv_y'].to(device); net_y=batch['net_y'].to(device); wind_y=batch['wind_y'].to(device)
            pv_p, net_p, wind_p = model(x, h, d)
            pv_err = pv_p - pv_y
            net_err = net_p - net_y
            wind_err = wind_p - wind_y
            pv_mse.append(torch.mean(pv_err**2).item())
            net_mse.append(torch.mean(net_err**2).item())
            wind_mse.append(torch.mean(wind_err**2).item())
            pv_rmse.append(torch.sqrt(torch.mean(pv_err**2)).item())
            net_rmse.append(torch.sqrt(torch.mean(net_err**2)).item())
            wind_rmse.append(torch.sqrt(torch.mean(wind_err**2)).item())
            pv_mae.append(torch.mean(torch.abs(pv_err)).item())
            net_mae.append(torch.mean(torch.abs(net_err)).item())
            wind_mae.append(torch.mean(torch.abs(wind_err)).item())
    return {
        'pv_mse': np.mean(pv_mse), 'net_mse': np.mean(net_mse), 'wind_mse': np.mean(wind_mse),
        'pv_rmse': np.mean(pv_rmse), 'net_rmse': np.mean(net_rmse), 'wind_rmse': np.mean(wind_rmse),
        'pv_mae': np.mean(pv_mae), 'net_mae': np.mean(net_mae), 'wind_mae': np.mean(wind_mae)
    }

def train_one(model, train_loader, val_loader, device, epochs, base_lr, max_lr):
    opt = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=len(train_loader)*epochs)
    best = math.inf; best_metrics=None
    for ep in range(epochs):
        model.train(); ep_losses=[]
        for batch in train_loader:
            x=batch['x'].to(device); h=batch['hour_idx'].to(device); d=batch['day_idx'].to(device)
            pv_y=batch['pv_y'].to(device); net_y=batch['net_y'].to(device); wind_y=batch['wind_y'].to(device)
            pv_p, net_p, wind_p = model(x,h,d)
            loss = criterion(torch.cat([pv_p, net_p, wind_p], dim=-1),
                             torch.cat([pv_y, net_y, wind_y], dim=-1))
            if torch.isnan(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); scheduler.step()
            ep_losses.append(loss.item())
        val_m = evaluate(model, val_loader, device)
        # Wind-slightly-emphasized MSE score: 0.3*pv + 0.3*net + 0.4*wind
        score = 0.3*val_m['pv_mse'] + 0.3*val_m['net_mse'] + 0.4*val_m['wind_mse']
        if score < best:
            best = score; best_metrics = val_m.copy()
    return best_metrics

# ---------------- NAS Loop ----------------

def run_trial(idx: int, space: Dict[str, Any], df: pd.DataFrame, device: torch.device):
    cfg = sample_random(space)
    seq_len = space['fixed']['seq_len']; pred_len = space['fixed']['pred_len']
    # Match baseline main.py: train 70%, val 15%, test 15% (here we use train/val only)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    train_ds = BESSDataset(train_df, seq_len, pred_len)
    val_ds = BESSDataset(val_df, seq_len, pred_len, scalers=(train_ds.scaler_weather, train_ds.scaler_signal,
                                                             train_ds.scaler_pv_y, train_ds.scaler_net_y, train_ds.scaler_wind_y))
    train_loader = DataLoader(train_ds, batch_size=space['fixed']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=space['fixed']['batch_size'], shuffle=False)
    model = build_model(len(train_ds.feature_cols), pred_len, cfg).to(device)
    flops, params = profile_model(model)
    if flops > space['constraints']['max_flops'] or params > space['constraints']['max_params']:
        return {'rejected': True, 'reason': 'resource_limit', 'config': cfg, 'flops': flops, 'params': params}
    metrics = train_one(model, train_loader, val_loader, device,
                        epochs=space['fixed']['epochs'], base_lr=space['fixed']['base_lr'], max_lr=space['fixed']['max_lr'])
    metrics.update({'flops': flops, 'params': params})
    trial_id = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
    out_dir = os.path.join(RUNS_DIR, trial_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'config.json'),'w') as f: json.dump(cfg,f,indent=2)
    with open(os.path.join(out_dir,'metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
    return {'rejected': False, 'config': cfg, 'metrics': metrics, 'id': trial_id}

def run_tpe_trial(trial: 'optuna.Trial', space: Dict[str, Any], df: pd.DataFrame, device: torch.device, trial_offset: int):
    """Optuna objective function for TPE phase."""
    ss = space['search_space']
    cfg = {}
    cfg['hidden_dim'] = trial.suggest_categorical('hidden_dim', ss['hidden_dim']['choices'])
    cfg['depthwise_kernel'] = trial.suggest_categorical('depthwise_kernel', ss['depthwise_kernel']['choices'])
    cfg['dilated_layers'] = trial.suggest_categorical('dilated_layers', ss['dilated_layers']['choices'])
    dil_idx = trial.suggest_categorical('dilation_pattern_idx', list(range(len(ss['dilation_pattern']['choices']))))
    dil = ss['dilation_pattern']['choices'][dil_idx]
    cfg['dilation_pattern'] = dil[:cfg['dilated_layers']]
    cfg['ema_decay'] = trial.suggest_categorical('ema_decay', ss['ema_decay']['choices'])
    cfg['head_dropout'] = trial.suggest_categorical('head_dropout', ss['head_dropout']['choices'])
    
    seq_len = space['fixed']['seq_len']; pred_len = space['fixed']['pred_len']
    n = len(df)
    train_end = int(n * 0.7); val_end = int(n * 0.85)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    train_ds = BESSDataset(train_df, seq_len, pred_len)
    val_ds = BESSDataset(val_df, seq_len, pred_len, scalers=(train_ds.scaler_weather, train_ds.scaler_signal,
                                                             train_ds.scaler_pv_y, train_ds.scaler_net_y, train_ds.scaler_wind_y))
    train_loader = DataLoader(train_ds, batch_size=space['fixed']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=space['fixed']['batch_size'], shuffle=False)
    model = build_model(len(train_ds.feature_cols), pred_len, cfg).to(device)
    flops, params = profile_model(model)
    if flops > space['constraints']['max_flops'] or params > space['constraints']['max_params']:
        print(f"[TPE Trial {trial.number + trial_offset}] REJECTED (resource_limit) FLOPs={flops:.0f} Params={params:.0f}")
        return 999.0  # penalty score for rejected
    
    metrics = train_one(model, train_loader, val_loader, device,
                        epochs=space['fixed']['epochs'], base_lr=space['fixed']['base_lr'], max_lr=space['fixed']['max_lr'])
    score = 0.3*metrics['pv_mse'] + 0.3*metrics['net_mse'] + 0.4*metrics['wind_mse']
    
    trial_id = f"{time.strftime('%Y%m%d%H%M%S')}_{random.randint(1000,9999)}"
    out_dir = os.path.join(RUNS_DIR, trial_id)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,'config.json'),'w') as f: json.dump(cfg,f,indent=2)
    metrics.update({'flops': flops, 'params': params})
    with open(os.path.join(out_dir,'metrics.json'),'w') as f: json.dump(metrics,f,indent=2)
    
    print(f"[TPE Trial {trial.number + trial_offset}] id={trial_id} score={score:.6f} pv_mse={metrics['pv_mse']:.6f} net_mse={metrics['net_mse']:.6f} wind_mse={metrics['wind_mse']:.6f} flops={flops:.0f} params={params:.0f} cfg={cfg}")
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--random-trials', type=int, default=200)
    ap.add_argument('--tpe-trials', type=int, default=200)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)
    space = load_space(SPACE_PATH)
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Phase 1: Random Search
    print(f"\n=== Phase 1: Random Search ({args.random_trials} trials) ===")
    results=[]
    for i in range(args.random_trials):
        r = run_trial(i, space, df, device)
        if r['rejected']:
            print(f"[Trial {i}] REJECTED ({r['reason']}) FLOPs={r['flops']:.0f} Params={r['params']:.0f}")
        else:
            m = r['metrics']; cfg=r['config']
            score = 0.3*m['pv_mse'] + 0.3*m['net_mse'] + 0.4*m['wind_mse']
            print(f"[Trial {i}] id={r['id']} score={score:.6f} pv_mse={m['pv_mse']:.6f} net_mse={m['net_mse']:.6f} wind_mse={m['wind_mse']:.6f} flops={m['flops']:.0f} params={m['params']:.0f} cfg={cfg}")
            results.append({'id': r['id'], 'score': score, **m, **cfg})
    
    # Phase 2: TPE Search
    if args.tpe_trials > 0 and optuna is not None:
        print(f"\n=== Phase 2: TPE Search ({args.tpe_trials} trials) ===")
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
        # Enqueue random phase results as initial observations
        dilation_choices = space['search_space']['dilation_pattern']['choices']
        for r in results:
            # Random phase stored truncated pattern (prefix up to dilated_layers).
            # Find first choice whose prefix matches the stored truncated pattern.
            target_prefix = r['dilation_pattern']
            dlayers = r['dilated_layers']
            match_idx = None
            for i, full in enumerate(dilation_choices):
                if full[:dlayers] == target_prefix:
                    match_idx = i
                    break
            if match_idx is None:
                # Fallback: use index 0
                match_idx = 0
            cfg_dict = {
                'hidden_dim': r['hidden_dim'],
                'depthwise_kernel': r['depthwise_kernel'],
                'dilated_layers': dlayers,
                'dilation_pattern_idx': match_idx,
                'ema_decay': r['ema_decay'],
                'head_dropout': r['head_dropout']
            }
            study.enqueue_trial(cfg_dict, skip_if_exists=True)
        
        study.optimize(lambda trial: run_tpe_trial(trial, space, df, device, args.random_trials), 
                      n_trials=args.tpe_trials)
        
        # Collect TPE results
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.value < 900:
                trial_dirs = sorted([d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))], 
                                   key=lambda x: os.path.getmtime(os.path.join(RUNS_DIR, x)))
                if trial_dirs:
                    latest_dir = trial_dirs[-1]
                    cfg_path = os.path.join(RUNS_DIR, latest_dir, 'config.json')
                    metrics_path = os.path.join(RUNS_DIR, latest_dir, 'metrics.json')
                    if os.path.exists(cfg_path) and os.path.exists(metrics_path):
                        with open(cfg_path) as f: cfg = json.load(f)
                        with open(metrics_path) as f: m = json.load(f)
                        if not any(r.get('id') == latest_dir for r in results):
                            results.append({'id': latest_dir, 'score': trial.value, **m, **cfg})
    
    if results:
        results.sort(key=lambda x: x['score'])
        with open(os.path.join(RUNS_DIR,'summary.json'),'w') as f: json.dump(results, f, indent=2)
        best=results[0]
        print('\n=== Best Trial Summary ===')
        print(json.dumps(best, indent=2))
        print(f"\nTotal valid trials: {len(results)}")

if __name__=='__main__':
    main()
