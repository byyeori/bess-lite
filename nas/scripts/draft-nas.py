import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from thop import profile
from torch.utils.data import DataLoader, Dataset

try:
    import optuna
except ImportError:  # pragma: no cover
    optuna = None

DATA_PATH = os.path.join("data", "urop_data_final.csv")
DEFAULT_CFG = os.path.join("nas", "nas_space.yaml")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_rolling_lag_features(df: pd.DataFrame, target_col: str, prefix: str,
                             windows: List[int], lags: List[int]) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    target = df[target_col]
    for w in windows:
        feats[f"{prefix}roll_mean_{w}h"] = target.rolling(w).mean()
        feats[f"{prefix}roll_std_{w}h"] = target.rolling(w).std()
    for l in lags:
        feats[f"{prefix}lag_{l}h"] = target.shift(l)
    return feats.bfill().ffill().fillna(0)


def prepare_dataframe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    df = df.rename(columns={"wind_kw": "wind_kW", "load_kw": "load_kW", "pv_kw": "pv_kW"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    history_specs = [("wind_kW", "wind_"), ("pv_kW", "pv_"), ("load_kW", "load_")]
    history_frames = [add_rolling_lag_features(df, col, prefix, windows=[6, 12, 24], lags=[1, 24])
                      for col, prefix in history_specs]
    df = pd.concat([df] + history_frames, axis=1)
    df = df.ffill().bfill()
    return df


def split_dataframe(df: pd.DataFrame, split_cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_span = [pd.Timestamp(ts.strip()) for ts in split_cfg["train"].split("->")]
    val_span = [pd.Timestamp(ts.strip()) for ts in split_cfg["val"].split("->")]
    train_start, train_end = train_span
    val_start, val_end = val_span
    train_mask = (df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)
    val_mask = (df["timestamp"] >= val_start) & (df["timestamp"] <= val_end)
    train_df = df.loc[train_mask].reset_index(drop=True)
    val_df = df.loc[val_mask].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Train/Val split produced empty dataframe. Check data_split in YAML.")
    return train_df, val_df


class BESSDataset(Dataset):
    weather_cols = ['DHI', 'DNI', 'GHI', 'Wind Speed', 'Temperature', 'Pressure']
    signal_cols = ['wind_kW', 'pv_kW', 'load_kW']

    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int,
                 scalers: Tuple = None, target_stats: Dict[str, Any] = None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        history_prefixes = ['wind_', 'pv_', 'load_']
        self.history_cols = sorted([
            c for c in df.columns
            if any(c.startswith(f"{prefix}roll_") or c.startswith(f"{prefix}lag_")
                   for prefix in history_prefixes)
        ])
        self.feature_cols = self.weather_cols + self.signal_cols + self.history_cols
        df = df.copy()
        df[self.feature_cols] = df[self.feature_cols].interpolate().bfill().ffill()

        if scalers is None:
            self.scaler_weather = StandardScaler().fit(df[self.weather_cols])
            self.scaler_signal = MinMaxScaler().fit(df[self.signal_cols])
            self.scaler_history = StandardScaler().fit(df[self.history_cols]) if self.history_cols else None
            self.scaler_wind_y = MinMaxScaler().fit(df[['wind_kW']])
            self.scaler_pv_y = MinMaxScaler().fit(df[['pv_kW']])
            self.scaler_load_y = MinMaxScaler().fit(df[['load_kW']])
        else:
            (
                self.scaler_weather,
                self.scaler_signal,
                self.scaler_history,
                self.scaler_wind_y,
                self.scaler_pv_y,
                self.scaler_load_y,
            ) = scalers

        weather = self.scaler_weather.transform(df[self.weather_cols])
        signal = self.scaler_signal.transform(df[self.signal_cols])
        history = self.scaler_history.transform(df[self.history_cols]) if self.history_cols else np.zeros((len(df), 0))
        self.features = np.concatenate([weather, signal, history], axis=1)
        self.wind_y = self.scaler_wind_y.transform(df[['wind_kW']]).squeeze()
        self.pv_y = self.scaler_pv_y.transform(df[['pv_kW']]).squeeze()
        self.load_y = self.scaler_load_y.transform(df[['load_kW']]).squeeze()
        self.hours = df['hour'].values
        self.days = df['day_of_week'].values

        if target_stats is None:
            self.target_stats = {
                'wind': self._build_target_stat(self.scaler_wind_y),
                'pv': self._build_target_stat(self.scaler_pv_y),
                'load': self._build_target_stat(self.scaler_load_y),
            }
        else:
            self.target_stats = target_stats

    @staticmethod
    def _build_target_stat(scaler: MinMaxScaler) -> Dict[str, float]:
        data_min = float(scaler.data_min_[0])
        data_max = float(scaler.data_max_[0])
        scale = max(data_max - data_min, 1e-6)
        return {'min': data_min, 'range': scale}

    def __len__(self) -> int:
        return max(0, len(self.features) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        x = self.features[idx:idx + self.seq_len]
        hour_idx = self.hours[idx:idx + self.seq_len]
        day_idx = self.days[idx:idx + self.seq_len]
        wind = self.wind_y[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        pv = self.pv_y[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        load = self.load_y[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return {
            'x': torch.from_numpy(x).float(),
            'hour_idx': torch.from_numpy(hour_idx).long(),
            'day_idx': torch.from_numpy(day_idx).long(),
            'wind_y': torch.from_numpy(wind).float(),
            'pv_y': torch.from_numpy(pv).float(),
            'load_y': torch.from_numpy(load).float(),
        }

    def get_scalers(self) -> Tuple:
        return (
            self.scaler_weather,
            self.scaler_signal,
            self.scaler_history,
            self.scaler_wind_y,
            self.scaler_pv_y,
            self.scaler_load_y,
        )


def sinusoidal_time_encoding(hour_idx: torch.Tensor, day_idx: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("sinusoidal_dim must be divisible by 4")
    harmonics = dim // 4
    hour = hour_idx.float() / 24.0
    day = day_idx.float() / 7.0
    emb_parts = []
    for k in range(1, harmonics + 1):
        hour_angle = 2 * math.pi * k * hour
        day_angle = 2 * math.pi * k * day
        emb_parts.append(torch.sin(hour_angle))
        emb_parts.append(torch.cos(hour_angle))
        emb_parts.append(torch.sin(day_angle))
        emb_parts.append(torch.cos(day_angle))
    return torch.stack(emb_parts, dim=-1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: int):
        super().__init__()
        reduced = max(1, channels // ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, reduced, 1),
            nn.ReLU(),
            nn.Conv1d(reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(self.pool(x))
        return x * scale


class DilatedResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int,
                 dropout: float, se_ratio: int, use_residual: bool):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size,
                                   padding=padding, dilation=dilation,
                                   groups=hidden_dim)
        self.pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = use_residual
        self.se = SEBlock(hidden_dim, se_ratio) if se_ratio > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn(out)
        if self.se is not None:
            out = self.se(out)
        out = self.act(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + x
        return out


class MultiHeadMemoryModel(nn.Module):
    def __init__(self, input_dim: int, pred_len: int, cfg: Dict[str, Any]):
        super().__init__()
        hidden_dim = cfg['hidden_dim']
        kernel_size = cfg['kernel_size']
        block_count = cfg['dilated_blocks']
        dilations = cfg['dilation_pattern']
        if len(dilations) != block_count:
            raise ValueError("dilation_pattern length must equal dilated_blocks")
        se_mode = cfg['se_block']
        if se_mode not in {'none', 'squeeze_ratio_8', 'squeeze_ratio_4'}:
            raise ValueError(f"Unknown se_block option: {se_mode}")
        se_ratio = 0
        if se_mode == 'squeeze_ratio_8':
            se_ratio = 8
        elif se_mode == 'squeeze_ratio_4':
            se_ratio = 4

        residual_mode = cfg['residual']
        if residual_mode not in {'all_blocks', 'last_block', 'off'}:
            raise ValueError(f"Unknown residual option: {residual_mode}")
        dropout = cfg['dropout']
        time_dim = cfg['sinusoidal_dim']

        self.pred_len = pred_len
        self.time_dim = time_dim
        self.memory_decay = nn.Parameter(torch.tensor(cfg['ema_decay']), requires_grad=cfg['learnable_ema'])

        self.initial = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2, groups=input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Conv1d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
        )

        blocks = []
        for idx in range(block_count):
            use_residual = False
            if residual_mode == 'all_blocks':
                use_residual = True
            elif residual_mode == 'last_block' and idx == block_count - 1:
                use_residual = True
            blocks.append(
                DilatedResidualBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilations[idx],
                    dropout=dropout,
                    se_ratio=se_ratio,
                    use_residual=use_residual,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.time_decoder = nn.Linear(hidden_dim + time_dim, hidden_dim)
        head_hidden = max(4, hidden_dim // 2)
        self.wind_head = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.pv_head = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
        self.load_head = nn.Sequential(nn.Linear(hidden_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x: torch.Tensor, hour_idx: torch.Tensor, day_idx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        B, T, F = x.shape
        z = self.initial(x.transpose(1, 2))
        for block in self.blocks:
            z = block(z)
        decay = torch.clamp(self.memory_decay, 0.0, 0.9999)
        weights = torch.pow(decay, torch.arange(z.shape[-1] - 1, -1, -1, device=z.device))
        weights = weights / (weights.sum() + 1e-6)
        memory = torch.sum(z * weights.view(1, 1, -1), dim=2)
        z = z.transpose(1, 2).contiguous()
        z[:, -1, :] = z[:, -1, :] + memory
        time_emb = sinusoidal_time_encoding(hour_idx, day_idx, self.time_dim).to(z.device)
        z = torch.cat([z, time_emb], dim=-1)
        z = self.time_decoder(z)
        z = z[:, -self.pred_len:, :]
        wind = self.wind_head(z).squeeze(-1)
        pv = self.pv_head(z).squeeze(-1)
        load = self.load_head(z).squeeze(-1)
        return wind, pv, load


def denorm_from_stats(values: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
    return values * stats['range'] + stats['min']


def multitask_loss(outputs, targets, weights, normalized, stats):
    wind_pred, pv_pred, load_pred = outputs
    wind_true, pv_true, load_true = targets
    if normalized:
        wind_pred = denorm_from_stats(wind_pred, stats['wind'])
        wind_true = denorm_from_stats(wind_true, stats['wind'])
        pv_pred = denorm_from_stats(pv_pred, stats['pv'])
        pv_true = denorm_from_stats(pv_true, stats['pv'])
        load_pred = denorm_from_stats(load_pred, stats['load'])
        load_true = denorm_from_stats(load_true, stats['load'])
    wind_loss = torch.mean(torch.abs(wind_pred - wind_true))
    pv_loss = torch.mean(torch.abs(pv_pred - pv_true))
    load_loss = torch.mean(torch.abs(load_pred - load_true))
    return weights[0] * wind_loss + weights[1] * pv_loss + weights[2] * load_loss


def evaluate(model: nn.Module, loader: DataLoader, stats: Dict[str, Dict[str, float]], device: torch.device) -> Dict[str, float]:
    model.eval()
    metrics = {k: [] for k in ["wind_mae", "pv_mae", "load_mae", "wind_rmse", "pv_rmse", "load_rmse"]}
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            hour = batch['hour_idx'].to(device)
            day = batch['day_idx'].to(device)
            wind_y = batch['wind_y'].to(device)
            pv_y = batch['pv_y'].to(device)
            load_y = batch['load_y'].to(device)
            wind_pred, pv_pred, load_pred = model(x, hour, day)
            wind_pred = denorm_from_stats(wind_pred, stats['wind'])
            pv_pred = denorm_from_stats(pv_pred, stats['pv'])
            load_pred = denorm_from_stats(load_pred, stats['load'])
            wind_true = denorm_from_stats(wind_y, stats['wind'])
            pv_true = denorm_from_stats(pv_y, stats['pv'])
            load_true = denorm_from_stats(load_y, stats['load'])
            for prefix, pred, true in [('wind', wind_pred, wind_true),
                                       ('pv', pv_pred, pv_true),
                                       ('load', load_pred, load_true)]:
                err = pred - true
                metrics[f"{prefix}_mae"].append(torch.mean(torch.abs(err)).item())
                metrics[f"{prefix}_rmse"].append(torch.sqrt(torch.mean(err ** 2)).item())
    return {k: float(np.mean(v)) for k, v in metrics.items()}


class DatasetFactory:
    def __init__(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        self.train_df = train_df
        self.val_df = val_df
        self.cache: Dict[Tuple[int, int], Tuple[BESSDataset, BESSDataset]] = {}

    def get(self, seq_len: int, pred_len: int) -> Tuple[BESSDataset, BESSDataset]:
        key = (seq_len, pred_len)
        if key not in self.cache:
            train_ds = BESSDataset(self.train_df, seq_len, pred_len)
            scalers = train_ds.get_scalers()
            stats = train_ds.target_stats
            val_ds = BESSDataset(self.val_df, seq_len, pred_len, scalers=scalers, target_stats=stats)
            self.cache[key] = (train_ds, val_ds)
        return self.cache[key]


class TrialLogger:
    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def new_trial_dir(self) -> Tuple[str, str]:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        rid = f"{ts}_{random.randint(1000, 9999)}"
        path = os.path.join(self.root, rid)
        os.makedirs(path, exist_ok=True)
        return rid, path

    @staticmethod
    def dump_config(path: str, config: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)

    @staticmethod
    def dump_metrics(path: str, metrics: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def profile_model(model: nn.Module, seq_len: int, input_dim: int, device: torch.device) -> Tuple[float, float]:
    model.eval()
    dummy_x = torch.randn(1, seq_len, input_dim).to(device)
    dummy_hour = torch.zeros(1, seq_len, dtype=torch.long).to(device)
    dummy_day = torch.zeros(1, seq_len, dtype=torch.long).to(device)
    with torch.no_grad():
        flops, params = profile(model, inputs=(dummy_x, dummy_hour, dummy_day), verbose=False)
    return float(flops), float(params)


def ensure_dilation_match(cfg: Dict[str, Any]) -> None:
    block_count = cfg['dilated_blocks']
    if len(cfg['dilation_pattern']) != block_count:
        raise ValueError("dilation_pattern length must equal dilated_blocks for NAS trial")


class NASRunner:
    def __init__(self, cfg_path: str, output_root: str, random_trials: int, tpe_trials: int):
        self.cfg = load_yaml(cfg_path)
        self.output_root = output_root
        self.random_trials = random_trials
        self.tpe_trials = tpe_trials
        self.seed = int(self.cfg['data_split']['seed'])
        set_seed(self.seed)
        df = prepare_dataframe(DATA_PATH)
        self.train_df, self.val_df = split_dataframe(df, self.cfg['data_split'])
        self.dataset_factory = DatasetFactory(self.train_df, self.val_df)
        self.logger = TrialLogger(output_root)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results: List[Dict[str, Any]] = []
        arch = self.cfg['search_space']['architecture']
        allowed_blocks = set(arch['dilated_blocks']['values'])
        self.dilation_patterns: Dict[str, List[int]] = {}
        for idx, pattern in enumerate(arch['dilation_pattern']['values']):
            key = f"pat_{idx}_{'-'.join(str(p) for p in pattern)}"
            if len(pattern) in allowed_blocks:
                self.dilation_patterns[key] = pattern
        if not self.dilation_patterns:
            raise ValueError("No valid dilation patterns found for allowed block counts")
        self.dilation_keys = tuple(self.dilation_patterns.keys())
        self.allowed_blocks = allowed_blocks

    def sample_random_config(self) -> Dict[str, Any]:
        space = self.cfg['search_space']
        arch = space['architecture']
        pattern_key = random.choice(self.dilation_keys)
        dilations = self.dilation_patterns[pattern_key]
        block_count = len(dilations)
        cfg = {
            'seq_len': random.choice(space['inputs']['seq_len']['values']),
            'pred_len': random.choice(space['inputs']['pred_len']['values']),
            'hidden_dim': random.choice(arch['hidden_dim']['values']),
            'kernel_size': random.choice(arch['kernel_size']['values']),
            'dilation_pattern': dilations,
            'dilated_blocks': block_count,
            'sinusoidal_dim': random.choice(space['temporal_embedding']['sinusoidal_dim']['values']),
            'ema_decay': random.choice(space['memory']['ema_decay']['values']),
            'learnable_ema': random.choice(space['memory']['learnable_ema']['values']),
            'se_block': random.choice(space['lightweight_blocks']['se_block']['values']),
            'residual': random.choice(space['lightweight_blocks']['residual']['values']),
            'dropout': random.choice(space['lightweight_blocks']['dropout']['values']),
            'target_weights': random.choice(space['loss']['target_weights']['values']),
            'normalized_mae': random.choice(space['loss']['normalized_mae']['values']),
        }
        ensure_dilation_match(cfg)
        return cfg

    def run(self) -> None:
        print(f"Device: {self.device}")
        self._run_random_phase()
        self._run_tpe_phase()
        self._summarize()

    def _run_random_phase(self) -> None:
        for idx in range(self.random_trials):
            cfg = self.sample_random_config()
            trial_id, trial_dir = self.logger.new_trial_dir()
            print(f"[Random {idx+1}/{self.random_trials}] Trial {trial_id}")
            result = self._run_single_trial(cfg, trial_id, trial_dir, stage="random")
            if result:
                self.results.append(result)

    def _run_tpe_phase(self) -> None:
        if self.tpe_trials <= 0:
            return
        if optuna is None:
            print("Optuna not installed; skipping TPE phase.")
            return

        def objective(trial) -> float:
            cfg = self._config_from_optuna(trial)
            trial_id, trial_dir = self.logger.new_trial_dir()
            result = self._run_single_trial(cfg, trial_id, trial_dir, stage="tpe")
            if result is None:
                raise optuna.exceptions.TrialPruned("Resource constraint or training failure")
            self.results.append(result)
            return result['metrics']['mae_sum']

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.tpe_trials)
        print("TPE best value:", study.best_value)

    def _config_from_optuna(self, trial) -> Dict[str, Any]:
        space = self.cfg['search_space']
        arch = space['architecture']
        pattern_key = trial.suggest_categorical('dilation_pattern_key', self.dilation_keys)
        dilations = self.dilation_patterns[pattern_key]
        block = len(dilations)
        cfg = {
            'seq_len': trial.suggest_categorical('seq_len', space['inputs']['seq_len']['values']),
            'pred_len': trial.suggest_categorical('pred_len', space['inputs']['pred_len']['values']),
            'hidden_dim': trial.suggest_categorical('hidden_dim', arch['hidden_dim']['values']),
            'kernel_size': trial.suggest_categorical('kernel', arch['kernel_size']['values']),
            'dilated_blocks': block,
            'dilation_pattern': dilations,
            'sinusoidal_dim': trial.suggest_categorical('sinusoidal_dim', space['temporal_embedding']['sinusoidal_dim']['values']),
            'ema_decay': trial.suggest_categorical('ema_decay', space['memory']['ema_decay']['values']),
            'learnable_ema': trial.suggest_categorical('learnable_ema', space['memory']['learnable_ema']['values']),
            'se_block': trial.suggest_categorical('se_block', space['lightweight_blocks']['se_block']['values']),
            'residual': trial.suggest_categorical('residual', space['lightweight_blocks']['residual']['values']),
            'dropout': trial.suggest_categorical('dropout', space['lightweight_blocks']['dropout']['values']),
            'target_weights': trial.suggest_categorical('target_weights', space['loss']['target_weights']['values']),
            'normalized_mae': trial.suggest_categorical('normalized_mae', space['loss']['normalized_mae']['values']),
        }
        ensure_dilation_match(cfg)
        return cfg

    def _run_single_trial(self, cfg: Dict[str, Any], trial_id: str, trial_dir: str, stage: str) -> Dict[str, Any]:
        try:
            train_ds, val_ds = self.dataset_factory.get(cfg['seq_len'], cfg['pred_len'])
        except ValueError as e:
            print(f"Trial {trial_id} skipped: {e}")
            return None
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"Trial {trial_id} skipped: dataset too small for seq/pred config")
            return None

        train_loader = DataLoader(train_ds, batch_size=self.cfg['training']['batch_size'],
                                  shuffle=True, drop_last=len(train_ds) >= self.cfg['training']['batch_size'])
        val_loader = DataLoader(val_ds, batch_size=self.cfg['training']['batch_size'], shuffle=False)
        model = MultiHeadMemoryModel(train_ds.features.shape[1], cfg['pred_len'], cfg).to(self.device)
        try:
            flops, params = profile_model(model, cfg['seq_len'], train_ds.features.shape[1], self.device)
        except Exception as exc:
            print(f"Trial {trial_id} profile failed: {exc}")
            return None
        if not self._passes_resource(flops, params):
            print(f"Trial {trial_id} skipped: exceeds resource budget")
            return None

        metrics = self._train_and_eval(model, train_loader, val_loader, cfg, train_ds.target_stats)
        metrics.update({'flops': flops, 'params': params, 'mae_sum': metrics['wind_mae'] + metrics['pv_mae'] + metrics['load_mae']})
        payload = {
            'trial_id': trial_id,
            'stage': stage,
            'config': cfg,
            'metrics': metrics,
        }
        self.logger.dump_config(os.path.join(trial_dir, 'config.yaml'), cfg)
        self.logger.dump_metrics(os.path.join(trial_dir, 'metrics.json'), metrics)
        print(f"Trial {trial_id} -> MAE Sum {metrics['mae_sum']:.4f}, FLOPs {flops:.0f}, Params {params:.0f}")
        return payload

    def _train_and_eval(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                        cfg: Dict[str, Any], target_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        optim_cfg = self.cfg['training']
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg['scheduler']['max_lr'], weight_decay=0)
        total_steps = max(1, optim_cfg['warmup_epochs'] * len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=optim_cfg['scheduler']['max_lr'],
            total_steps=total_steps,
            pct_start=optim_cfg['scheduler']['pct_start'],
            div_factor=optim_cfg['scheduler']['div_factor'],
            final_div_factor=optim_cfg['scheduler']['final_div_factor'],
        )
        best_metrics = None
        best_score = float('inf')
        weights = torch.tensor(cfg['target_weights'], device=self.device, dtype=torch.float32)

        for epoch in range(optim_cfg['warmup_epochs']):
            model.train()
            for batch in train_loader:
                x = batch['x'].to(self.device)
                hour = batch['hour_idx'].to(self.device)
                day = batch['day_idx'].to(self.device)
                wind_y = batch['wind_y'].to(self.device)
                pv_y = batch['pv_y'].to(self.device)
                load_y = batch['load_y'].to(self.device)
                optimizer.zero_grad()
                outputs = model(x, hour, day)
                loss = multitask_loss(outputs,
                                      (wind_y, pv_y, load_y),
                                      weights,
                                      cfg['normalized_mae'],
                                      target_stats)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            if (epoch + 1) % 10 == 0 or epoch == optim_cfg['warmup_epochs'] - 1:
                val_metrics = evaluate(model, val_loader, target_stats, self.device)
                score = val_metrics['wind_mae'] + val_metrics['pv_mae'] + val_metrics['load_mae']
                if score < best_score:
                    best_score = score
                    best_metrics = val_metrics
        return best_metrics or evaluate(model, val_loader, target_stats, self.device)

    def _passes_resource(self, flops: float, params: float) -> bool:
        limit_flops = self.cfg['constraints']['flops_max'] * (1 + self.cfg['resource_filter']['tolerance'])
        limit_params = self.cfg['constraints']['params_max'] * (1 + self.cfg['resource_filter']['tolerance'])
        return flops <= limit_flops and params <= limit_params

    def _summarize(self) -> None:
        if not self.results:
            print("No successful NAS trials.")
            return
        def rank_key(item: Dict[str, Any]):
            metrics = item['metrics']
            return (
                metrics.get('mae_sum', float('inf')),
                metrics.get('load_rmse', float('inf')),
                metrics.get('flops', float('inf')),
            )

        ranked = sorted(self.results, key=rank_key)
        print("\nTop 5 Trials (MAE sum | FLOPs | Stage | ID):")
        for item in ranked[:5]:
            metrics = item['metrics']
            print(f"{metrics['mae_sum']:.4f} | {metrics['flops']:.0f} | {item['stage']} | {item['trial_id']}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run NAS over MemoryModel search space")
    parser.add_argument('--config', type=str, default=DEFAULT_CFG, help='YAML config path')
    parser.add_argument('--output', type=str, default=os.path.join('runs', 'nas'), help='Output root folder')
    parser.add_argument('--random-trials', type=int, default=60, help='Number of random trials')
    parser.add_argument('--tpe-trials', type=int, default=40, help='Number of TPE trials (requires optuna)')
    return parser.parse_args()


def main():
    args = parse_args()
    runner = NASRunner(cfg_path=args.config,
                       output_root=args.output,
                       random_trials=args.random_trials,
                       tpe_trials=args.tpe_trials)
    runner.run()


if __name__ == '__main__':
    main()
