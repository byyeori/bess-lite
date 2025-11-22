import pickle
import json
import yaml
import os
import numpy as np

# 1. 베이스라인 결과 (정규화된 값)
baseline = {
    'wind_mae': 0.1108,
    'pv_mae': 0.0823,
    'load_mae': 0.0361,
    'wind_rmse': 0.1718,
    'pv_rmse': 0.1318,
    'load_rmse': 0.0445,
    'flops': 286056,
    'params': 13483
}

# 2. 데이터에서 직접 범위 계산
import pandas as pd
df = pd.read_csv('data/urop_data_final.csv')

wind_range = df['wind_kw'].max() - df['wind_kw'].min()
pv_range = df['pv_kw'].max() - df['pv_kw'].min()
load_range = df['load_kw'].max() - df['load_kw'].min()

print("Data Ranges (kW):")
print(f"  Wind: {wind_range:.1f} kW (min={df['wind_kw'].min():.1f}, max={df['wind_kw'].max():.1f})")
print(f"  PV:   {pv_range:.1f} kW (min={df['pv_kw'].min():.1f}, max={df['pv_kw'].max():.1f})")
print(f"  Load: {load_range:.1f} kW (min={df['load_kw'].min():.1f}, max={df['load_kw'].max():.1f})")
print()

# 3. NAS Top-1 결과 로드
best_trial_id = '20251122165212_4752'
with open(f'../runs/nas/{best_trial_id}/metrics.json') as f:
    nas_metrics = json.load(f)
with open(f'../runs/nas/{best_trial_id}/config.yaml') as f:
    nas_config = yaml.safe_load(f)

# 4. NAS 결과를 정규화 (kW → normalized)
nas_normalized = {
    'wind_mae': nas_metrics['wind_mae'] / wind_range,
    'pv_mae': nas_metrics['pv_mae'] / pv_range,
    'load_mae': nas_metrics['load_mae'] / load_range,
    'wind_rmse': nas_metrics['wind_rmse'] / wind_range,
    'pv_rmse': nas_metrics['pv_rmse'] / pv_range,
    'load_rmse': nas_metrics['load_rmse'] / load_range,
    'flops': nas_metrics['flops'],
    'params': nas_metrics['params']
}

# 5. 비교 출력
print("="*90)
print("BASELINE (draft.py) vs NAS Top-1 (Normalized MAE)")
print("="*90)
print(f"\n{'Metric':<15} {'Baseline':<15} {'NAS Top-1':<15} {'Improvement':<15}")
print("-"*90)

wind_improve = (baseline['wind_mae'] - nas_normalized['wind_mae']) / baseline['wind_mae'] * 100
pv_improve = (baseline['pv_mae'] - nas_normalized['pv_mae']) / baseline['pv_mae'] * 100
load_improve = (baseline['load_mae'] - nas_normalized['load_mae']) / baseline['load_mae'] * 100

print(f"{'Wind MAE':<15} {baseline['wind_mae']:<15.4f} {nas_normalized['wind_mae']:<15.4f} {wind_improve:>+13.1f}%")
print(f"{'PV MAE':<15} {baseline['pv_mae']:<15.4f} {nas_normalized['pv_mae']:<15.4f} {pv_improve:>+13.1f}%")
print(f"{'Load MAE':<15} {baseline['load_mae']:<15.4f} {nas_normalized['load_mae']:<15.4f} {load_improve:>+13.1f}%")

total_base = baseline['wind_mae'] + baseline['pv_mae'] + baseline['load_mae']
total_nas = nas_normalized['wind_mae'] + nas_normalized['pv_mae'] + nas_normalized['load_mae']
total_improve = (total_base - total_nas) / total_base * 100

print(f"{'Total MAE':<15} {total_base:<15.4f} {total_nas:<15.4f} {total_improve:>+13.1f}%")

print("\n" + "-"*90)
print(f"{'Wind RMSE':<15} {baseline['wind_rmse']:<15.4f} {nas_normalized['wind_rmse']:<15.4f}")
print(f"{'PV RMSE':<15} {baseline['pv_rmse']:<15.4f} {nas_normalized['pv_rmse']:<15.4f}")
print(f"{'Load RMSE':<15} {baseline['load_rmse']:<15.4f} {nas_normalized['load_rmse']:<15.4f}")

print("\n" + "="*90)
print("Resource Usage")
print("="*90)
flops_reduce = (1 - nas_normalized['flops'] / baseline['flops']) * 100
params_reduce = (1 - nas_normalized['params'] / baseline['params']) * 100

print(f"{'FLOPs':<15} {baseline['flops']:>13,} {nas_normalized['flops']:>15,} {flops_reduce:>+13.1f}%")
print(f"{'Params':<15} {baseline['params']:>13,} {nas_normalized['params']:>15,} {params_reduce:>+13.1f}%")

print("\n" + "="*90)
print("Summary")
print("="*90)
print(f"✅ Performance: Total MAE improved by {total_improve:+.1f}%")
print(f"✅ Efficiency:  FLOPs reduced by {-flops_reduce:.1f}%, Params reduced by {-params_reduce:.1f}%")
print(f"✅ NAS Config:  seq_len={nas_config['seq_len']}, pred_len={nas_config['pred_len']}, "
      f"hidden={nas_config['hidden_dim']}, dilation={nas_config['dilation_pattern']}")
