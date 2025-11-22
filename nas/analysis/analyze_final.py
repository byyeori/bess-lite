import os
import json
import yaml

root = '../runs/nas'
trials = []

for d in os.listdir(root):
    metrics_path = os.path.join(root, d, 'metrics.json')
    config_path = os.path.join(root, d, 'config.yaml')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        trials.append({
            'id': d,
            'metrics': metrics,
            'config': config
        })

trials.sort(key=lambda x: x['metrics']['wind_mae'] + x['metrics']['pv_mae'] + x['metrics']['load_mae'])

print(f'Total successful trials: {len(trials)}')
print('='*90)

# Top 1 상세 분석
best = trials[0]
m = best['metrics']
c = best['config']
mae_sum = m['wind_mae'] + m['pv_mae'] + m['load_mae']

print(f"\n🏆 BEST MODEL (Trial {best['id']})")
print('='*90)
print(f"\n📊 Overall Performance:")
print(f"   Total MAE Sum: {mae_sum:.2f} kW")
print(f"\n⚡ Per-Feature Breakdown:")
print(f"   Wind:")
print(f"      MAE:  {m['wind_mae']:.2f} kW")
print(f"      RMSE: {m['wind_rmse']:.2f} kW")
print(f"   PV:")
print(f"      MAE:  {m['pv_mae']:.2f} kW")
print(f"      RMSE: {m['pv_rmse']:.2f} kW")
print(f"   Load:")
print(f"      MAE:  {m['load_mae']:.2f} kW")
print(f"      RMSE: {m['load_rmse']:.2f} kW")

print(f"\n💾 Resource Usage:")
print(f"   FLOPs:  {m['flops']:,.0f} ({m['flops']/1e3:.1f}K)")
print(f"   Params: {m['params']:,.0f}")

print(f"\n🔧 Architecture Config:")
print(f"   Input:  seq_len={c['seq_len']}, pred_len={c['pred_len']}")
print(f"   Model:  hidden_dim={c['hidden_dim']}, kernel_size={c['kernel_size']}")
print(f"   Blocks: dilated_blocks={c['dilated_blocks']}, dilation_pattern={c['dilation_pattern']}")
print(f"   SE:     {c['se_block']}")
print(f"   Residual: {c['residual']}")
print(f"   Dropout: {c['dropout']}")
print(f"   Time:   sinusoidal_dim={c['sinusoidal_dim']}")
print(f"   Memory: ema_decay={c['ema_decay']}, learnable_ema={c['learnable_ema']}")
print(f"   Loss:   target_weights={c['target_weights']}, normalized_mae={c['normalized_mae']}")

# Top 10 비교
print(f"\n\n📈 Top 10 Comparison:")
print('='*90)
print(f"{'Rank':<5} {'Trial ID':<25} {'Total MAE':<12} {'Wind':<10} {'PV':<10} {'Load':<10}")
print('-'*90)
for i, t in enumerate(trials[:10], 1):
    m = t['metrics']
    total = m['wind_mae'] + m['pv_mae'] + m['load_mae']
    print(f"{i:<5} {t['id']:<25} {total:<12.1f} {m['wind_mae']:<10.1f} {m['pv_mae']:<10.1f} {m['load_mae']:<10.1f}")

# Feature별 평균 성능 (Top 20)
print(f"\n\n📊 Average Performance (Top 20 models):")
print('='*90)
wind_maes = [t['metrics']['wind_mae'] for t in trials[:20]]
pv_maes = [t['metrics']['pv_mae'] for t in trials[:20]]
load_maes = [t['metrics']['load_mae'] for t in trials[:20]]

print(f"Wind:  MAE = {sum(wind_maes)/len(wind_maes):.1f} ± {(max(wind_maes)-min(wind_maes))/2:.1f} kW")
print(f"PV:    MAE = {sum(pv_maes)/len(pv_maes):.1f} ± {(max(pv_maes)-min(pv_maes))/2:.1f} kW")
print(f"Load:  MAE = {sum(load_maes)/len(load_maes):.1f} ± {(max(load_maes)-min(load_maes))/2:.1f} kW")

# 각 피처별 베스트 모델
print(f"\n\n🎯 Best Model per Feature:")
print('='*90)
best_wind = min(trials, key=lambda x: x['metrics']['wind_mae'])
best_pv = min(trials, key=lambda x: x['metrics']['pv_mae'])
best_load = min(trials, key=lambda x: x['metrics']['load_mae'])

print(f"Best Wind:  {best_wind['id']}")
print(f"            MAE = {best_wind['metrics']['wind_mae']:.1f} kW (Total MAE = {best_wind['metrics']['wind_mae']+best_wind['metrics']['pv_mae']+best_wind['metrics']['load_mae']:.1f})")
print(f"\nBest PV:    {best_pv['id']}")
print(f"            MAE = {best_pv['metrics']['pv_mae']:.1f} kW (Total MAE = {best_pv['metrics']['wind_mae']+best_pv['metrics']['pv_mae']+best_pv['metrics']['load_mae']:.1f})")
print(f"\nBest Load:  {best_load['id']}")
print(f"            MAE = {best_load['metrics']['load_mae']:.1f} kW (Total MAE = {best_load['metrics']['wind_mae']+best_load['metrics']['pv_mae']+best_load['metrics']['load_mae']:.1f})")

print('\n' + '='*90)
