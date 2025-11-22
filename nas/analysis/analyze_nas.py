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

print(f'Total successful trials: {len(trials)}\n')
print('='*80)
print('Top 10 by MAE sum:')
print('='*80)
for i, t in enumerate(trials[:10]):
    m = t['metrics']
    mae_sum = m['wind_mae'] + m['pv_mae'] + m['load_mae']
    print(f"{i+1}. {t['id']}")
    print(f"   MAE_sum={mae_sum:.1f} (Wind:{m['wind_mae']:.1f}, PV:{m['pv_mae']:.1f}, Load:{m['load_mae']:.1f})")
    print(f"   RMSE (Wind:{m['wind_rmse']:.1f}, PV:{m['pv_rmse']:.1f}, Load:{m['load_rmse']:.1f})")
    print(f"   FLOPs={m['flops']:.0f}, Params={m['params']}")
    print()

best = trials[0]
print('='*80)
print(f"Best Config (Trial {best['id']})")
print('='*80)
c = best['config']
print(f"seq_len={c['seq_len']}, pred_len={c['pred_len']}, hidden_dim={c['hidden_dim']}, kernel_size={c['kernel_size']}")
print(f"dilated_blocks={c['dilated_blocks']}, dilation_pattern={c['dilation_pattern']}")
print(f"se_block={c['se_block']}, residual={c['residual']}, dropout={c['dropout']}")
print(f"sinusoidal_dim={c['sinusoidal_dim']}, ema_decay={c['ema_decay']}, learnable_ema={c['learnable_ema']}")
print(f"target_weights={c['target_weights']}, normalized_mae={c['normalized_mae']}")

print('\n' + '='*80)
print('Resource Efficiency (MAE per MFLOPs, lower=better):')
print('='*80)
efficiency = []
for t in trials[:20]:
    m = t['metrics']
    mae_sum = m['wind_mae'] + m['pv_mae'] + m['load_mae']
    eff = mae_sum / (m['flops'] / 1e6)
    efficiency.append((t['id'], eff, mae_sum, m['flops']))

efficiency.sort(key=lambda x: x[1])
for i, (tid, eff, mae, flops) in enumerate(efficiency[:5]):
    print(f"{i+1}. {tid}: {eff:.2f} MAE/MFLOPs (MAE={mae:.0f}, FLOPs={flops:.0f})")
