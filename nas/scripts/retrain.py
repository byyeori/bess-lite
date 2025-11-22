import argparse
import json
import os
from importlib import util as importlib_util
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader


def _load_draft_module():
    """Dynamically load draft-nas.py so we can reuse the exact data/model code."""
    module_path = os.path.join(os.path.dirname(__file__), "draft-nas.py")
    spec = importlib_util.spec_from_file_location("nas_draft", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Unable to load draft module from {module_path}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


draft = _load_draft_module()
DEFAULT_CFG = draft.DEFAULT_CFG
DATA_PATH = draft.DATA_PATH


class RetrainRunner:
    def __init__(self,
                 cfg_path: str,
                 nas_root: str,
                 output_root: str,
                 top_k: Optional[int],
                 epochs: int,
                 device: Optional[str],
                 eval_every: int,
                 trial_ids: Optional[List[str]] = None):
        self.cfg = draft.load_yaml(cfg_path)
        retrain_cfg = self.cfg.get('retraining', {})
        self.top_k = top_k or retrain_cfg.get('top_k', 3)
        self.epochs = epochs
        self.eval_every = max(1, eval_every)
        self.nas_root = nas_root
        self.output_root = output_root
        self.trial_filter = set(trial_ids) if trial_ids else None

        os.makedirs(self.output_root, exist_ok=True)
        draft.set_seed(int(self.cfg['data_split']['seed']))
        df = draft.prepare_dataframe(DATA_PATH)
        self.train_df, self.val_df = draft.split_dataframe(df, self.cfg['data_split'])
        self.dataset_factory = draft.DatasetFactory(self.train_df, self.val_df)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.batch_size = retrain_cfg.get('batch_size', self.cfg['training']['batch_size'])
        self.scheduler_cfg = retrain_cfg.get('scheduler', self.cfg['training']['scheduler'])

    def run(self) -> None:
        ranked_trials = self._select_trials()
        if not ranked_trials:
            print("No NAS trials available for retraining.")
            return
        targets = ranked_trials[:self.top_k]
        print(f"Retraining {len(targets)} trial(s) on device {self.device}...")
        for idx, trial in enumerate(targets, 1):
            print(f"[{idx}/{len(targets)}] Retraining trial {trial['trial_id']} (MAE sum {trial['metrics']['mae_sum']:.4f})")
            self._retrain_single(trial)

    def _select_trials(self) -> List[Dict[str, Any]]:
        trials = []
        if not os.path.isdir(self.nas_root):
            raise FileNotFoundError(f"NAS root not found: {self.nas_root}")
        for name in os.listdir(self.nas_root):
            trial_dir = os.path.join(self.nas_root, name)
            if not os.path.isdir(trial_dir):
                continue
            if self.trial_filter and name not in self.trial_filter:
                continue
            metrics_path = os.path.join(trial_dir, 'metrics.json')
            cfg_path = os.path.join(trial_dir, 'config.yaml')
            if not (os.path.isfile(metrics_path) and os.path.isfile(cfg_path)):
                continue
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            trial_cfg = draft.load_yaml(cfg_path)
            mae_sum = metrics.get('mae_sum')
            if mae_sum is None:
                mae_sum = metrics.get('wind_mae', float('inf')) + metrics.get('pv_mae', float('inf')) + metrics.get('load_mae', float('inf'))
                metrics['mae_sum'] = mae_sum
            trials.append({
                'trial_id': name,
                'metrics': metrics,
                'config': trial_cfg,
                'dir': trial_dir,
            })
        trials.sort(key=lambda item: (
            item['metrics'].get('mae_sum', float('inf')),
            item['metrics'].get('load_rmse', float('inf')),
            item['metrics'].get('flops', float('inf')),
        ))
        return trials

    def _retrain_single(self, trial: Dict[str, Any]) -> None:
        cfg = trial['config']
        try:
            train_ds, val_ds = self.dataset_factory.get(cfg['seq_len'], cfg['pred_len'])
        except ValueError as exc:
            print(f"Skipping {trial['trial_id']}: {exc}")
            return
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, drop_last=len(train_ds) >= self.batch_size)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        model = draft.MultiHeadMemoryModel(train_ds.features.shape[1], cfg['pred_len'], cfg).to(self.device)
        best_metrics, best_state = self._train_full(model, train_loader, val_loader, cfg, train_ds.target_stats)
        self._persist_outputs(trial['trial_id'], cfg, best_metrics, best_state)

    def _train_full(self, model: torch.nn.Module, train_loader: DataLoader,
                    val_loader: DataLoader, cfg: Dict[str, Any],
                    target_stats: Dict[str, Dict[str, float]]):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.scheduler_cfg['max_lr'])
        total_steps = max(1, self.epochs * len(train_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.scheduler_cfg['max_lr'],
            total_steps=total_steps,
            pct_start=self.scheduler_cfg['pct_start'],
            div_factor=self.scheduler_cfg['div_factor'],
            final_div_factor=self.scheduler_cfg['final_div_factor'],
        )
        weights = torch.tensor(cfg['target_weights'], device=self.device, dtype=torch.float32)
        best_metrics = None
        best_score = float('inf')
        best_state = None
        for epoch in range(self.epochs):
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
                loss = draft.multitask_loss(outputs, (wind_y, pv_y, load_y), weights, cfg['normalized_mae'], target_stats)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            if ((epoch + 1) % self.eval_every == 0) or (epoch + 1 == self.epochs):
                val_metrics = draft.evaluate(model, val_loader, target_stats, self.device)
                score = val_metrics['wind_mae'] + val_metrics['pv_mae'] + val_metrics['load_mae']
                if score < best_score:
                    best_score = score
                    best_metrics = val_metrics
                    best_metrics['mae_sum'] = score
                    best_metrics['epoch'] = epoch + 1
                    best_state = {
                        'model_state': model.state_dict(),
                        'metrics': best_metrics,
                        'epoch': epoch + 1,
                        'config': cfg,
                    }
                    print(f"  Epoch {epoch+1}: new best MAE sum {score:.4f}")
        if best_metrics is None:
            best_metrics = draft.evaluate(model, val_loader, target_stats, self.device)
            score = best_metrics['wind_mae'] + best_metrics['pv_mae'] + best_metrics['load_mae']
            best_metrics['mae_sum'] = score
            best_metrics['epoch'] = self.epochs
            best_state = {
                'model_state': model.state_dict(),
                'metrics': best_metrics,
                'epoch': self.epochs,
                'config': cfg,
            }
        return best_metrics, best_state

    def _persist_outputs(self, trial_id: str, cfg: Dict[str, Any],
                         metrics: Dict[str, Any], state: Dict[str, Any]) -> None:
        out_dir = os.path.join(self.output_root, trial_id)
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.join(out_dir, 'config.yaml')
        metrics_path = os.path.join(out_dir, 'metrics.json')
        ckpt_path = os.path.join(out_dir, 'best.ckpt')
        draft.TrialLogger.dump_config(config_path, cfg)
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        torch.save(state, ckpt_path)
        print(f"  Saved retrain artifacts to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Retrain top NAS trials with the full recipe")
    parser.add_argument('--config', type=str, default=DEFAULT_CFG, help='YAML config path used during NAS')
    parser.add_argument('--nas-root', type=str, default=os.path.join('runs', 'nas'), help='Folder with NAS trial outputs')
    parser.add_argument('--output', type=str, default=os.path.join('runs', 'retrain'), help='Retrain output root')
    parser.add_argument('--top-k', type=int, default=None, help='Override top-k selection (defaults to YAML retraining.top_k)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of full-training epochs')
    parser.add_argument('--device', type=str, default=None, help='Force device string (e.g., "cpu" or "cuda")')
    parser.add_argument('--eval-every', type=int, default=5, help='Validation frequency in epochs')
    parser.add_argument('--trial-ids', type=str, default=None, help='Comma-separated trial IDs to retrain explicitly')
    return parser.parse_args()


def main():
    args = parse_args()
    trial_ids = args.trial_ids.split(',') if args.trial_ids else None
    runner = RetrainRunner(cfg_path=args.config,
                           nas_root=args.nas_root,
                           output_root=args.output,
                           top_k=args.top_k,
                           epochs=args.epochs,
                           device=args.device,
                           eval_every=args.eval_every,
                           trial_ids=trial_ids)
    runner.run()


if __name__ == '__main__':
    main()
