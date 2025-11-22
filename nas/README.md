# NAS (Neural Architecture Search) Workspace

This folder contains the NAS pipeline for BESS energy forecasting model optimization.

## Structure

```
nas/
├── scripts/           # Execution scripts
│   ├── draft-nas.py   # Main NAS runner (random + TPE sampling)
│   └── retrain.py     # Full retraining script for top-k configs
├── config/            # Configuration files
│   └── nas_space.yaml # Search space definition
├── analysis/          # Result analysis scripts
│   ├── analyze_nas.py       # Quick NAS results overview
│   ├── analyze_final.py     # Detailed top-k analysis
│   └── compare_with_baseline.py  # Baseline comparison
├── outputs/           # Generated outputs (logs, plots)
└── __pycache__/
```

## Results Location

- NAS trial results: `../runs/nas/<trial_id>/`
- Retrained models: `../runs/retrain/<trial_id>/`

## Quick Start

### Run NAS
```bash
# Random sampling
.\.venv\Scripts\python.exe scripts\draft-nas.py --random-trials 100 --tpe-trials 0

# TPE optimization
.\.venv\Scripts\python.exe scripts\draft-nas.py --random-trials 0 --tpe-trials 80
```

### Analyze Results
```bash
.\.venv\Scripts\python.exe analysis\analyze_final.py
.\.venv\Scripts\python.exe analysis\compare_with_baseline.py
```

### Retrain Top Models
```bash
.\.venv\Scripts\python.exe scripts\retrain.py --top-k 5
```

## Latest Results (2025-11-23)

- Total trials: 127 (100 random + 27 TPE)
- Best config: Trial 20251122165212_4752
  - MAE sum: 6,887 kW (Wind: 3,831 / PV: 2,655 / Load: 401)
  - FLOPs: 102K, Params: 5.1K
  - Config: seq_len=24, pred_len=3, hidden=24, dilation=[1,3,9]
- Baseline comparison: **+31% MAE improvement, -64% FLOPs**
