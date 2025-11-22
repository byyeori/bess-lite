# NAS Resource Constraints (Cortex-M7 Target)

- **FLOPs**: ≤ 3.0 × 10^5 per inference
- **Parameters**: ≤ 30 K (≈ 30 KB @ INT8)
- **Latency Goal**: ≤ 2 ms on Cortex-M7 @ 400 MHz (proxy: FLOPs on PC)
- **Allowed Ops**: Depthwise/Pointwise Conv1d (with dilation), standard Conv1d, SE/channel attention, simple MLP block, residual add, fused BatchNorm→Conv, ReLU, Linear

---

## NAS Search Space

### 1. Input / Prediction Configuration

| ID | Item | Baseline | Candidates | Notes |
| --- | --- | --- | --- | --- |
| 1.1 | seq_len | 24 | {18, 24, 30, 36, 48} | Longer window = better seasonality, higher latency |
| 1.2 | pred_len | 3 | {3, 4, 6} | Larger pred_len increases FLOPs, monitor budget |

### 2. Channel / Receptive Field

| ID | Item | Baseline | Candidates | Notes |
| --- | --- | --- | --- | --- |
| 2.1 | hidden_dim | 32 | {24, 32, 40, 48, 56} | Wider channels improve accuracy but raise FLOPs |
| 2.2 | Dilated block count | 3 | {2, 3, 4, 5} | Each block = depthwise+pointwise conv pair |
| 2.3 | Dilation pattern | (2,4,8) | {(1,2), (2,4)}, {(1,2,4)}, {(2,4,8)}, {(1,3,9)}, {(1,2,4,8)}, {(2,4,8,16)} | Select tuples matching block count |
| 2.4 | Kernel size | 5 | {3, 5} | Kernel shared across depthwise convs |

### 3. Temporal Embedding

| ID | Item | Baseline | Candidates | Notes |
| --- | --- | --- | --- | --- |
| 3.1 | sin/cos dimension | 4 | {4, 8, 16, 24} | Higher dim captures more seasonal harmonics |

### 4. Memory Module

| ID | Item | Baseline | Candidates | Notes |
| --- | --- | --- | --- | --- |
| 4.1 | EMA decay | 0.9 | {0.8, 0.9, 0.95} | Higher decay keeps longer memory |
| 4.2 | Learnable EMA | off | {off, on} | `on` adds small param count |

### 5. Lightweight Blocks (Attention / Regularization)

| ID | Item | Baseline | Candidates | Notes |
| --- | --- | --- | --- | --- |
| 5.1 | SE block | none | {none, squeeze ratio 8, squeeze ratio 4} | Applies per block |
| 5.2 | Residual | all blocks on | {all blocks, last block only, off} | Off saves adds but risks training stability |
| 5.3 | Dropout / DropPath | 0 | {0, 0.05, 0.1, 0.15, 0.2} | DropPath = stochastic depth

### 6. Loss Configuration

| ID | Item | Baseline | Candidates | Notes |
| --- | --- | --- | --- | --- |
| 6.1 | Target weights (Wind / PV / Load) | [1, 1, 1] | [1.0, 1.0, 1.0]; [0.6, 1.0, 1.4]; [1.3, 1.0, 0.7]; [1.0, 1.4, 0.8]; [0.8, 1.2, 1.0] | Keep sum ≈3 to stabilize loss scale |
| 6.2 | Normalized MAE | off | {off, on} | Optional scaling by per-target std |

---

## Notes

1. Apply FLOPs/parameter constraints after each candidate is sampled; discard invalid configs early.
2. Pred_len changes the head size and FLOPs; ensure latency proxy accounts for it.
3. Warm-up NAS phase can run shorter epochs (e.g., 80) before full retraining.

---

## NAS Run Protocol (Finalized)

### Data Split & Seed
- Train 구간: **2014-01-01 00:00 ~ 2014-09-30 23:00**
- Validation 구간: **2014-10-01 00:00 ~ 2014-12-31 23:00**
- Random seed: **42** (PyTorch, NumPy, DataLoader 모두 동일하게 적용)

### Warm-up Training Recipe
- Epoch: **80**
- Batch size: **64**
- Optimizer: **Adam (β₁=0.9, β₂=0.999)**
- Scheduler: **OneCycleLR** with `max_lr=8e-4`, `pct_start=0.3`, `div_factor=10`, `final_div_factor=10`
- 나머지 하이퍼파라미터(gradient clip 등)는 baseline과 동일 유지

### Resource Filter Rule
- FLOPs/Params 계산: **thop** 사용, dummy 입력 `(batch=1, seq_len=후보 seq_len)`
- pred_len은 해당 후보 값으로 설정해 헤드 연산까지 포함
- 허용 한도: `floPs ≤ 3.0e5`, `params ≤ 3.0e4`; 프로파일 오차 보정을 위해 **±1% 이내 초과**는 허용
- 기준 초과 시 해당 후보는 즉시 탈락시켜 학습을 시작하지 않음

### Evaluation & Ranking
- 1차 지표: **Wind/PV/Load MAE 합계** (validation set)
- Tie-breaker 1: **Load RMSE**가 낮은 모델 우선
- Tie-breaker 2: 동일하면 **FLOPs**가 작은 모델 우선

### Logging & Checkpoints
- Trial ID: `YYYYMMDDHHMMSS_rand4` 형식으로 생성
- 각 시도는 `runs/nas/<trial_id>/` 폴더 사용
- Warm-up 단계 저장물: `config.yaml`, `metrics.json`만 보관 (checkpoint 생략)
- Top-k 재학습 단계에서만 `best.ckpt`를 추가 저장 (필요 시 FP16로 저장 가능)

### Top-k Retraining
- Warm-up 후 상위 **5개** 후보를 선별
- 재학습은 baseline 설정(200 epoch, batch 64, Adam + 동일 OneCycle)으로 실행
- 재학습 완료 후 `best.ckpt` 확보까지 NAS 단계에서 마무리하고, 양자화(PTQ/QAT)는 별도 절차로 수행
