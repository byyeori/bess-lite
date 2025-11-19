# fetch_caiso.py
# 사용법(Windows PowerShell):
#   cd "<...>\UROP-2025\code"
#   .\.venv\Scripts\Activate.ps1
#   python -m pip install -U pip gridstatus pandas
#   python fetch_caiso.py

from pathlib import Path
import re
import pandas as pd
from gridstatus import CAISO

# ---------------------- 경로/파일 설정 ----------------------
HERE = Path(__file__).resolve().parent
DATASET = HERE.parent / "dataset" / "bess-nsrdb-kern-2024-30min"
SAM_FILE = DATASET / "sam" / "pv-kern-2024-30min-utc-sam-v1.csv"
OUT_DIR = DATASET / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "kern-2024-30min-utc-merged.csv"

# ---------------------- 1) SAM pv_kW 로드 -------------------
df_pv = pd.read_csv(SAM_FILE)

# SAM Export의 첫 컬럼이 시간열(예: "30 Minute Data")인 경우가 많음
time_col = df_pv.columns[0]
pv_col = next((c for c in df_pv.columns if "system power generated" in c.lower()), None)
if pv_col is None:
    raise RuntimeError(f"SAM CSV에서 PV 열을 못 찾음. columns={list(df_pv.columns)}")

# 파일명에서 연도 파악(예: pv-kern-2024-30min-utc-sam-v1.csv)
m = re.search(r"-(\d{4})-30min", SAM_FILE.name)
year_in_name = int(m.group(1)) if m else None

def parse_ts(v: str) -> pd.Timestamp:
    s = str(v).strip()
    # 이미 연도가 있으면 그대로 파싱
    if re.search(r"\b(20\d{2}|19\d{2})\b", s):
        return pd.to_datetime(s, utc=True, errors="coerce")
    # 연도 없으면 파일명 연도 보강
    if not year_in_name:
        raise RuntimeError("타임스탬프에 연도가 없고, 파일명에서도 연도 추출 실패")
    return pd.to_datetime(f"{s}, {year_in_name}", utc=True, errors="coerce")

ts = df_pv[time_col].apply(parse_ts)
if ts.isna().any():
    bad = df_pv.loc[ts.isna(), time_col].head(3).tolist()
    raise RuntimeError(f"SAM timestamp 파싱 실패 예시: {bad}")

df_pv = (
    df_pv.assign(timestamp=ts)
         .rename(columns={pv_col: "pv_kW"})
         [["timestamp", "pv_kW"]]
         .sort_values("timestamp")
         .reset_index(drop=True)
)

# ---------------------- 2) CAISO 부하/가격 -------------------
iso = CAISO()

# 2-1) 부하 (MW -> kW), 30분 리샘플, UTC 통일
load_raw = iso.get_load(start="2024-01-01", end="2024-12-31")

# 컬럼 자동 탐색(버전별 명칭 편차 대응)
t_col = next((c for c in load_raw.columns if "time" in c.lower()), None)
ld_cols = [c for c in load_raw.columns if ("load" in c.lower() or "demand" in c.lower()) and c != t_col]
if t_col is None or not ld_cols:
    raise RuntimeError(f"CAISO load 컬럼 탐색 실패: {list(load_raw.columns)}")
ld_col = next((c for c in ld_cols if "mw" in c.lower()), ld_cols[0])

load = load_raw.rename(columns={t_col: "timestamp", ld_col: "load_MW"}).copy()
ts = pd.to_datetime(load["timestamp"], errors="coerce")
# tz-aware면 UTC로 변환, 아니면 Pacific으로 가정 후 UTC로 변환
if getattr(ts.dt, "tz", None) is not None:
    load["timestamp"] = ts.dt.tz_convert("UTC")
else:
    load["timestamp"] = ts.dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC")

load = (
    load[["timestamp", "load_MW"]]
        .set_index("timestamp")
        .resample("30T").mean()
        .assign(load_kW=lambda d: d["load_MW"] * 1000.0)
        .drop(columns=["load_MW"])
        .reset_index()
)

# 2-2) LMP 가격 — 여러 후보(로케이션/마켓) 순차 시도
def fetch_lmp_first_ok(iso, start, end):
    location_candidates = [
        "DLAP_SCE-APND",  # SCE DLAP
        "TH_SP15",        # Trading Hub SP15
        "SP15",           # 일부 버전에서 허용
    ]
    market_candidates = [
        "DAY_AHEAD_HOURLY",
        "DAY_AHEAD_15_MIN",
        "REAL_TIME_5_MIN",
    ]
    last_err = None
    for loc in location_candidates:
        for mkt in market_candidates:
            try:
                df = iso.get_lmp(start=start, end=end, market=mkt, locations=[loc])
                if df is not None and len(df) > 0:
                    df["__loc"] = loc
                    df["__market"] = mkt
                    return df
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"LMP 조회 실패. 마지막 오류: {last_err}")

lmp_raw = fetch_lmp_first_ok(iso, start="2024-01-01", end="2024-12-31")

# 컬럼 정규화 + 30분 시계열로 보간
t_col = next((c for c in lmp_raw.columns if "time" in c.lower()), None)
p_col = next((c for c in lmp_raw.columns if c.lower() in ["lmp", "price", "lmp ($/mwh)"]), None)
if t_col is None or p_col is None:
    raise RuntimeError(f"LMP 컬럼 탐색 실패: {list(lmp_raw.columns)}")

lmp = lmp_raw.rename(columns={t_col: "timestamp", p_col: "price"})[["timestamp", "price", "__loc", "__market"]]
ts = pd.to_datetime(lmp["timestamp"], errors="coerce")
if getattr(ts.dt, "tz", None) is not None:
    lmp["timestamp"] = ts.dt.tz_convert("UTC")
else:
    lmp["timestamp"] = ts.dt.tz_localize("America/Los_Angeles").dt.tz_convert("UTC")

lmp = (
    lmp.set_index("timestamp")
       .resample("30T")
       .interpolate("time")
       .reset_index()
)

print("LMP source:", lmp_raw["__loc"].iloc[0], lmp_raw["__market"].iloc[0])

# ---------------------- 3) 병합 & 저장 -----------------------
df = (
    df_pv.merge(load, on="timestamp", how="inner")
         .merge(lmp[["timestamp", "price"]], on="timestamp", how="inner")
)
df["wind_kW"] = 0.0
df["net_load"] = df["load_kW"] - df["pv_kW"] - df["wind_kW"]

df.to_csv(OUT_FILE, index=False)
print("Saved:", OUT_FILE)
print(df.head())
print("shape:", df.shape)
