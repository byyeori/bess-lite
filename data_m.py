import pandas as pd
import os

BASE_DIR = "data"

nsrdb_path = os.path.join(BASE_DIR, "nsrdb2024.csv")
data_path = os.path.join(BASE_DIR, "data2024.csv")
save_path = os.path.join(BASE_DIR, "merged_2024.csv")

nsrdb = pd.read_csv(nsrdb_path, parse_dates=["timestamp"])
data = pd.read_csv(data_path, parse_dates=["timestamp"])

start_nsrdb = nsrdb["timestamp"].min()
start_data = data["timestamp"].min()
start_time = max(start_nsrdb, start_data)

nsrdb = nsrdb[nsrdb["timestamp"] >= start_time]
data = data[data["timestamp"] >= start_time]
merged = pd.merge(data, nsrdb, on="timestamp", how="inner")

merged.to_csv(save_path, index=False)
print(f"병합 완료: {merged.shape[0]}행 저장됨")
print(f"저장 위치: {save_path}")
print(f"병합 기준 시작 시각: {start_time}")
print(merged.head(5))