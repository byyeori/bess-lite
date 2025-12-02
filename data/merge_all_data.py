import pandas as pd

# 1. 파일 불러오기
df_wind = pd.read_csv("/home/pemra/wdg/wdg_urop/urop_data/urop_wind_kw_60.csv")
df_load = pd.read_csv("/home/pemra/wdg/wdg_urop/urop_data/urop_load_60.csv")
df_weather = pd.read_csv("/home/pemra/wdg/wdg_urop/urop_data/urop_weather_60.csv")
df_pv = pd.read_csv("/home/pemra/wdg/wdg_urop/urop_data/urop_pv_kw_60.csv")

# 2. 날짜/시간 형식 통일 함수 (Wind, PV용)
def parse_wind_pv_time(ts):
    # "Jan 1, 12:00 am" 형식에 연도(2014)를 추가하여 변환
    ts_with_year = f"2014 {ts}"
    return pd.to_datetime(ts_with_year, format='%Y %b %d, %I:%M %p')

# 3. 각 데이터프레임 시간축 변환 및 정리

# Wind Data
df_wind['timestamp_dt'] = df_wind['Time stamp'].apply(parse_wind_pv_time)
df_wind = df_wind.rename(columns={'System power generated | (kW)': 'wind_kw'})
df_wind = df_wind[['timestamp_dt', 'wind_kw']]

# Load Data
df_load['timestamp_dt'] = pd.to_datetime(df_load['Date'])
df_load = df_load.rename(columns={'load': 'load_kw'})
df_load = df_load[['timestamp_dt', 'load_kw']]

# Weather Data
# Year, Month, Day, Hour 컬럼을 합쳐서 datetime 생성
df_weather['timestamp_dt'] = pd.to_datetime(df_weather[['Year', 'Month', 'Day', 'Hour']])
# 불필요한 시간 관련 컬럼 제외하고 선택
weather_cols = [c for c in df_weather.columns if c not in ['Year', 'Month', 'Day', 'Hour', 'Minute', 'timestamp_dt']]
df_weather = df_weather[['timestamp_dt'] + weather_cols]

# PV Data
df_pv['timestamp_dt'] = df_pv['Time stamp'].apply(parse_wind_pv_time)
df_pv = df_pv.rename(columns={'System power generated | (kW)': 'pv_kw'})
df_pv = df_pv[['timestamp_dt', 'pv_kw']]

# 4. 데이터 병합 (시간축 기준)
df_merged = df_wind.merge(df_load, on='timestamp_dt', how='outer')
df_merged = df_merged.merge(df_weather, on='timestamp_dt', how='outer')
df_merged = df_merged.merge(df_pv, on='timestamp_dt', how='outer')

# 시간순 정렬
df_merged = df_merged.sort_values('timestamp_dt')

# 5. 요청하신 형식 ("1/1/2014 00")으로 날짜 포맷 변경
def custom_date_format(dt):
    return f"{dt.month}/{dt.day}/{dt.year} {dt.hour:02d}"

# timestamp 컬럼을 맨 앞으로 배치
df_merged.insert(0, 'timestamp', df_merged['timestamp_dt'].apply(custom_date_format))

# 임시로 사용한 datetime 컬럼 삭제
df_merged = df_merged.drop(columns=['timestamp_dt'])

# 6. 결과 저장
output_filename = 'urop_final_data.csv'
df_merged.to_csv(output_filename, index=False)

print(f"통합 완료! {output_filename}에 저장되었습니다.")
print(df_merged.head())