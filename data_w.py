import pandas as pd
import os

WEATHER_PATH = os.path.join("data", "w2024.csv")  

df = pd.read_csv(WEATHER_PATH, skiprows=2)

df['timestamp'] = pd.to_datetime(dict(
    year=df['Year'],
    month=df['Month'],
    day=df['Day'],
    hour=df['Hour'],
    minute=df['Minute']
), utc=True)

keep_cols = [
    'timestamp',       
    'DHI',
    'DNI',
    'GHI',
    'Wind Speed',
    'Temperature',
    'Pressure'
]

df_selected = df[[c for c in keep_cols if c in df.columns]]

output_path = os.path.join("data", "nsrdb2024.csv")
df_selected.to_csv(output_path, index=False)

print(f"저장 완료 : {output_path}")
print(df_selected.head(10))


