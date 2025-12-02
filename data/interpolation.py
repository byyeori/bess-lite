import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

def impute_load_data(csv_path):
    """
    전력 수요 데이터 결측치 보간
    
    전략:
    1. 짧은 결측치(<=3h): Linear interpolation
    2. 긴 결측치: 같은 요일/시간 평균
    3. 남은 결측치: KNN imputation
    """
    # 1. 데이터 읽기
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    print(f"원본 결측치: {df['load'].isnull().sum()}개")
    
    # 2. 시간 features 추가
    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 3. 짧은 결측치: Linear interpolation
    df['load'] = df['load'].interpolate(method='linear', limit=3, limit_direction='both')
    print(f"Linear 보간 후: {df['load'].isnull().sum()}개 남음")
    
    # 4. 긴 결측치: 같은 요일/시간 평균
    null_indices = df[df['load'].isnull()].index.tolist()
    
    for idx in null_indices:
        hour = df.loc[idx, 'hour']
        dow = df.loc[idx, 'dayofweek']
        month = df.loc[idx, 'month']
        
        # 같은 월, 요일, 시간
        similar = df[
            (df['hour'] == hour) & 
            (df['dayofweek'] == dow) & 
            (df['month'] == month) &
            (df['load'].notnull())
        ]
        
        if len(similar) >= 3:
            # 중앙값 사용 (이상치 영향 적음)
            df.loc[idx, 'load'] = similar['load'].median()
        else:
            # 월 조건 완화
            similar = df[
                (df['hour'] == hour) & 
                (df['dayofweek'] == dow) & 
                (df['load'].notnull())
            ]
            if len(similar) >= 2:
                df.loc[idx, 'load'] = similar['load'].median()
    
    print(f"Seasonal 보간 후: {df['load'].isnull().sum()}개 남음")
    
    # 5. 남은 결측치: KNN Imputation
    if df['load'].isnull().sum() > 0:
        features = ['load', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month']
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[features] = imputer.fit_transform(df[features])
        print(f"KNN 보간 후: {df['load'].isnull().sum()}개 남음")
    
    # 6. 최종 확인
    print(f"\n✅ 보간 완료!")
    print(f"최종 결측치: {df['load'].isnull().sum()}개")
    
    # 7. 정리
    df_clean = df[['Date', 'load', 'zone']].copy()
    
    return df_clean


# 사용
df_imputed = impute_load_data('/home/pemra/wdg/wdg_urop/urop_data/data/CAISO_2014_SCE_Filled_NaN.csv')

# 저장
df_imputed.to_csv('/home/pemra/wdg/wdg_urop/urop_data/data/urop_load_60.csv', index=False)
print("\n💾 저장 완료: urop_load_60.csv")