# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# =========================
# 1️⃣ 数据读取
# =========================
def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path, engine="python")
    else:
        return pd.read_excel(path, engine="openpyxl")


# =========================
# 2️⃣ 缺失值处理（修复版）
# =========================
def fill_missing(df):
    """
    缺失值处理：
    - 插值
    - 前向 + 后向填充
    """
    df = df.interpolate()
    df = df.bfill().ffill()
    return df


# =========================
# 3️⃣ 时间特征
# =========================
def add_time_features(df, time_col='时间'):

    df[time_col] = pd.to_datetime(df[time_col])

    df['hour'] = df[time_col].dt.hour
    df['weekday'] = df[time_col].dt.weekday
    df['month'] = df[time_col].dt.month

    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # 周期编码（非常关键）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    return df


# =========================
# 4️⃣ 节假日特征
# =========================
def add_holiday_feature(df, time_col='时间'):
    import chinese_calendar as calendar

    df[time_col] = pd.to_datetime(df[time_col])

    df['is_holiday'] = df[time_col].apply(
        lambda x: int(calendar.is_holiday(x))
    )

    return df


# =========================
# 5️⃣ 季节特征
# =========================
def add_season_feature(df, time_col='时间'):

    df[time_col] = pd.to_datetime(df[time_col])
    m = df[time_col].dt.month

    df['season'] = m.map(
        lambda x: 0 if x in [3,4,5] else
                  1 if x in [6,7,8] else
                  2 if x in [9,10,11] else 3
    )

    return df


# =========================
# 6️⃣ ⭐Lag + Rolling（核心增强）
# =========================
def add_lag_features(df, target='负荷'):

    df = df.sort_values("时间").reset_index(drop=True)

    # lag特征
    df[f'{target}_lag1'] = df[target].shift(1)
    df[f'{target}_lag24'] = df[target].shift(24)
    df[f'{target}_lag168'] = df[target].shift(168)

    # rolling特征（趋势）
    df['roll_mean_24'] = df[target].rolling(24).mean()
    df['roll_std_24'] = df[target].rolling(24).std()

    df = df.dropna().reset_index(drop=True)

    return df


# =========================
# 7️⃣ 时间序列切分（防泄漏）
# =========================
def split_dataset(df, train_ratio=0.7, val_ratio=0.15):

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train = df[:train_end]
    val = df[train_end:val_end]
    test = df[val_end:]

    return train, val, test


# =========================
# 8️⃣ 归一化（只用train fit）
# =========================
def normalize_train_val_test(train, val, test, feature_cols, target_col):

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_x = scaler_x.fit_transform(train[feature_cols])
    val_x = scaler_x.transform(val[feature_cols])
    test_x = scaler_x.transform(test[feature_cols])

    train_y = scaler_y.fit_transform(train[[target_col]])
    val_y = scaler_y.transform(val[[target_col]])
    test_y = scaler_y.transform(test[[target_col]])

    return train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y


# =========================
# 9️⃣ LSTM序列构造
# =========================
def create_sequences(X, y, seq_len=90):

    Xs, ys = [], []

    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])

    return np.array(Xs), np.array(ys)


# =========================
# 🔟 反归一化
# =========================
def inverse_transform_y(scaler_y, y):
    y = np.array(y)
    return scaler_y.inverse_transform(y)


# =========================
# 1️⃣1️⃣ 评价指标
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-6, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
