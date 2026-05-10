# -*- coding: utf-8 -*-
"""
features.py
功能：
1. 对电力负荷数据集进行特征工程
2. 使用XGBoost生成特征重要性
3. 生成论文级特征重要性图：
   - 按特征类型分颜色
   - 按重要性排序
   - 带图例
   - 高分辨率保存到 utils 同级 data/fig
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os
import chinese_calendar as calendar
from matplotlib.patches import Patch


# =========================
# 数据处理函数
# =========================
def load_data(path):
    if path.endswith(".csv"):
        return pd.read_csv(path, engine="python")
    else:
        return pd.read_excel(path, engine="openpyxl")


def fill_missing(df):
    df = df.interpolate().bfill().ffill()
    return df


def add_time_features(df, time_col='时间'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['hour'] = df[time_col].dt.hour
    df['weekday'] = df[time_col].dt.weekday
    df['month'] = df[time_col].dt.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    return df


def add_holiday_feature(df, time_col='时间'):
    df[time_col] = pd.to_datetime(df[time_col])
    df['is_holiday'] = df[time_col].apply(lambda x: int(calendar.is_holiday(x)))
    return df


def add_season_feature(df, time_col='时间'):
    df[time_col] = pd.to_datetime(df[time_col])
    m = df[time_col].dt.month
    df['season'] = m.map(lambda x: 0 if x in [3, 4, 5] else
    1 if x in [6, 7, 8] else
    2 if x in [9, 10, 11] else 3)
    return df


def add_lag_features(df, target='负荷'):
    df = df.sort_values("时间").reset_index(drop=True)
    df[f'{target}_lag1'] = df[target].shift(1)
    df[f'{target}_lag24'] = df[target].shift(24)
    df[f'{target}_lag168'] = df[target].shift(168)
    df['roll_mean_24'] = df[target].rolling(24).mean()
    df['roll_std_24'] = df[target].rolling(24).std()
    df = df.dropna().reset_index(drop=True)
    return df


# =========================
# 生成论文级特征重要性图
# =========================
def feature_importance_xgb(df, target='负荷', time_col='时间'):
    feature_cols = [c for c in df.columns if c not in [time_col, target]]

    n = len(df)
    train_df = df[:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.85)]

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(
        train_df[feature_cols], train_df[target],
        eval_set=[(val_df[feature_cols], val_df[target])],
        early_stopping_rounds=20,
        verbose=False
    )

    importance_dict = model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values(by='importance', ascending=True)  # 倒序显示在图上

    # =========================
    # 分类颜色
    # =========================
    color_map = []
    for f in importance_df['feature']:
        if 'lag' in f or 'roll' in f:
            color_map.append('orange')  # 滞后/滚动
        elif f in ['hour', 'hour_sin', 'hour_cos', 'weekday', 'weekday_sin', 'weekday_cos', 'month', 'is_weekend']:
            color_map.append('skyblue')  # 时间特征
        elif f in ['is_holiday']:
            color_map.append('green')  # 节假日
        elif f in ['season']:
            color_map.append('purple')  # 季节
        else:
            color_map.append('grey')  # 其他

    # =========================
    # 保存路径
    # =========================
    fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'fig')
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, 'feature_importance_xgb_paper.png')

    # =========================
    # 论文风格绘图
    # =========================
    plt.figure(figsize=(10, 8))
    bars = plt.barh(importance_df['feature'], importance_df['importance'], color=color_map, edgecolor='black',
                    height=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title("XGBoost特征重要性（论文级彩色图）", fontsize=14, fontweight='bold')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 图例
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', label='时间特征'),
        Patch(facecolor='orange', edgecolor='black', label='滞后/滚动'),
        Patch(facecolor='green', edgecolor='black', label='节假日'),
        Patch(facecolor='purple', edgecolor='black', label='季节'),
        Patch(facecolor='grey', edgecolor='black', label='其他')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[INFO] 论文级特征重要性图已保存: {fig_path}")

    return importance_df


# =========================
# 主程序示例
# =========================
if __name__ == "__main__":
    path = "../data/train.xlsx"  # 修改为你的数据路径
    df = load_data(path)
    df = fill_missing(df)
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_season_feature(df)
    df = add_lag_features(df)

    importance_df = feature_importance_xgb(df)
    print(importance_df)