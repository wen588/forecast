# -*- coding: utf-8 -*-

import os
import sys
import io
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from utils.common import *

# =========================
# ⭐中文支持
# =========================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# ⭐路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "data", "fig")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(FIG_DIR, exist_ok=True)


# =========================
# ⭐模型定义（必须一致）
# =========================
class BPNNModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================
# ⭐自动加载最新模型
# =========================
def load_latest(model_name, model_class, input_size, device):

    files = [f for f in os.listdir(MODEL_DIR) if model_name in f]

    if not files:
        raise ValueError(f"没有找到 {model_name} 模型")

    latest = sorted(files)[-1]
    path = os.path.join(MODEL_DIR, latest)

    model = model_class(input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

    print(f"已加载模型: {path}")

    return model


# =========================
# ⭐主函数
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========= 数据 =========
    df = load_data("../data/train.xlsx")
    df = df.sort_values("时间").reset_index(drop=True)

    df = fill_missing(df)
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_season_feature(df)
    df = add_lag_features(df, target="负荷")

    feature_cols = [
        "最高温度℃", "最低温度℃", "平均温度℃",
        "相对湿度(平均)", "降雨量（mm）",
        "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
        "is_weekend",
        "is_holiday",
        "season",
        "负荷_lag1",
        "负荷_lag24",
        "负荷_lag168",
        "roll_mean_24",
        "roll_std_24"
    ]

    target_col = "负荷"

    train, val, test = split_dataset(df)

    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = \
        normalize_train_val_test(train, val, test, feature_cols, target_col)

    # ========= 构造序列 =========
    seq_len = 90
    X_test, y_test = create_sequences(test_x, test_y, seq_len)

    X_seq = torch.tensor(X_test, dtype=torch.float32).to(device)

    # =========================
    # ⭐加载模型
    # =========================
    bpnn = load_latest("BPNN", BPNNModel, len(feature_cols), device)
    rnn = load_latest("RNN", RNNModel, len(feature_cols), device)
    lstm = load_latest("LSTM", LSTMModel, len(feature_cols), device)

    # =========================
    # ⭐预测
    # =========================
    with torch.no_grad():
        pred_rnn = rnn(X_seq).cpu().numpy()
        pred_lstm = lstm(X_seq).cpu().numpy()

    # BPNN输入处理
    X_bpnn = X_test[:, -1, :]
    X_bpnn = torch.tensor(X_bpnn, dtype=torch.float32).to(device)

    with torch.no_grad():
        pred_bpnn = bpnn(X_bpnn).cpu().numpy()

    # ========= 反归一化 =========
    y_true = inverse_transform_y(scaler_y, y_test)

    pred_bpnn = inverse_transform_y(scaler_y, pred_bpnn)
    pred_rnn = inverse_transform_y(scaler_y, pred_rnn)
    pred_lstm = inverse_transform_y(scaler_y, pred_lstm)

    # =========================
    # ⭐指标
    # =========================
    results = {
        "BPNN": [rmse(y_true, pred_bpnn), mae(y_true, pred_bpnn), mape(y_true, pred_bpnn)],
        "RNN":  [rmse(y_true, pred_rnn), mae(y_true, pred_rnn), mape(y_true, pred_rnn)],
        "LSTM": [rmse(y_true, pred_lstm), mae(y_true, pred_lstm), mape(y_true, pred_lstm)]
    }

    print("\n===== 模型对比 =====")
    for k, v in results.items():
        print(f"{k}: RMSE={v[0]:.3f} MAE={v[1]:.3f} MAPE={v[2]:.2f}")

    # =========================
    # ⭐图1：预测对比
    # =========================
    plt.figure(figsize=(12,6))
    plt.plot(y_true[:300], label="真实值", linewidth=2)
    plt.plot(pred_bpnn[:300], label="BPNN")
    plt.plot(pred_rnn[:300], label="RNN")
    plt.plot(pred_lstm[:300], label="LSTM")

    plt.title("模型预测对比图")
    plt.legend()

    p1 = os.path.join(FIG_DIR, f"预测对比_{datetime.now().strftime('%m%d_%H%M')}.png")
    plt.savefig(p1, dpi=300)
    plt.close()

    # =========================
    # ⭐图2：误差柱状图
    # =========================
    labels = ["RMSE","MAE","MAPE"]
    x = np.arange(3)

    plt.figure()
    plt.bar(x-0.25, results["BPNN"], 0.25, label="BPNN")
    plt.bar(x,      results["RNN"],  0.25, label="RNN")
    plt.bar(x+0.25, results["LSTM"], 0.25, label="LSTM")

    plt.xticks(x, labels)
    plt.title("误差对比")
    plt.legend()

    p2 = os.path.join(FIG_DIR, f"误差对比_{datetime.now().strftime('%m%d_%H%M')}.png")
    plt.savefig(p2, dpi=300)
    plt.close()

    # =========================
    # ⭐图3：残差分布（论文加分）
    # =========================
    plt.figure()
    plt.hist(y_true - pred_lstm, bins=50, alpha=0.7)

    plt.title("LSTM残差分布")
    plt.xlabel("误差")
    plt.ylabel("频数")

    p3 = os.path.join(FIG_DIR, f"残差分布_{datetime.now().strftime('%m%d_%H%M')}.png")
    plt.savefig(p3, dpi=300)
    plt.close()

    print("\n已生成论文图：")
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()