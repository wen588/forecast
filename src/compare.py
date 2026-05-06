# -*- coding: utf-8 -*-

import os
import sys
import io
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.common import *
from utils.log import Logger


# =========================
# 中文支持
# =========================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# 路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model")
FIG_DIR = os.path.join(BASE_DIR, "data", "fig")

os.makedirs(FIG_DIR, exist_ok=True)


logger = Logger(BASE_DIR, "compare").get_logger()


# =========================
# 模型定义（必须和训练一致）
# =========================

class BPNNModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class RNNModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


class LSTMModel(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =========================
# 加载模型（统一安全版）
# =========================
def load_model(model_class, name, input_size, device):

    path = os.path.join(MODEL_DIR, f"{name}_best.pth")

    model = model_class(input_size).to(device)

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到模型: {path}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    model.eval()

    logger.info(f"加载模型成功: {path}")

    return model


# =========================
# 评估
# =========================
def predict(model, X, device):
    with torch.no_grad():
        X = torch.tensor(X, dtype=torch.float32).to(device)
        return model(X).cpu().numpy()


# =========================
# 主函数
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"使用设备: {device}")

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

    # ========= LSTM需要序列 =========
    seq_len = 90
    X_test_seq, y_test_seq = create_sequences(test_x, test_y, seq_len)

    # ========= BPNN / RNN直接用 =========
    X_test_flat = test_x
    y_test_flat = test_y

    # ========= 加载模型 =========
    bpnn = load_model(BPNNModel, "BPNN", len(feature_cols), device)
    rnn = load_model(RNNModel, "RNN", len(feature_cols), device)
    lstm = load_model(LSTMModel, "LSTM", len(feature_cols), device)

    # ========= 预测 =========
    pred_bpnn = predict(bpnn, X_test_flat, device)
    pred_rnn = predict(rnn, X_test_seq, device)
    pred_lstm = predict(lstm, X_test_seq, device)

    # ========= 反归一化 =========
    y_true = inverse_transform_y(scaler_y, y_test_flat[:len(pred_bpnn)])

    pred_bpnn = inverse_transform_y(scaler_y, pred_bpnn)
    pred_rnn = inverse_transform_y(scaler_y, pred_rnn[:len(y_true)])
    pred_lstm = inverse_transform_y(scaler_y, pred_lstm[:len(y_true)])


    # =========================
    # ⭐图1：预测对比曲线（论文核心图）
    # =========================
    plt.figure(figsize=(12,5))

    plt.plot(y_true, label="真实值", linewidth=2)
    plt.plot(pred_bpnn, label="BPNN")
    plt.plot(pred_rnn, label="RNN")
    plt.plot(pred_lstm, label="LSTM")

    plt.title("负荷预测对比图")
    plt.legend()

    path1 = os.path.join(FIG_DIR, "Fig1_预测对比.png")
    plt.savefig(path1, dpi=300, bbox_inches="tight")
    plt.close()


    # =========================
    # ⭐图2：误差对比
    # =========================
    def mae_fn(y, p): return np.mean(np.abs(y - p))

    errors = [
        mae_fn(y_true, pred_bpnn),
        mae_fn(y_true, pred_rnn),
        mae_fn(y_true, pred_lstm)
    ]

    plt.figure()
    plt.bar(["BPNN", "RNN", "LSTM"], errors)

    plt.title("MAE误差对比")

    path2 = os.path.join(FIG_DIR, "Fig2_误差对比.png")
    plt.savefig(path2, dpi=300, bbox_inches="tight")
    plt.close()


    # =========================
    # ⭐图3：局部放大（论文必备）
    # =========================
    zoom = slice(0, 200)

    plt.figure(figsize=(12,5))
    plt.plot(y_true[zoom], label="真实")
    plt.plot(pred_lstm[zoom], label="LSTM")

    plt.title("局部预测效果（LSTM）")

    path3 = os.path.join(FIG_DIR, "Fig3_局部放大.png")
    plt.savefig(path3, dpi=300, bbox_inches="tight")
    plt.close()


    # =========================
    # ⭐图4：散点图（拟合能力）
    # =========================
    plt.figure()

    plt.scatter(y_true, pred_lstm, alpha=0.5)

    plt.title("LSTM预测拟合散点图")

    path4 = os.path.join(FIG_DIR, "Fig4_散点图.png")
    plt.savefig(path4, dpi=300, bbox_inches="tight")
    plt.close()


    # =========================
    # 输出指标
    # =========================
    logger.info("===== 论文级结果 =====")
    logger.info(f"BPNN MAE: {mae_fn(y_true, pred_bpnn):.4f}")
    logger.info(f"RNN  MAE: {mae_fn(y_true, pred_rnn):.4f}")
    logger.info(f"LSTM MAE: {mae_fn(y_true, pred_lstm):.4f}")

    logger.info("图像已生成（论文四图完成）")


if __name__ == "__main__":
    main()