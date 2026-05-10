# -*- coding: utf-8 -*-

import os
import torch
import numpy as np

from utils.common import *
from models import LSTMModel, RNNModel, BPNNModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")


# =========================
# 模型加载
# =========================
def load(model_cls, path, input_size, device):
    model = model_cls(input_size)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


# =========================
# 主程序
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =========================
    # 数据读取与特征工程
    # =========================
    df = load_data("../data/test.xlsx")
    df = fill_missing(df)
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_season_feature(df)
    df = add_lag_features(df, "负荷")

    feature_cols = [
        "最高温度℃", "最低温度℃", "平均温度℃",
        "相对湿度(平均)", "降雨量（mm）",
        "hour_sin", "hour_cos",
        "weekday_sin", "weekday_cos",
        "is_weekend", "is_holiday", "season",
        "负荷_lag1", "负荷_lag24", "负荷_lag168",
        "roll_mean_24", "roll_std_24"
    ]

    train, val, test = split_dataset(df)

    _, _, _, _, test_x, test_y, sx, sy = \
        normalize_train_val_test(train, val, test, feature_cols, "负荷")

    # =========================
    # 序列构造（LSTM / RNN）
    # =========================
    seq_len = 90
    X_seq, y_seq = create_sequences(test_x, test_y, seq_len)

    X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    # =========================
    # 加载模型
    # =========================
    lstm = load(LSTMModel, os.path.join(MODEL_DIR, "LSTM_best.pth"), len(feature_cols), device)
    rnn = load(RNNModel, os.path.join(MODEL_DIR, "RNN_best.pth"), len(feature_cols), device)
    bpnn = load(BPNNModel, os.path.join(MODEL_DIR, "BPNN_best.pth"), len(feature_cols), device)

    # =========================
    # 预测
    # =========================
    with torch.no_grad():

        # LSTM / RNN（序列输入）
        lstm_pred = lstm(X_seq_tensor).cpu().numpy()
        rnn_pred = rnn(X_seq_tensor).cpu().numpy()

        # =========================
        # ⭐ BPNN修复关键点
        # =========================
        # 只取最后一个时间步（17维）
        X_bpnn = X_seq[:, -1, :]
        X_bpnn = torch.tensor(X_bpnn, dtype=torch.float32).to(device)

        bpnn_pred = bpnn(X_bpnn).cpu().numpy()

    # =========================
    # 反归一化
    # =========================
    y_true = inverse_transform_y(sy, y_seq)

    lstm_pred = inverse_transform_y(sy, lstm_pred)
    rnn_pred = inverse_transform_y(sy, rnn_pred)
    bpnn_pred = inverse_transform_y(sy, bpnn_pred)

    # =========================
    # 对齐长度（防止边界差异）
    # =========================
    min_len = min(len(y_true), len(lstm_pred), len(rnn_pred), len(bpnn_pred))

    y_true = y_true[:min_len]
    lstm_pred = lstm_pred[:min_len]
    rnn_pred = rnn_pred[:min_len]
    bpnn_pred = bpnn_pred[:min_len]

    # =========================
    # 保存结果
    # =========================
    np.save("y_true.npy", y_true)
    np.save("lstm.npy", lstm_pred)
    np.save("rnn.npy", rnn_pred)
    np.save("bpnn.npy", bpnn_pred)

    print("✔ 预测结果已保存")

    # =========================
    # 评价指标
    # =========================
    print("\n========= 模型评估 =========")

    print(f"LSTM RMSE: {rmse(y_true, lstm_pred):.4f}")
    print(f"RNN  RMSE: {rmse(y_true, rnn_pred):.4f}")
    print(f"BPNN RMSE: {rmse(y_true, bpnn_pred):.4f}")

    print("\nMAE:")
    print(f"LSTM MAE: {mae(y_true, lstm_pred):.4f}")
    print(f"RNN  MAE: {mae(y_true, rnn_pred):.4f}")
    print(f"BPNN MAE: {mae(y_true, bpnn_pred):.4f}")

    print("\nMAPE:")
    print(f"LSTM MAPE: {mape(y_true, lstm_pred):.4f}")
    print(f"RNN  MAPE: {mape(y_true, rnn_pred):.4f}")
    print(f"BPNN MAPE: {mape(y_true, bpnn_pred):.4f}")


# =========================
# 入口
# =========================
if __name__ == "__main__":
    main()