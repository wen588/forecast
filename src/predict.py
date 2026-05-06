# -*- coding: utf-8 -*-

import os
import sys
import io
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset

from utils.common import (
    load_data,
    fill_missing,
    add_time_features,
    add_holiday_feature,
    add_season_feature,
    add_lag_features,
    split_dataset,
    normalize_train_val_test,
    create_sequences,
    inverse_transform_y,
    rmse, mae, mape
)

from utils.log import Logger


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
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# ⭐日志
# =========================
logger = Logger(BASE_DIR, "rnn").get_logger()


# =========================
# 1️⃣ RNN模型（SimpleRNN）
# =========================
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]   # 取最后时刻
        return self.fc(out)


# =========================
# 2️⃣ 训练函数
# =========================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.0005, device='cuda'):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    train_losses, val_losses = [], []

    best_loss = float("inf")
    best_model = model.state_dict()

    patience = 5
    wait = 0

    for epoch in range(epochs):

        # ===== train =====
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()

            pred = model(X_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== val =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                pred = model(X_batch)
                loss = criterion(pred, y_batch)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"第 {epoch+1} 轮 | 训练损失: {train_loss:.6f} | 验证损失: {val_loss:.6f}")

        # ===== early stopping =====
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            logger.info("早停触发")
            break

    model.load_state_dict(best_model)
    return model, train_losses, val_losses


# =========================
# ⭐画图
# =========================
def plot_loss(train_losses, val_losses):

    plt.figure()

    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")

    plt.title("RNN训练损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    name = f"RNN损失曲线_{datetime.now().strftime('%m%d_%H%M')}.png"
    path = os.path.join(FIG_DIR, name)

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"图像已保存：{path}")


# =========================
# 3️⃣ 主函数
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

    # ========= 特征 =========
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

    # ========= 划分 =========
    train, val, test = split_dataset(df)

    # ========= 归一化 =========
    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = \
        normalize_train_val_test(train, val, test, feature_cols, target_col)

    # ========= 序列 =========
    seq_len = 90

    X_train, y_train = create_sequences(train_x, train_y, seq_len)
    X_val, y_val = create_sequences(val_x, val_y, seq_len)
    X_test, y_test = create_sequences(test_x, test_y, seq_len)

    # ========= DataLoader =========
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=64,
        shuffle=False
    )

    # ========= 模型 =========
    model = RNNModel(input_size=len(feature_cols))

    # ========= 训练 =========
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device=device
    )

    # ========= 画图 =========
    plot_loss(train_losses, val_losses)

    # ========= 测试 =========
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        pred = model(X_test_tensor).cpu().numpy()

    # ========= 反归一化 =========
    y_test_inv = inverse_transform_y(scaler_y, y_test)
    pred_inv = inverse_transform_y(scaler_y, pred)

    # ========= 指标 =========
    logger.info("===== 测试结果 =====")
    logger.info(f"RMSE: {rmse(y_test_inv, pred_inv):.4f}")
    logger.info(f"MAE : {mae(y_test_inv, pred_inv):.4f}")
    logger.info(f"MAPE: {mape(y_test_inv, pred_inv):.4f}")

    # ========= 保存模型 =========
    model_path = os.path.join(
        MODEL_DIR,
        f"RNN_{datetime.now().strftime('%m%d_%H%M')}.pth"
    )

    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存：{model_path}")


if __name__ == "__main__":
    main()