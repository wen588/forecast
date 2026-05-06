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

from utils.common import *
from utils.log import Logger   # ⭐引入日志

# =========================
# ⭐中文输出支持
# =========================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# ⭐路径管理
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIG_DIR = os.path.join(BASE_DIR, "data", "fig")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# ⭐日志初始化
# =========================
logger = Logger(BASE_DIR, "lstm_train").get_logger()


# =========================
# 1. LSTM模型
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.norm(out)
        return self.fc(out)


# =========================
# 2. 训练函数（日志版）
# =========================
def train_model(model, train_loader, val_loader, epochs=50, lr=0.0005, device='cuda'):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    train_losses, val_losses = [], []

    best_loss = float("inf")
    best_model = model.state_dict()

    for epoch in range(epochs):

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

        # ===== 验证 =====
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

        # ===== 保存最佳模型 =====
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model.state_dict()

            best_path = os.path.join(
                MODEL_DIR,
                f"LSTM最佳模型_{datetime.now().strftime('%m%d_%H%M')}.pth"
            )

            torch.save(best_model, best_path)
            logger.info(f"保存最佳模型: {best_path}")

    model.load_state_dict(best_model)

    return model, train_losses, val_losses


# =========================
# ⭐画图
# =========================
def plot_loss(train_losses, val_losses):

    plt.figure()

    plt.plot(train_losses, label="训练损失")
    plt.plot(val_losses, label="验证损失")

    plt.title("训练过程损失变化")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    path = os.path.join(
        FIG_DIR,
        f"训练损失曲线_{datetime.now().strftime('%m%d_%H%M')}.png"
    )

    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"已保存图像: {path}")


# =========================
# 3. 主函数
# =========================
def main():

    logger.info("===== 开始训练 LSTM =====")

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        "is_weekend", "is_holiday", "season",
        "负荷_lag1", "负荷_lag24", "负荷_lag168",
        "roll_mean_24", "roll_std_24"
    ]

    target_col = "负荷"

    train, val, test = split_dataset(df)

    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = \
        normalize_train_val_test(train, val, test, feature_cols, target_col)

    seq_len = 90

    X_train, y_train = create_sequences(train_x, train_y, seq_len)
    X_val, y_val = create_sequences(val_x, val_y, seq_len)
    X_test, y_test = create_sequences(test_x, test_y, seq_len)

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        ),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        ),
        batch_size=64,
        shuffle=False
    )

    model = LSTMModel(len(feature_cols))

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device=device
    )

    plot_loss(train_losses, val_losses)

    # ========= 测试 =========
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    y_test_inv = inverse_transform_y(scaler_y, y_test)
    pred_inv = inverse_transform_y(scaler_y, pred)

    logger.info("===== 测试结果 =====")
    logger.info(f"RMSE: {rmse(y_test_inv, pred_inv):.4f}")
    logger.info(f"MAE : {mae(y_test_inv, pred_inv):.4f}")
    logger.info(f"MAPE: {mape(y_test_inv, pred_inv):.4f}")


if __name__ == "__main__":
    main()