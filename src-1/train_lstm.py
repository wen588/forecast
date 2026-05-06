# -*- coding: utf-8 -*-

import os
import sys
import io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from utils.common import *
from utils.log import Logger
from models import LSTMModel   # ⭐统一模型来源

# =========================
# 中文支持
# =========================
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# =========================
# 路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE_DIR, "data", "fig")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# 日志
# =========================
logger = Logger(BASE_DIR, "lstm_train").get_logger()

# =========================
# 模型（已统一在 models.py，这里不再重复）
# =========================


# =========================
# 训练函数（稳定版）
# =========================
def train_model(model, train_loader, val_loader, epochs=30, lr=0.0005, device="cuda"):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    train_losses, val_losses = [], []

    best_loss = float("inf")
    best_state = None

    best_path = os.path.join(MODEL_DIR, "LSTM_best.pth")  # ⭐固定名字

    patience = 5
    wait = 0

    for epoch in range(epochs):

        # ========= train =========
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ========= val =========
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).float()
                pred = model(x)
                val_loss += criterion(pred, y).item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"[LSTM] Epoch {epoch+1} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        # ========= 保存唯一最佳模型 =========
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()

            torch.save(best_state, best_path)
            logger.info(f"✔ 更新最佳模型: {best_path}")

            wait = 0
        else:
            wait += 1

        if wait >= patience:
            logger.info("早停触发")
            break

    model.load_state_dict(best_state)

    return model, train_losses, val_losses


# =========================
# 画图
# =========================
def plot_loss(train_losses, val_losses):

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")

    plt.title("LSTM Loss Curve")
    plt.legend()

    path = os.path.join(FIG_DIR, "LSTM_loss.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✔ 保存曲线: {path}")


# =========================
# 主函数
# =========================
def main():

    logger.info("===== LSTM训练开始 =====")

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
        "is_weekend", "is_holiday", "season",
        "负荷_lag1", "负荷_lag24", "负荷_lag168",
        "roll_mean_24", "roll_std_24"
    ]

    target_col = "负荷"

    # ========= 划分 =========
    train, val, test = split_dataset(df)

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
    model = LSTMModel(input_size=len(feature_cols))

    # ========= 训练 =========
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device=device
    )

    # ========= 曲线 =========
    plot_loss(train_losses, val_losses)

    # ========= 测试 =========
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    y_true = inverse_transform_y(scaler_y, y_test)
    pred = inverse_transform_y(scaler_y, pred)

    logger.info("===== 测试结果 =====")
    logger.info(f"RMSE: {rmse(y_true, pred):.4f}")
    logger.info(f"MAE : {mae(y_true, pred):.4f}")
    logger.info(f"MAPE: {mape(y_true, pred):.4f}")


if __name__ == "__main__":
    main()