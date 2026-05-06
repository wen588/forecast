# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from utils.common import *
from utils.log import Logger
from models import BPNNModel


# =========================
# 路径
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FIG_DIR = os.path.join(BASE_DIR, "data", "fig")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logger = Logger(BASE_DIR, "bpnn_train").get_logger()


# =========================
# ⭐训练函数
# =========================
def train_model(model, train_loader, val_loader, epochs=30, lr=0.001, device="cuda"):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_loss = float("inf")
    best_path = os.path.join(MODEL_DIR, "BPNN_best.pth")

    train_losses, val_losses = [], []

    for epoch in range(epochs):

        # ========= train =========
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()

            loss = criterion(model(x), y)

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
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"[BPNN] Epoch {epoch+1} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        # ========= 保存最佳模型 =========
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"✔ 更新最佳模型: {best_path}")

    return model, train_losses, val_losses


# =========================
# ⭐loss曲线
# =========================
def plot_loss(train_losses, val_losses):

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")

    plt.title("BPNN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    save_path = os.path.join(FIG_DIR, "BPNN_loss.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✔ loss曲线保存: {save_path}")


# =========================
# ⭐主函数
# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ========= 数据 =========
    df = load_data("../data/train.xlsx")
    df = fill_missing(df)
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_season_feature(df)
    df = add_lag_features(df, target="负荷")

    feature_cols = [
        "最高温度℃","最低温度℃","平均温度℃",
        "相对湿度(平均)","降雨量（mm）",
        "hour_sin","hour_cos",
        "weekday_sin","weekday_cos",
        "is_weekend","is_holiday","season",
        "负荷_lag1","负荷_lag24","负荷_lag168",
        "roll_mean_24","roll_std_24"
    ]

    train, val, test = split_dataset(df)

    train_x, train_y, val_x, val_y, test_x, test_y, sx, sy = \
        normalize_train_val_test(train, val, test, feature_cols, "负荷")

    # ========= BPNN 不用序列 =========
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x), torch.tensor(train_y)),
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(torch.tensor(val_x), torch.tensor(val_y)),
        batch_size=64,
        shuffle=False
    )

    model = BPNNModel(len(feature_cols))

    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, device=device
    )

    plot_loss(train_losses, val_losses)

    # =========================
    # ⭐测试 + 论文级输出
    # =========================
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(test_x, dtype=torch.float32).to(device)
        pred = model(X_test_tensor).cpu().numpy()

    y_true = inverse_transform_y(sy, test_y)
    pred = inverse_transform_y(sy, pred)

    rmse_val = rmse(y_true, pred)
    mae_val = mae(y_true, pred)
    mape_val = mape(y_true, pred)

    logger.info("\n========== BPNN 测试结果 ==========")
    logger.info(f"{'指标':<8}{'数值':>12}")
    logger.info("-" * 22)
    logger.info(f"{'RMSE':<8}{rmse_val:>12.4f}")
    logger.info(f"{'MAE':<8}{mae_val:>12.4f}")
    logger.info(f"{'MAPE':<8}{mape_val:>12.4f}%")
    logger.info("-" * 22)


if __name__ == "__main__":
    main()