# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.common import *
from utils.log import Logger
from models import RNNModel

# =========================
# 路径设置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FIG_DIR = os.path.join(BASE_DIR, "data/fig")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logger = Logger(BASE_DIR, "rnn_train").get_logger()


# =========================
# 模型训练函数
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_loss = float("inf")
    best_state = None
    best_path = os.path.join(MODEL_DIR, "RNN_best.pth")

    train_l, val_l = [], []

    for epoch in range(1, epochs + 1):
        # ===== 训练 =====
        model.train()
        train_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ===== 验证 =====
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).float()
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        train_l.append(train_loss)
        val_l.append(val_loss)

        logger.info(f"[RNN] Epoch {epoch} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, best_path)

    # 加载最优模型
    model.load_state_dict(best_state)

    # ⭐ 打印最优模型信息
    print("\n===== 训练完成 =====")
    print(f"✔ 最优验证损失: {best_loss:.6f}")
    print(f"✔ 最优模型已保存至: {best_path}\n")

    return model, train_l, val_l


# =========================
# 论文级评估函数
def evaluate(model, X_test, y_test, scaler_y, device, name="RNN"):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    y_true = inverse_transform_y(scaler_y, y_test)
    pred = inverse_transform_y(scaler_y, pred)

    # 计算指标
    rmse_val = rmse(y_true, pred)
    mae_val = mae(y_true, pred)
    mape_val = mape(y_true, pred)

    # 日志输出
    logger.info("===== 测试结果 =====")
    logger.info(f"[{name}] RMSE: {rmse_val:.4f} | MAE: {mae_val:.4f} | MAPE: {mape_val:.4f}")

    # ⭐ 终端打印
    print("===== 测试结果 =====")
    print(f"[{name}]")
    print(f"RMSE : {rmse_val:.4f}")
    print(f"MAE  : {mae_val:.4f}")
    print(f"MAPE : {mape_val:.4f}\n")

    return y_true, pred, rmse_val, mae_val, mape_val


# =========================
# 绘制训练损失曲线
def plot_loss(train_l, val_l):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(train_l, label="Train Loss")
    plt.plot(val_l, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RNN Training Loss Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "RNN_loss.png"), dpi=300)
    plt.close()


# =========================
# 主函数
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== 数据预处理 =====
    df = load_data("../data/train.xlsx")
    df = fill_missing(df)
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_season_feature(df)
    df = add_lag_features(df, "负荷")

    feature_cols = [
        "最高温度℃","最低温度℃","平均温度℃",
        "相对湿度(平均)","降雨量（mm）",
        "hour_sin","hour_cos",
        "weekday_sin","weekday_cos",
        "is_weekend","is_holiday","season",
        "负荷_lag1","负荷_lag24","负荷_lag168",
        "roll_mean_24","roll_std_24"
    ]

    # ===== 划分数据集 =====
    train, val, test = split_dataset(df)
    train_x, train_y, val_x, val_y, test_x, test_y, sx, sy = \
        normalize_train_val_test(train, val, test, feature_cols, "负荷")

    # ===== 创建序列 =====
    seq_len = 90
    X_train, y_train = create_sequences(train_x, train_y, seq_len)
    X_val, y_val = create_sequences(val_x, val_y, seq_len)
    X_test, y_test = create_sequences(test_x, test_y, seq_len)

    # ===== 数据加载器 =====
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64, shuffle=False)

    # ===== 模型训练 =====
    model = RNNModel(len(feature_cols))
    model, train_l, val_l = train_model(model, train_loader, val_loader, device=device)

    # ===== 绘制损失曲线 =====
    plot_loss(train_l, val_l)
    print("✔ RNN损失曲线已生成\n")

    # ===== 论文级指标 =====
    y_true, pred, rmse_val, mae_val, mape_val = evaluate(model, X_test, y_test, sy, device, "RNN")


if __name__ == "__main__":
    main()