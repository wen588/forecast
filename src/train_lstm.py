# LSTM训练脚本


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
from models import LSTMModel

# 中文支持
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FIG_DIR = os.path.join(BASE_DIR, "data", "fig")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logger = Logger(BASE_DIR, "lstm_train").get_logger()

# 训练函数
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device="cuda"):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    train_losses, val_losses = [], []
    best_loss = float('inf')
    best_state = None
    best_path = os.path.join(MODEL_DIR, "LSTM_best.pth")
    patience, wait = 5, 0

    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).float()
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(f"[LSTM] Epoch {epoch+1} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, best_path)
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            logger.info("早停触发")
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses

# 绘制训练曲线
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

# 主函数
def main():
    logger.info("===== LSTM训练开始 =====")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_data(os.path.join(BASE_DIR, "data/train.xlsx"))
    df = df.sort_values("时间").reset_index(drop=True)
    df = fill_missing(df)
    df = add_time_features(df)
    df = add_holiday_feature(df)
    df = add_season_feature(df)
    df = add_lag_features(df, target="负荷")

    feature_cols = ["最高温度℃","最低温度℃","平均温度℃","相对湿度(平均)","降雨量（mm）",
                    "hour_sin","hour_cos","weekday_sin","weekday_cos","is_weekend",
                    "is_holiday","season","负荷_lag1","负荷_lag24","负荷_lag168",
                    "roll_mean_24","roll_std_24"]

    target_col = "负荷"
    train, val, test = split_dataset(df)
    train_x, train_y, val_x, val_y, test_x, test_y, scaler_x, scaler_y = normalize_train_val_test(train, val, test,
                                                                                                  feature_cols, target_col)

    X_train, y_train = create_sequences(train_x, train_y, 90)
    X_val, y_val = create_sequences(val_x, val_y, 90)
    X_test, y_test = create_sequences(test_x, test_y, 90)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), batch_size=64, shuffle=False)

    mo
    del = LSTMModel(len(feature_cols))
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, device=device)
    plot_loss(train_losses, val_losses)

    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()
    y_true = inverse_transform_y(scaler_y, y_test)
    pred = inverse_transform_y(scaler_y, pred)
    logger.info(f"RMSE: {rmse(y_true, pred):.4f}")
    logger.info(f"MAE : {mae(y_true, pred):.4f}")
    logger.info(f"MAPE: {mape(y_true, pred):.4f}")

if __name__ == "__main__":
    main()

