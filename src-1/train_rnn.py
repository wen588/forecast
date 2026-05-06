# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils.common import *
from utils.log import Logger
from models import RNNModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
FIG_DIR = os.path.join(BASE_DIR, "data/fig")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

logger = Logger(BASE_DIR, "rnn_train").get_logger()


# =========================
def train_model(model, train_loader, val_loader, epochs=30, lr=0.0005, device="cuda"):

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    best_loss = float("inf")
    best_state = None
    best_path = os.path.join(MODEL_DIR, "RNN_best.pth")

    train_l, val_l = [], []

    for epoch in range(epochs):

        # ===== train =====
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

        # ===== val =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device).float(), y.to(device).float()
                val_loss += criterion(model(x), y).item()

        val_loss /= len(val_loader)

        train_l.append(train_loss)
        val_l.append(val_loss)

        logger.info(f"[RNN] Epoch {epoch+1} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, best_path)

    model.load_state_dict(best_state)
    return model, train_l, val_l


# =========================
# ⭐论文级评估指标（重点）
# =========================
def evaluate(model, X_test, y_test, scaler_y, device, name="RNN"):

    model.eval()

    with torch.no_grad():
        pred = model(torch.tensor(X_test, dtype=torch.float32).to(device)).cpu().numpy()

    y_true = inverse_transform_y(scaler_y, y_test)
    pred = inverse_transform_y(scaler_y, pred)

    logger.info("===== 测试结果 =====")
    logger.info(f"[{name}]")
    logger.info(f"RMSE: {rmse(y_true, pred):.4f}")
    logger.info(f"MAE : {mae(y_true, pred):.4f}")
    logger.info(f"MAPE: {mape(y_true, pred):.4f}")

    return y_true, pred


# =========================
def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    train, val, test = split_dataset(df)

    train_x, train_y, val_x, val_y, test_x, test_y, sx, sy = \
        normalize_train_val_test(train, val, test, feature_cols, "负荷")

    X_train, y_train = create_sequences(train_x, train_y, 90)
    X_val, y_val = create_sequences(val_x, val_y, 90)
    X_test, y_test = create_sequences(test_x, test_y, 90)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), 64, True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val), torch.tensor(y_val)), 64, False)

    model = RNNModel(len(feature_cols))

    model, tr, vl = train_model(model, train_loader, val_loader, device=device)

    # ⭐最终论文指标输出
    evaluate(model, X_test, y_test, sy, device, "RNN")


if __name__ == "__main__":
    main()