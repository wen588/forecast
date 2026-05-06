# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os

# =========================
# 指标函数
# =========================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# =========================
# 创建输出目录
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_DIR = os.path.join(BASE_DIR, "data", "fig")
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 加载数据
# =========================
def load_data():
    y_true = np.load("y_true.npy")
    lstm = np.load("lstm.npy")
    rnn = np.load("rnn.npy")
    bpnn = np.load("bpnn.npy")
    return y_true, lstm, rnn, bpnn


# =========================
# 图1：预测对比图
# =========================
def plot_prediction(y_true, lstm, rnn, bpnn):

    n = 300  # ⭐ 防止太密
    y_true = y_true[:n]
    lstm = lstm[:n]
    rnn = rnn[:n]
    bpnn = bpnn[:n]

    plt.figure(figsize=(16, 6))

    plt.plot(y_true, label="True", linewidth=2)
    plt.plot(lstm, label="LSTM", linestyle="--")
    plt.plot(rnn, label="RNN", linestyle="-.")
    plt.plot(bpnn, label="BPNN", linestyle=":")

    plt.title("Load Forecast Comparison", fontsize=16)
    plt.xlabel("Time Step")
    plt.ylabel("Load")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "comparison.png"), dpi=300)
    plt.close()


# =========================
# 图2：误差对比
# =========================
def plot_error(y_true, lstm, rnn, bpnn):

    error_lstm = np.abs(y_true - lstm)
    error_rnn = np.abs(y_true - rnn)
    error_bpnn = np.abs(y_true - bpnn)

    n = 300
    error_lstm = error_lstm[:n]
    error_rnn = error_rnn[:n]
    error_bpnn = error_bpnn[:n]

    plt.figure(figsize=(16, 6))

    plt.plot(error_lstm, label="LSTM Error")
    plt.plot(error_rnn, label="RNN Error")
    plt.plot(error_bpnn, label="BPNN Error")

    plt.title("Prediction Error Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "error.png"), dpi=300)
    plt.close()


# =========================
# 图3：指标柱状图
# =========================
def plot_metrics(y_true, lstm, rnn, bpnn):

    models = ["LSTM", "RNN", "BPNN"]

    rmse_vals = [
        rmse(y_true, lstm),
        rmse(y_true, rnn),
        rmse(y_true, bpnn)
    ]

    mae_vals = [
        mae(y_true, lstm),
        mae(y_true, rnn),
        mae(y_true, bpnn)
    ]

    mape_vals = [
        mape(y_true, lstm),
        mape(y_true, rnn),
        mape(y_true, bpnn)
    ]

    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(10, 6))

    plt.bar(x - width, rmse_vals, width, label="RMSE")
    plt.bar(x, mae_vals, width, label="MAE")
    plt.bar(x + width, mape_vals, width, label="MAPE")

    plt.xticks(x, models)
    plt.ylabel("Value")
    plt.title("Model Performance Comparison")

    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "metrics.png"), dpi=300)
    plt.close()


def plot_zoom(y_true, lstm, rnn, bpnn):

    # ⭐ 选取局部区间（可以改）
    start, end = 200, 300

    plt.figure(figsize=(10, 5))

    plt.plot(y_true[start:end], label="True", linewidth=2)
    plt.plot(lstm[start:end], label="LSTM", linestyle="--")
    plt.plot(rnn[start:end], label="RNN", linestyle="-.")
    plt.plot(bpnn[start:end], label="BPNN", linestyle=":")

    plt.title("Local Load Forecast Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Load")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "zoom.png"), dpi=300)
    plt.close()


def plot_scatter(y_true, lstm, rnn, bpnn):

    plt.figure(figsize=(6, 6))

    plt.scatter(y_true, lstm, alpha=0.4, label="LSTM")
    plt.scatter(y_true, rnn, alpha=0.4, label="RNN")
    plt.scatter(y_true, bpnn, alpha=0.4, label="BPNN")

    # ⭐ 理想预测线 y=x
    min_val = min(y_true.min(), lstm.min(), rnn.min(), bpnn.min())
    max_val = max(y_true.max(), lstm.max(), rnn.max(), bpnn.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal")

    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Prediction Scatter Comparison")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "scatter.png"), dpi=300)
    plt.close()

# =========================
# 主函数
# =========================
def main():

    print("📊 开始生成可视化图...")

    y_true, lstm, rnn, bpnn = load_data()

    # 对齐长度
    min_len = min(len(y_true), len(lstm), len(rnn), len(bpnn))
    y_true = y_true[:min_len]
    lstm = lstm[:min_len]
    rnn = rnn[:min_len]
    bpnn = bpnn[:min_len]

    plot_prediction(y_true, lstm, rnn, bpnn)
    print("✔ 对比图完成")

    plot_error(y_true, lstm, rnn, bpnn)
    print("✔ 误差图完成")

    plot_metrics(y_true, lstm, rnn, bpnn)
    print("✔ 指标图完成")

    plot_zoom(y_true, lstm, rnn, bpnn)
    print("✔ 局部放大图完成")

    plot_scatter(y_true, lstm, rnn, bpnn)
    print("✔ 散点图完成")

    print(f"\n🎉 所有图已保存到: {SAVE_DIR}/")


# =========================
# 入口
# =========================
if __name__ == "__main__":
    main()