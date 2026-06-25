import pickle as pkl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

shift = 7
threshold = 0.65
trade_size = 1.0
model_path = "/home/gavin/Desktop/coding_projects/python/business_tycoon_prediction/resources/models/rf_shift_7.pkl"
csv = "resources/data/data.csv"


def load_and_process(path):
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()
    data["money"] = data["money"].astype(int)

    data["ret1"] = data["money"].pct_change()
    data["ret2"] = data["money"].pct_change(2)
    data["ret5"] = data["money"].pct_change(5)

    for lag in range(1, 4):
        data[f"ret_lag{lag}"] = data["ret1"].shift(lag)

    roll = data["money"].rolling(10)
    data["dist_from_max"] = (data["money"] - roll.max()) / roll.std().clip(lower=1e-6)
    data["dist_from_min"] = (data["money"] - roll.min()) / roll.std().clip(lower=1e-6)
    data["zscore_10"] = (data["money"] - roll.mean()) / roll.std().clip(lower=1e-6)
    data["vol_5"] = data["ret1"].rolling(5).std()
    data["vol_10"] = data["ret1"].rolling(10).std()
    data["mom_3"] = data["ret1"].rolling(3).sum()
    data["mom_5"] = data["ret1"].rolling(5).sum()

    streak, s = [], 0
    for d in data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)):
        s = 0 if d == 0 else (s + d if (s > 0) == (d > 0) else d)
        streak.append(s)
    data["streak"] = streak

    data["future_price"] = data["money"].shift(-shift)
    data = data.dropna(subset=["future_price"])

    features = data.drop(
        columns=["money", "ret1", "future_price", "time"], errors="ignore"
    ).dropna()
    data = data.loc[features.index]

    return data["money"].values, data["future_price"].values, features


def run_backtest(prices, future_prices, up_proba, down_proba):
    pnl_curve, trades = [0.0], []
    position, entry, running = None, None, 0.0

    for i, price in enumerate(prices):

        if up_proba[i] >= threshold:
            signal = "buy"
        elif down_proba[i] >= threshold:
            signal = "sell"
        else:
            signal = "hold"
        
        if signal == "buy" and position == "short":
            pnl = (entry - price) * trade_size
            running += pnl
            trades.append({"entry": entry, "exit": price, "pnl": pnl})
            position = None
        elif signal == "sell" and position == "long":
            pnl = (price - entry) * trade_size
            running += pnl
            trades.append({"entry": entry, "exit": price, "pnl": pnl})
            position = None

        if signal == "buy" and position is None:
            position, entry = "long", price
        if signal == "sell" and position is None:
            position, entry = "short", price

        if position == "long":
            pnl_curve.append(running + (price - entry) * trade_size)
        elif position == "short":
            pnl_curve.append(running + (entry - price) * trade_size)
        else:
            pnl_curve.append(running)

    if position == "long":
        pnl = (prices[-1] - entry) * trade_size
        running += pnl
        trades.append({"entry": entry, "exit": prices[-1], "pnl": pnl})
    elif position == "short":
        pnl = (entry - prices[-1]) * trade_size
        running += pnl
        trades.append({"entry": entry, "exit": prices[-1], "pnl": pnl})

    return pnl_curve, trades


def calibration_data(up_proba, actual, n_bins=10):
    preds = (up_proba >= 0.5).astype(int)
    correct = (preds == actual).astype(int)
    conf = np.where(up_proba >= 0.5, up_proba, 1 - up_proba)

    bins = np.linspace(0.5, 1.0, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    acc, cnt = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        acc.append(correct[mask].mean() if mask.sum() >= 5 else np.nan)
        cnt.append(int(mask.sum()))

    return centres, acc, cnt


def print_stats(pnl_curve, trades, n_steps, up_proba, down_proba):
    closed = [t["pnl"] for t in trades]
    wins = [p for p in closed if p > 0]
    losses = [p for p in closed if p <= 0]
    fired = int(
        ((up_proba >= threshold) | (down_proba >= threshold)).sum()
    )

    print(f"\n{'=' * 45}")
    print(f"  BACKTEST  (threshold={threshold})")
    print(f"{'=' * 45}")
    print(f"Total steps:     {n_steps}")
    print(f"Signals fired:   {fired}  ({fired / n_steps * 100:.1f}% of steps)")
    print(f"Trades closed:   {len(closed)}")
    print(
        f"Win rate:        {len(wins) / len(closed) * 100:.1f}%"
        if closed
        else "Win rate: n/a"
    )
    print(f"Avg win:         {np.mean(wins):+.2f}" if wins else "Avg win:  n/a")
    print(f"Avg loss:        {np.mean(losses):+.2f}" if losses else "Avg loss: n/a")
    print(f"Total PnL:       {pnl_curve[-1]:+.2f} price units")
    print(f"{'=' * 45}\n")


def plot_results(pnl_curve, trades, centres, acc, cnt):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(
        "Backtest",
        fontsize=13,
        fontweight="bold",
        color="darkred",
    )

    axes[0].plot(pnl_curve, color="steelblue")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Profit (price units)")
    axes[0].set_xlabel("Step")

    pnls = [t["pnl"] for t in trades]
    colors = ["green" if p > 0 else "red" for p in pnls]
    axes[1].bar(range(len(pnls)), pnls, color=colors)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Per-Trade PnL")
    axes[1].set_ylabel("Profit (price units)")
    axes[1].set_xlabel("Trade #")

    axes[2].bar(
        centres,
        cnt,
        width=0.04,
        alpha=0.3,
        color="steelblue",
        label="# predictions in bucket",
    )
    ax2 = axes[2].twinx()
    ax2.plot(centres, acc, "ro-", label="Actual accuracy in bucket")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random (50%)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Accuracy")
    axes[2].set_xlabel("Model confidence")
    axes[2].set_ylabel("# predictions")
    axes[2].set_title("Calibration")
    axes[2].legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("resources/plots/backtest.png", dpi=120)
    plt.close()
    print("Saved → resources/plots/backtest.png")



with open(model_path, "rb") as f:
    model = pkl.load(f)

prices, future_prices, features = load_and_process(csv)

proba = model.predict_proba(features)
up_proba = proba[:, 1]
down_proba = proba[:, 0]

pnl_curve, trades = run_backtest(prices, future_prices, up_proba, down_proba)

actual = (future_prices > prices).astype(int)
centres, accuracy, count = calibration_data(up_proba, actual)

print_stats(pnl_curve, trades, len(features), up_proba, down_proba)
plot_results(pnl_curve, trades, centres, accuracy, count)
