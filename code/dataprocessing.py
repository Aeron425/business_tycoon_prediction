import pandas as pd

data = pd.read_csv("resources/data/raw_data.csv")
data.columns = data.columns.str.strip()
data["money"] = data["money"].astype(int)

data["percentage_change"] = data["money"].pct_change().mul(100).round(2)
data["direction"] = data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

streak = 0
for i in range(1, len(data)):
    if data["direction"].iloc[i] == data["direction"].iloc[i - 1]:
        if data["direction"].iloc[i] == 1:
            streak += 1
        elif data["direction"].iloc[i] == -1:
            streak -= 1
    else:
        streak = data["direction"].iloc[i]
    data.loc[i, "streak"] = streak

for lag in range(1, 6):
    data[f"money_lag{lag}"] = data["money"].shift(lag)
    data[f"streak_lag{lag}"] = data["streak"].shift(lag)

data.to_csv("resources/data/readable_data.csv", index=False)

data["rolling_mean_5"] = data["money"].rolling(5).mean()
data["rolling_std_5"] = data["money"].rolling(5).std()
data["rolling_min_5"] = data["money"].rolling(5).min()
data["rolling_max_5"] = data["money"].rolling(5).max()

data.columns = data.columns.str.strip()
data["direction"] = data["direction"].shift(-1)
data = data.dropna()
data = data[data["direction"] != 0]

data.to_csv("resources/data/model_ready_data.csv", index=False)