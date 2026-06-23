import pandas as pd

data = pd.read_csv('resources/data/data.csv')
data.columns = data.columns.str.strip()
data["Money"] = data["Money"].astype(int)

data["Percentage Change"] = data["Money"].pct_change().mul(100).round(2)
data["Direction"] = data["Money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

streak = 0
for i in range(1, len(data)):
    if data["Direction"].iloc[i] == data["Direction"].iloc[i - 1]:
        if data["Direction"].iloc[i] == 1:
            streak += 1
        elif data["Direction"].iloc[i] == -1:
            streak -= 1
    else:
        streak = data["Direction"].iloc[i]
    data.loc[i, "Streak"] = streak

for lag in range(1, 6):
    data[f'Money_lag{lag}'] = data['Money'].shift(lag)
    data[f'Direction_lag{lag}'] = data['Direction'].shift(lag)
    data[f'Streak_lag{lag}'] = data['Streak'].shift(lag)

data['Rolling_mean_5'] = data['Money'].rolling(5).mean()
data['Rolling_std_5'] = data['Money'].rolling(5).std()
data['Rolling_min_5'] = data['Money'].rolling(5).min()
data['Rolling_max_5'] = data['Money'].rolling(5).max()

data.to_csv('resources/data/processed_data.csv', index=False)