import pandas
import numpy

data = pandas.read_csv('data.csv')

data["Money"] = data["Money"].astype(int)
data["Percentage Change"] = data["Money"].pct_change() * 100.
data["Percentage Change"] = data["Percentage Change"].astype(float).apply(lambda x : round(x, 2))
data["Direction"] = data["Money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

data["Streak"] = 0
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


data.to_csv('processed_data.csv')
    