import pickle as pkl
import pandas as pd
import matplotlib as plt
import os

shift = 7

with open(r"C:\Users\Student\Desktop\business_tycoon_prediction\resources\models\rf_shift_5.pkl", "rb") as file:
    loaded_model = pkl.load(file)

user_input = input("Input price for last six minutes seperated by commas ").strip()
list_of_input = user_input.split(",")
print(list_of_input)

data = pd.DataFrame({"money": list_of_input})
print(f"data = {data}")

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
    data[f"direction_lag{lag}"] = data["direction"].shift(lag)

data["rolling_std_5"] = data["money"].rolling(5).std()
data.columns = data.columns.str.strip()
data = data.dropna()

features = data.drop(["direction"], axis=1)

print(f"processed data {data}")
print(f"features {features}")

print(loaded_model.predict(features))

