import pickle as pkl
import pandas as pd
import matplotlib as plt
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

predict = []

shift = 7

with open(r"C:\Users\Student\Desktop\business_tycoon_prediction\resources\models\rf_shift_7.pkl", "rb") as file:
    loaded_model = pkl.load(file)


data = pd.read_csv("resources/data/test_data.csv")
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

for lag in range(1, 4):
    data[f"money_lag{lag}"] = data["money"].shift(lag)

data.columns = data.columns.str.strip()
data = data.dropna()

data["future_price"] = data["money"].shift(-shift)
data["target"] = data["future_price"] > data["money"]
data = data.dropna()

data.to_csv(f"resources/data/model_data_shift_{shift}.csv")

labels = data["target"]
features = data.drop(["target", "future_price", "direction", "time"], axis=1)

print(f"processed data {data}")
print(f"features {features}")

predict = loaded_model.predict(features)

predictp = loaded_model.predict_proba(features)

print(predictp)


print(f"Accuracy Score: {accuracy_score(predict, labels)}")

print(f"Confusion Matrix: {confusion_matrix(predict, labels)}")

