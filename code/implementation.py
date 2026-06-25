import pickle as pkl
import pandas as pd

SHIFT = 7
MODEL_PATH = "/home/gavin/Desktop/coding_projects/python/business_tycoon_prediction/resources/models/rf_shift_7.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pkl.load(f)

user_input = input("Enter the last 11 prices separated by commas: ").strip()
prices = [int(x.strip()) for x in user_input.split(",")]

if len(prices) < 11:
    print(f"Need at least 11 prices, got {len(prices)}. Please try again.")
    exit()

data = pd.DataFrame({"money": prices})

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

direction = data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
streak, s = [], 0
for d in direction:
    if d == 0:
        s = 0
    elif (s > 0 and d > 0) or (s < 0 and d < 0):
        s += d
    else:
        s = d
    streak.append(s)
data["streak"] = streak

data = data.drop(columns=["money", "ret1"])
data = data.dropna()

features = data.iloc[[-1]]
pred = model.predict(features)[0]
proba = model.predict_proba(features)[0]

direction_label = "UP" if pred == 1 else "DOWN"
confidence = proba[1] if pred == 1 else proba[0]

print(f"\nPrediction (shift={SHIFT}): {direction_label}")
print(f"Confidence: {confidence * 100:.1f}%")
print(f"  P(Up):   {proba[1] * 100:.1f}%")
print(f"  P(Down): {proba[0] * 100:.1f}%")
