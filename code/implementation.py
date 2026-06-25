import math
import pickle as pkl
from itertools import accumulate
import os
os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"

import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SHIFT = 7
MODEL_PATH  = r"C:\Users\Student\Desktop\business_tycoon_prediction\resources\models\lgbm_shift_7.pkl"
TREES_PDF   = r"C:\Users\Student\Desktop\business_tycoon_prediction\resources\trees.pdf"

with open(MODEL_PATH, "rb") as f:
    model = pkl.load(f)

user_input = input("Enter the last 11 prices separated by commas: ").strip()
prices = [float(x.strip()) for x in user_input.split(",")]

if len(prices) < 11:
    print(f"Need at least 11 prices, got {len(prices)}. Please try again.")
    exit()

data = pd.DataFrame({"money": prices})

data["percentage_change1"] = data["money"].pct_change()
data["percentage_change2"] = data["money"].pct_change(2)
data["percentage_change5"] = data["money"].pct_change(5)

for lag in range(1, 4):
    data[f"percentage_change_lag{lag}"] = data["percentage_change1"].shift(lag)

roll = data["money"].rolling(10)
roll_std = roll.std().clip(lower=1e-6)
data["dist_from_max"] = (data["money"] - roll.max()) / roll_std
data["dist_from_min"] = (data["money"] - roll.min()) / roll_std
data["zscore_10"]     = (data["money"] - roll.mean()) / roll_std

data["vol_5"]  = data["percentage_change1"].rolling(5).std()
data["vol_10"] = data["percentage_change1"].rolling(10).std()

data["percent_change_over_3"] = data["percentage_change1"].rolling(3).sum()
data["percent_change_over_5"] = data["percentage_change1"].rolling(5).sum()

direction = data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
def _step(s, d):
    if d == 0:                                     return 0
    elif (s > 0 and d > 0) or (s < 0 and d < 0): return s + d
    else:                                          return d
data["streak"] = list(accumulate(direction, _step))

data = data.drop(columns=["money", "percentage_change1"]).dropna()

features = data.iloc[[-1]]
pred  = model.predict(features)[0]
proba = model.predict_proba(features)[0]

direction_label = "UP" if pred == 1 else "DOWN"
confidence = proba[int(pred)]

print(f"\nPrediction (shift={SHIFT}): {direction_label}")
print(f"Confidence: {confidence * 100:.1f}%")
print(f"  P(Up):   {proba[1] * 100:.1f}%")
print(f"  P(Down): {proba[0] * 100:.1f}%")

# n_trees = model.booster_.num_trees()
# print(f"\nSaving {n_trees} trees to {TREES_PDF} ...")

# with PdfPages(TREES_PDF) as pdf:
#     for i in range(n_trees):
#         fig, ax = plt.subplots(figsize=(20, 10))
#         lgb.plot_tree(model.booster_, tree_index=i, ax=ax, show_info=["split_gain", "leaf_count", "data_count"])
#         ax.set_title(f"Tree {i}")
#         pdf.savefig(fig, bbox_inches="tight")
#         plt.close(fig)
#         print(f"  [{i+1}/{n_trees}] saved", end="\r")

# print(f"\nDone — {TREES_PDF}")