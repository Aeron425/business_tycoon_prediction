from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os


CV = StratifiedKFold(n_splits=5, shuffle=False)
SHIFTS = [5, 7]  # how many steps ahead to predict


def process_data(shift=1):
    data = pd.read_csv("resources/data/data.csv")
    data.columns = data.columns.str.strip()
    data["money"] = data["money"].astype(int)
    data = data[(data["money"] >= 13) & (data["money"] <= 17)]  # filter outliers

    # percentage change over different time periods
    data["percentage_change1"] = data["money"].pct_change()
    data["percentage_change2"] = data["money"].pct_change(2)
    data["percentage_change5"] = data["money"].pct_change(5)

    # lagged returns so the model can see recent momentum
    for lag in range(1, 4):
        data[f"percentage_change_lag{lag}"] = data["percentage_change1"].shift(lag)

    # position relative to recent 10 minute range
    roll = data["money"].rolling(10)
    roll_std = roll.std()
    roll_std = roll_std.where(roll_std > 0, 0.000001)  # avoids division by zero

    data["dist_from_max"] = (data["money"] - roll.max()) / roll_std
    data["dist_from_min"] = (data["money"] - roll.min()) / roll_std
    data["zscore_10"]     = (data["money"] - roll.mean()) / roll_std

    # short and long term volatility
    data["vol_5"]  = data["percentage_change1"].rolling(5).std()
    data["vol_10"] = data["percentage_change1"].rolling(10).std()

    # cumulative return over recent windows
    data["percent_change_over_3"] = data["percentage_change1"].rolling(3).sum()
    data["percent_change_over_5"] = data["percentage_change1"].rolling(5).sum()

    # how many consecutive up or down moves in a row
    direction = data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    streak, s = [], 0
    for d in direction:
        if d == 0:                                     s = 0
        elif (s > 0 and d > 0) or (s < 0 and d < 0): s += d
        else:                                          s = d
        streak.append(s)
    data["streak"] = streak

    # defines a boolean to measure if the prediction is correct
    # needs future data as we are predicting in the future
    data["future_price"] = data["money"].shift(-shift)
    data["target"] = (data["future_price"] > data["money"]).astype(int)
    data = data.dropna()

    features = data.drop(columns=["target", "future_price", "time", "money", "percentage_change1"])
    labels = data["target"]

    data.to_csv(f"resources/data/model_data_shift_{shift}.csv", index=False)
    return labels, features


def time_split(labels, features, test_ratio=0.2):
    # chronological split
    n = len(features)
    split = int(n * (1 - test_ratio))
    return (
        features.iloc[:split],
        features.iloc[split:],
        labels.iloc[:split],
        labels.iloc[split:],
    )


def lgbm_eval(labels, features, n_estimators, max_depth, num_leaves, min_child_samples, scale_pos_weight):
    model = LGBMClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        min_child_samples=int(min_child_samples),
        scale_pos_weight=float(scale_pos_weight),
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    return cross_val_score(model, features, labels, cv=CV, scoring="accuracy").mean()


def optimise(labels, features, n_iter=40, init_points=10):
    # bayesian optimisation finds good hyperparams without brute forcing a grid
    opt = BayesianOptimization(
        f=lambda n_estimators, max_depth, num_leaves, min_child_samples, scale_pos_weight: lgbm_eval(
            labels, features, n_estimators, max_depth, num_leaves, min_child_samples, scale_pos_weight
        ),
        pbounds={
            "n_estimators":      (50, 500),
            "max_depth":         (3, 15),
            "num_leaves":        (8, 128),
            "min_child_samples": (5, 100),
            "scale_pos_weight":  (1.0, 10.0),
        },
        random_state=42,
        verbose=2,
    )
    opt.maximize(init_points=init_points, n_iter=n_iter)
    return opt.max["params"]


def evaluate(labels, features, best_params, shift):
    os.makedirs("resources/models", exist_ok=True)

    X_train, X_test, y_train, y_test = time_split(labels, features)

    model = LGBMClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        num_leaves=int(best_params["num_leaves"]),
        min_child_samples=int(best_params["min_child_samples"]),
        scale_pos_weight=best_params["scale_pos_weight"],
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    with open(f"resources/models/lgbm_shift_{shift}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"\n[shift={shift}] Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Majority-class baseline: {baseline * 100:.2f}%")

    return model, X_train, X_test, y_test, preds, proba


def plot_results(results):
    os.makedirs("resources/plots", exist_ok=True)
    n = len(results)

    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    for i, (shift, model, X_train, X_test, y_test, preds, proba) in enumerate(results):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, preds), display_labels=["Down", "Up"]
        ).plot(ax=axes[i, 0], cmap="Blues", colorbar=False)
        axes[i, 0].set_title(f"Confusion Matrix (shift={shift})")

        imp = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()
        axes[i, 1].barh(imp.index, imp.values)
        axes[i, 1].set_xlabel("Importance")
        axes[i, 1].set_title(f"Feature Importances (shift={shift})")

    plt.tight_layout()
    plt.savefig("resources/plots/lgbm_shift_comparison.png", dpi=120)
    plt.close()


    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    for i, (shift, model, X_train, X_test, y_test, preds, proba) in enumerate(results):
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
        sorted_idx = perm.importances_mean.argsort()
        ax = axes[i] if n > 1 else axes
        ax.barh(
            X_test.columns[sorted_idx],
            perm.importances_mean[sorted_idx],
            xerr=perm.importances_std[sorted_idx],
            color="steelblue", ecolor="gray", capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Permutation Importance (shift={shift})")
        ax.set_xlabel("Mean accuracy decrease")

    plt.tight_layout()
    plt.savefig("resources/plots/lgbm_perm_importance.png", dpi=120)
    plt.close()


results = []
for shift in SHIFTS:
    print(f"\n{'=' * 50}\nshift={shift}\n{'=' * 50}")
    labels, features = process_data(shift=shift)
    print(f"Features: {list(features.columns)}")
    print(f"Class balance — Up: {labels.mean():.2%}  Down: {(1 - labels.mean()):.2%}")
    best_params = optimise(labels, features)
    model, X_train, X_test, y_test, preds, proba = evaluate(labels, features, best_params, shift)
    results.append((shift, model, X_train, X_test, y_test, preds, proba))

plot_results(results)