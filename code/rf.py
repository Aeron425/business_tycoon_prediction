from sklearn.ensemble import RandomForestClassifier
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
SHIFTS = [5, 7]


def process_data(shift=1):
    data = pd.read_csv("resources/data/data.csv")
    data.columns = data.columns.str.strip()
    data["money"] = data["money"].astype(int)
    data = data[(data["money"] >= 13) & (data["money"] <= 17)]

    data["percentage_change1"] = data["money"].pct_change()
    data["percentage_change2"] = data["money"].pct_change(2)
    data["percentage_change5"] = data["money"].pct_change(5)

    for lag in range(1, 3):
        data[f"percentage_change_lag{lag}"] = data["percentage_change1"].shift(lag)

    roll = data["money"].rolling(10)
    roll_std = roll.std()
    roll_std = roll_std.where(roll_std > 0, 0.000001)

    data["dist_from_max"] = (data["money"] - roll.max()) / roll_std
    data["dist_from_min"] = (data["money"] - roll.min()) / roll_std
    data["zscore_10"]     = (data["money"] - roll.mean()) / roll_std

    data["vol_5"] = data["percentage_change1"].rolling(5).std()
    data["vol_10"] = data["percentage_change1"].rolling(10).std()

    data["percent_change_over_3"] = data["percentage_change1"].rolling(3).sum()
    data["percent_change_over_5"] = data["percentage_change1"].rolling(5).sum()

    direction = (
        data["money"].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    )
    streak = []
    s = 0
    for d in direction:
        if d == 0:
            s = 0
        elif (s > 0 and d > 0) or (s < 0 and d < 0):
            s += d
        else:
            s = d
        streak.append(s)
    data["streak"] = streak

    data["future_price"] = data["money"].shift(-shift)
    data["target"] = (data["future_price"] > data["money"]).astype(int)
    data = data.dropna()

    features = data.drop(columns=["target", "future_price", "time", "money", "percentage_change1"])
    labels = data["target"]

    data.to_csv(f"resources/data/model_data_shift_{shift}.csv", index=False)
    return labels, features


def time_split(labels, features, test_ratio=0.2):
    n = len(features)
    split = int(n * (1 - test_ratio))
    return (
        features.iloc[:split],
        features.iloc[split:],
        labels.iloc[:split],
        labels.iloc[split:],
    )


def rf_eval(labels, features, n_estimators, max_depth, min_samples_leaf):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_leaf=int(min_samples_leaf),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return cross_val_score(model, features, labels, cv=CV, scoring="accuracy").mean()


def optimise(labels, features, n_iter=30, init_points=10):
    opt = BayesianOptimization(
        f=lambda n_estimators, max_depth, min_samples_leaf: rf_eval(
            labels, features, n_estimators, max_depth, min_samples_leaf
        ),
        pbounds={
            "n_estimators": (50, 500),
            "max_depth": (3, 15),
            "min_samples_leaf": (1, 50),
        },
        random_state=42,
        verbose=2,
    )
    opt.maximize(init_points=init_points, n_iter=n_iter)
    return opt.max["params"]


def evaluate(labels, features, best_params, shift):
    os.makedirs("resources/models", exist_ok=True)

    X_train, X_test, y_train, y_test = time_split(labels, features)

    model = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        min_samples_leaf=int(best_params["min_samples_leaf"]),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    with open(f"resources/models/rf_shift_{shift}.pkl", "wb") as f:
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
    plt.savefig("resources/plots/shift_comparison.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    for i, (shift, model, X_train, X_test, y_test, preds, proba) in enumerate(results):
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        sorted_idx = perm.importances_mean.argsort()
        ax = axes[i] if n > 1 else axes
        ax.barh(
            X_test.columns[sorted_idx],
            perm.importances_mean[sorted_idx],
            xerr=perm.importances_std[sorted_idx],
            color="steelblue",
            ecolor="gray",
            capsize=3,
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"Permutation Importance (shift={shift})")
        ax.set_xlabel("Mean accuracy decrease")

    plt.tight_layout()
    plt.savefig("resources/plots/perm_importance.png", dpi=120)
    plt.close()

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 4))
    for i, (shift, model, X_train, X_test, y_test, preds, proba) in enumerate(results):
        correct = (preds == y_test.values).astype(int)
        ax = axes[i] if n > 1 else axes
        ax.scatter(proba, correct, alpha=0.15, s=8)
        bins = np.linspace(0, 1, 11)
        bin_idx = np.digitize(proba, bins) - 1
        for b in range(10):
            mask = bin_idx == b
            if mask.sum() > 5:
                ax.plot(
                    bins[b] + 0.05,
                    correct[mask].mean(),
                    "ro",
                    markersize=6,
                )
        ax.set_xlabel("Predicted probability (Up)")
        ax.set_ylabel("Correct (1) / Wrong (0)")
        ax.set_title(f"Calibration check (shift={shift})")

    plt.tight_layout()
    plt.savefig("resources/plots/calibration.png", dpi=120)
    plt.close()
    print("Plots saved to resources/plots/")


results = []
for shift in SHIFTS:
    print(f"\n{'=' * 50}\nshift={shift}\n{'=' * 50}")
    labels, features = process_data(shift=shift)
    print(f"Features: {list(features.columns)}")
    print(f"Class balance — Up: {labels.mean():.2%}  Down: {(1 - labels.mean()):.2%}")
    best_params = optimise(labels, features)
    model, X_train, X_test, y_test, preds, proba = evaluate(
        labels, features, best_params, shift
    )
    results.append((shift, model, X_train, X_test, y_test, preds, proba))

plot_results(results)