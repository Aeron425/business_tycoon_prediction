from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SHIFTS = [3, 5, 7, 10]


def process_data(shift=1):
    data = pd.read_csv("resources/data/data.csv")
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
        data.loc[i, "Streak"] = streak

    for lag in range(1, 6):
        data[f"money_lag{lag}"] = data["money"].shift(lag)
        data[f"direction_lag{lag}"] = data["direction"].shift(lag)
        data[f"Streak_lag{lag}"] = data["Streak"].shift(lag)

    data["rolling_std_5"] = data["money"].rolling(5).std()
    data.columns = data.columns.str.strip()
    data = data.dropna()

    data["future_price"] = data["money"].shift(-shift)
    data["target"] = data["future_price"] > data["money"]
    data = data.dropna()

    data.to_csv(f"resources/data/model_data_shift_{shift}.csv")

    labels = data["target"]
    features = data.drop(["target", "future_price", "direction", "time"], axis=1)
    return labels, features


def rf_eval(labels, features, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    return cross_val_score(model, features, labels, cv=CV, scoring="accuracy").mean()


def optimise(labels, features, n_iter=30, init_points=10):
    opt = BayesianOptimization(
        f=lambda n_estimators, max_depth: rf_eval(labels, features, n_estimators, max_depth),
        pbounds={"n_estimators": (50, 500), "max_depth": (3, 15)},
        random_state=42,
        verbose=2,
    )
    opt.maximize(init_points=init_points, n_iter=n_iter)
    return opt.max["params"]


def evaluate(labels, features, best_params, shift):
    os.makedirs("resources/models", exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    with open(f"resources/models/rf_shift_{shift}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    return model, X_train, X_test, y_test, preds


def plot_results(results):
    os.makedirs("resources/plots", exist_ok=True)
    n = len(results)

    fig, axes = plt.subplots(n, 2, figsize=(14, 5 * n))
    for i, (shift, model, X_train, X_test, y_test, preds) in enumerate(results):
        ConfusionMatrixDisplay(
            confusion_matrix(y_test, preds), display_labels=["Down", "Up"]
        ).plot(ax=axes[i, 0], cmap="Blues", colorbar=False)
        axes[i, 0].set_title(f"Confusion Matrix (shift={shift})")

        axes[i, 1].barh(X_train.columns, model.feature_importances_)
        axes[i, 1].set_xlabel("Importance")
        axes[i, 1].set_ylabel("Features")
        axes[i, 1].set_title(f"Feature Importances (shift={shift})")

    plt.tight_layout()
    plt.savefig("resources/plots/shift_comparison.png", dpi=120)
    plt.close()
    print("Saved to resources/plots/shift_comparison.png")

    fig, axes = plt.subplots(n, 1, figsize=(10, 5 * n))
    for i, (shift, model, X_train, X_test, y_test, preds) in enumerate(results):
        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)

        sorted_idx = perm.importances_mean.argsort()
        axes[i].barh(
            X_test.columns[sorted_idx],
            perm.importances_mean[sorted_idx],
            xerr=perm.importances_std[sorted_idx],
            color="steelblue",
            ecolor="gray",
            capsize=3,
        )
        axes[i].axvline(0, color="black", linewidth=0.8, linestyle="--")
        axes[i].set_title(f"Permutation Importance (shift={shift})")
        axes[i].set_xlabel("Mean accuracy decrease")

    plt.tight_layout()
    plt.savefig("resources/plots/perm_importance.png", dpi=120)
    plt.close()
    print("Saved to resources/plots/perm_importance.png")


results = []
for shift in SHIFTS:
    print(f"\n{"="*50}\nRunning shift={shift}\n{"="*50}")
    labels, features = process_data(shift=shift)
    best_params = optimise(labels, features)
    model, X_train, X_test, y_test, preds = evaluate(labels, features, best_params, shift)
    results.append((shift, model, X_train, X_test, y_test, preds))

plot_results(results)