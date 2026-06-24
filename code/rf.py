from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


CV = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
SHIFTS = [1, 3, 5, 7, 10, 15]


def process_data(filepath = "resources/data/processed_data.csv", shift = 1):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    data = data.dropna()

    data["future_price"] = data["money"].shift(-shift)
    data["target"] = data["future_price"] > data["money"]
    data = data.dropna()

    labels = data["target"]
    features = data.drop(["target", "future_price", "direction", "time"], axis = 1)
    return labels, features


def rf_eval(labels, features, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators = int(n_estimators),
        max_depth = int(max_depth),
        class_weight = "balanced",
        n_jobs = -1,
        random_state = 42,
    )
    return cross_val_score(model, features, labels, cv = CV, scoring = "accuracy").mean()


def optimise(labels, features, n_iter = 30, init_points = 10):
    opt = BayesianOptimization(
        f = lambda n_estimators, max_depth: rf_eval(labels, features, n_estimators, max_depth),
        pbounds = {"n_estimators": (50, 500), "max_depth": (3, 15)},
        random_state = 42,
        verbose = 2,
    )
    opt.maximize(init_points = init_points, n_iter = n_iter)
    return opt.max["params"]


def evaluate(labels, features, best_params, shift):
    os.makedirs("resources/models", exist_ok = True)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42, stratify = labels)

    model = RandomForestClassifier(
        n_estimators = int(best_params["n_estimators"]),
        max_depth = int(best_params["max_depth"]),
        class_weight = "balanced",
        n_jobs = -1,
        random_state = 42,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    with open(f"resources/models/rf_shift_{shift}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names = ["Down", "Up"]))

    return model, y_test, preds, X_train


def plot_results(results):
    os.makedirs("resources/plots", exist_ok = True)

    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize = (14, 5 * n))

    for i, (shift, model, y_test, preds, X_train) in enumerate(results):
        ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels = ["Down", "Up"]).plot(ax = axes[i, 0], cmap = "Blues", colorbar = False)
        axes[i, 0].set_title(f"Confusion Matrix (shift={shift})")

        axes[i, 1].barh(X_train.columns, model.feature_importances_)
        axes[i, 1].set_xlabel("Importance")
        axes[i, 1].set_ylabel("Features")
        axes[i, 1].set_title(f"Feature Importances (shift={shift})")

    plt.tight_layout()
    plt.savefig("resources/plots/shift_comparison.png", dpi = 120)
    plt.close()
    print("Saved to resources/plots/shift_comparison.png")


results = []
for shift in SHIFTS:
    print(f"\n{"="*50}\nRunning shift={shift}\n{"="*50}")
    labels, features = process_data(shift = shift)
    best_params = optimise(labels, features)
    model, y_test, preds, X_train = evaluate(labels, features, best_params, shift)
    results.append((shift, model, y_test, preds, X_train))

plot_results(results)