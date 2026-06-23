from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import pickle


CV = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

SHIFT = 1


def process_data(filepath = "resources/data/model_ready_data.csv", shift = 1):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()

    data["future_price"] = data["money"].shift(-shift)
    data["target"] = (data["future_price"] > data["money"]).astype(int)
    data = data.dropna()

    labels = data["target"]
    features = data.drop(["target", "future_price", "direction", "time"], axis = 1)
    return labels, features


def rf_eval(labels, features, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators = int(n_estimators),
        max_depth = int(max_depth),
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


def evaluate(labels, features, best_params=None):

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42, stratify = labels)

    model = RandomForestClassifier(
        n_estimators = int(best_params["n_estimators"]),
        max_depth = int(best_params["max_depth"]),
        n_jobs = -1,
        random_state = 42,
    )

    # with open("resources/models/rf.pkl", "rb") as f:
    #     model = pickle.load(f)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    with open(f"resources/models/rf_shift_{SHIFT}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to resources/models/rf_shift_{SHIFT}.pkl")

    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names = ["Down", "Up"]))

    plt.barh(X_train.columns, model.feature_importances_)
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Random Forest: Feature Importances")
    plt.tight_layout()
    plt.savefig(f"resources/plots/rf_feature_importances_{SHIFT}.png", dpi = 120)
    plt.close()

    ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels = ["Down", "Up"]).plot(cmap = "Blues")
    plt.title("Random Forest: Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"resources/plots/rf_confusion_matrix_{SHIFT}.png", dpi = 120)
    plt.close()


labels, features = process_data(shift=SHIFT)
best_params = optimise(labels, features)
evaluate(labels, features, best_params=best_params)

# accuracy
# Shift 15, 71%
# Shift 10, 72%
# Shift 7, 73%
# Shift 5, 71%
# Shift 3, 69%
# Shift 1, 66%