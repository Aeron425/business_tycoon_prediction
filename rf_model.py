from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import os 
import pickle


CV = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)


def process_data(filepath = "resources/data/processed_data.csv"):
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip()
    data["Direction"] = data["Direction"].shift(-1)
    data = data.dropna()
    data = data[data["Direction"] != 0]

    labels = data["Direction"].map({-1: 0, 1: 1})
    features = data.drop(["Direction", "Time"], axis = 1)

    return labels, features


def rf_eval(labels, features, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators = int(n_estimators),
        max_depth = int(max_depth),
        n_jobs = -1,
        random_state = 42,
    )

    return cross_val_score(model, features, labels, cv = CV, scoring = "accuracy").mean()


def optimise(labels, features, n_iter = 50, init_points = 10):
    opt = BayesianOptimization(
        f = lambda n_estimators, max_depth: rf_eval(labels, features, n_estimators, max_depth),
        pbounds = {"n_estimators": (50, 500), "max_depth": (3, 15)},
        random_state = 42,
        verbose = 2,
    )

    opt.maximize(init_points = init_points, n_iter = n_iter)
    return opt.max["params"]


def evaluate(labels, features, best_params):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 42, stratify = labels)
#
#   model = RandomForestClassifier(
#       n_estimators = int(best_params["n_estimators"]),
#       max_depth = int(best_params["max_depth"]),
#       n_jobs = -1,
#       random_state = 42,
#   )

    model = pickle.load(open("resources/models/rf.pkl", "rb"))

    os.makedirs("resources/models", exist_ok = True)
    with open("resources/models/rf.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to resources/models/rf.pkl")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names = ["Down", "Up"]))
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels = ["Down", "Up"]).plot(cmap = "Blues")
    plt.title("Random Forest: Confusion Matrix")
    plt.tight_layout()
    plt.savefig("resources/plots/rf_confusion_matrix.png", dpi = 120)
    plt.close()


labels, features = process_data("processed_data.csv")
# best_params = optimise(labels, features)
evaluate(labels, features, None)