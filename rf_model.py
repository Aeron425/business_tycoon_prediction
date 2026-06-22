from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from bayes_opt import BayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def process_data(filepath="resources/data/processedx2_data.csv"):
    data = pd.read_csv(filepath)

    labels = data["Direction"].map({-1.0: 0, 1.0: 1})
    features = data.drop(["Direction"], axis=1)

    return labels, features

def rf_eval(labels, features, n_estimators, max_depth):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        n_jobs=-1,
        random_state=42,
    )
    return cross_val_score(model, features, labels, cv=CV, scoring="accuracy").mean()

def optimise(labels, features, n_iter=50, init_points=10):
    opt = BayesianOptimization(
        f=lambda n_estimators, max_depth: rf_eval(labels, features, n_estimators, max_depth),
        pbounds={"n_estimators": (50, 500), "max_depth": (3, 15)},
        random_state=42,
        verbose=2,
    )
    opt.maximize(init_points=init_points, n_iter=n_iter)
    return opt.max["params"]

def evaluate(labels, features, best_params):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    model = RandomForestClassifier(
        n_estimators=int(best_params["n_estimators"]),
        max_depth=int(best_params["max_depth"]),
        n_jobs=-1,
        random_state=42,
    )

    os.makedirs("resources/models", exist_ok=True)
    with open("resources/models/rf.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved to resources/models/rf.pkl")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, preds) * 100:.2f}%")
    print(classification_report(y_test, preds, target_names=["Down", "Up"]))

    importances = model.feature_importances_
    features = X_train.columns
    plt.bar(len(features) - np.arange(len(features)), importances[::-1], align="center")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Random Forest: Feature Importances")
    plt.tight_layout()
    plt.savefig("resources/plots/rf_feature_importances.png", dpi=120)
    plt.close()
    
    ConfusionMatrixDisplay(confusion_matrix(y_test, preds), display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.title("Random Forest: Confusion Matrix")
    plt.tight_layout()
    
    os.makedirs("resources/plots", exist_ok=True)
    plt.savefig("resources/plots/rf_confusion_matrix.png", dpi=120)
    plt.close()

labels, features = process_data()
best_params = optimise(labels, features)
evaluate(labels, features, best_params)