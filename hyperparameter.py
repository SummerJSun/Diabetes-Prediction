import argparse
import json
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import matplotlib.pyplot as plt





def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Grid Search CV for hyperparameter tuning.
    """
    start = time.time()
    grid = GridSearchCV(clf, param_grid=pgrid, scoring='roc_auc', cv=5)
    grid.fit(xTrain, yTrain)
    best_clf = grid.best_estimator_
    y_pred = best_clf.predict(xTest)
    y_scores = best_clf.predict_proba(xTest)[:, 1]
    best_params = grid.best_params_
    elapsed_time = time.time() - start
    results = {"AUC": metrics.roc_auc_score(yTest, y_scores),
               "AUPRC": metrics.average_precision_score(yTest, y_scores),
               "F1": metrics.f1_score(yTest, y_pred),
               "Time": elapsed_time}
    fpr, tpr, _ = roc_curve(yTest, y_scores)
    roc = {"fpr": fpr, "tpr": tpr}
    return results, roc, best_params

def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Random Search CV for hyperparameter tuning.
    """
    start = time.time()
    total_combinations = np.prod([len(v) for v in pgrid.values()])
    n_iter = max(10, int(total_combinations * 0.33))
    random_search = RandomizedSearchCV(clf, param_distributions=pgrid, n_iter=n_iter, scoring='roc_auc', cv=5)
    random_search.fit(xTrain, yTrain)
    best_clf = random_search.best_estimator_
    y_pred = best_clf.predict(xTest)
    y_scores = best_clf.predict_proba(xTest)[:, 1]
    best_params = random_search.best_params_
    elapsed_time = time.time() - start
    results = {"AUC": metrics.roc_auc_score(yTest, y_scores),
               "AUPRC": metrics.average_precision_score(yTest, y_scores),
               "F1": metrics.f1_score(yTest, y_pred),
               "Time": elapsed_time}
    fpr, tpr, _ = roc_curve(yTest, y_scores)
    roc = {"fpr": fpr, "tpr": tpr}
    return results, roc, best_params

def get_parameter_grid(mName):
    """
    Get the parameter grid based on the model name with an expanded range of hyperparameters.
    """
    grids = {
        "DT": {
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 10, 20, 40],
            'min_samples_leaf': [1, 2, 4, 6],
            'criterion': ['gini', 'entropy']
        },
        "LR": {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
            'penalty': ['none', 'l1', 'l2', 'elasticnet'],
            'class_weight': [None, 'balanced']
        },
        "RF": {
            'n_estimators': [10, 50, 100, 200, 300],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [None, 10, 20, 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        },
        "NB": {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "NN": {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'invscaling', 'adaptive']
        }
    }
    return grids.get(mName, {})


models = {
    "DT": DecisionTreeClassifier(),
    "LR": LogisticRegression(),
    "RF": RandomForestClassifier(),
    "NB": GaussianNB(),
    "NN": MLPClassifier(max_iter=1000)
}



def evaluate_model(model, xTest, yTest):
    y_pred = model.predict(xTest)
    y_scores = model.predict_proba(xTest)[:, 1]

    accuracy = accuracy_score(yTest, y_pred)
    precision = precision_score(yTest, y_pred)
    recall = recall_score(yTest, y_pred)
    f1 = f1_score(yTest, y_pred)
    roc_auc = roc_auc_score(yTest, y_scores)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(yTest, y_scores, n_bins=10)

    plt.figure(figsize=(6, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration plot')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('True probability in each bin')
    plt.legend()
    plt.title('Calibration Curve')
    plt.show()

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC AUC': roc_auc}, confusion_matrix(yTest, y_pred)

def main():
    # xTrain, xTest, yTrain, yTest
    models = {
        "DT": DecisionTreeClassifier(),
        "LR": LogisticRegression(),
        "RF": RandomForestClassifier(),
        "NB": GaussianNB(),
        "NN": MLPClassifier(max_iter=1000)
    }
    
    bestAUC = 0 
    bestModel = None

    for mName, clf in models.items():
        print(f"Tuning {mName}...")
        grid = get_parameter_grid(mName)
        results, roc, best_params = eval_randomsearch(clf, grid, xTrain, yTrain, xTest, yTest)
        if results['ROC AUC'] > bestAUC:
            bestAUC = results['ROC AUC']
            bestModel = clf.set_params(**best_params)
            bestModel.fit(xTrain, yTrain)  

    metrics, conf_matrix = evaluate_model(bestModel, xTest, yTest)
    print("Performance Metrics:", metrics)
    print("Confusion Matrix:\n", conf_matrix)
