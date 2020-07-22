from sklearn import model_selection
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

def svm_param_selection(x, y, n_folds, metric):
    # Iperparametri per svm
    parameters = [
                  {"kernel": ['rbf'], 'C': [0.1, 1, 10, 25, 50, 100],
                   "gamma": [10e-4, 10e-3, 10e-2, 10e-1, 10, 10 ** 2, 10 ** 3, 10 ** 4],
                   "decision_function_shape": ["ovo", "ovr"]
                   },
                  {
                   "kernel": ['linear'], "C": [0.1, 1, 10], "decision_function_shape": ["ovo", "ovr"]
                  }
    ]
    clf = model_selection.GridSearchCV(SVC(), param_grid=parameters, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(x, y)

    print("Best parameters:\n")
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_

def decision_tree_param_selection(X, y, n_folds, metric):
    # Iperparametri per DecisionTree
    param_grid = {
               'criterion': ['entropy', 'gini'],
               'splitter': ['best', 'random'],
               'max_features': [None ,'auto', 'log2'],
               'min_samples_leaf': [1, 5, 10],
               'max_depth': [10, 25, 50, 100],
               'min_samples_split': [2, 7, 15]
    }

    clf = model_selection.GridSearchCV(DecisionTreeClassifier(),  param_grid=param_grid, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_

def random_forest_param_selection(X, y, n_folds, metric):
    # Iperparametri per RandomForest
    param_grid = {
        'criterion': ['entropy', 'gini'],
        'max_features': [None, 'auto', 'log2'],
        'min_samples_leaf': [1, 5, 10],
        'max_depth': [10, 25, 50, 100],
        'min_samples_split': [2, 7, 15],
        'n_estimators': [100, 150, 300, 400]
    }


    clf = model_selection.GridSearchCV(RandomForestClassifier(),  param_grid=param_grid, scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_

def mlp_param_selection(X, y, n_folds, metric):
    # Iperparametri per MLP
    parameters = [{
        'hidden_layer_sizes': [(100, 50), (100, 50, 25), (200,)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'learning_rate_init': [0.1, 0.01, 1e-3, 1e-4],
        'learning_rate': ['invscaling', 'constant', 'adaptive'],
    }]

    clf = model_selection.GridSearchCV(MLPClassifier(max_iter=10000), param_grid=parameters,
                                       scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_

def k_neighbors_classifier_parm_selection(X, y, n_folds, metric):
    # griglia degli iperparametri
    parameters = [{
        'n_neighbors': [5, 7, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [30, 35, 40, 50],
        'p': [1, 2, 3, 4],
    }]


    clf = model_selection.GridSearchCV(KNeighborsClassifier(), param_grid=parameters,
                                       scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_

def naive_bayes_param_selection(X, y, n_folds, metric):
    # Iperparametri per NaiveBayes
    parameters = [{
        'priors': [None, [0.25, 0.25, 0.25, 0.25]],
        'var_smoothing': [10e-9, 10e-6, 10e-3, 10e-1]
    }]

    clf = model_selection.GridSearchCV(GaussianNB(), param_grid=parameters,
                                       scoring=metric,
                                       cv=n_folds, refit=True,
                                       n_jobs=-1)
    clf.fit(X, y)

    print("Best parameters:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.4f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    return clf.best_estimator_