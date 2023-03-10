# baseline Support Vector Machine / K-Nearest Neighbors classifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import warnings
import xgboost as xgb
warnings.filterwarnings("ignore")

def evaluate_baseline(data):
    """
    Helper function split input DataFrame into training/testing sets,
    fit SVM model, and evlauate performance on the testing set.
    """
    # split data into train/test sets
    df = pd.DataFrame(data)
    X_train, y_train = df[df['is_train'] == 1.0].drop(columns=['vid', 'label', 'is_test', 'is_train']), df[df['is_train']==1.0]['label']
    X_test, y_test = df[df['is_test'] == 1.0].drop(columns=['vid', 'label', 'is_train', 'is_test']), df[df['is_test']==1.0]['label']
    # fit SVM model to training data
    np.random.seed(42)
    model = SVC(gamma='auto').fit(X_train, y_train)
    # get predictions and evaluate performance on test set
    preds = model.predict(X_test)
    test_acc = np.mean(y_test.to_numpy() == preds)
    return test_acc

def evaluate_KNN(data):
    """
    Helper function split input DataFrame into training/testing sets,
    fit KNN model, and evlauate performance on the testing set.
    """
    # split data into train/test sets
    df = pd.DataFrame(data)
    X_train, y_train = df[df['is_train'] == 1.0].drop(columns=['vid', 'label', 'is_test', 'is_train']), df[df['is_train']==1.0]['label']
    X_test, y_test = df[df['is_test'] == 1.0].drop(columns=['vid', 'label', 'is_train', 'is_test']), df[df['is_test']==1.0]['label']
    # fit KNN model (k = 3) to training data
    np.random.seed(42)
    model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
    # get predictions and evaluate performance on test set
    preds = model.predict(X_test)
    test_acc = np.mean(y_test.to_numpy() == preds)
    return test_acc


def evaluate_xgb(data):
    """
    Helper function split input DataFrame into training/testing sets,
    fit XGBoost model, and evlauate performance on the testing set.
    """
    # split data into train/test sets
    df = pd.DataFrame(data).astype(float)
    X_train, y_train = df[df['is_train'] == 1.0].drop(columns=['vid', 'label', 'is_test', 'is_train']), df[df['is_train']==1.0]['label']
    X_test, y_test = df[df['is_test'] == 1.0].drop(columns=['vid', 'label', 'is_train', 'is_test']), df[df['is_test']==1.0]['label']
    # fit xgboost to training data
    np.random.seed(42)
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.001, objective='binary:logistic').fit(X_train,y_train)
    preds = model.predict(X_test)
    test_acc = np.mean(y_test.to_numpy() == preds)
    return test_acc