# baseline Support Vector Machine classifier
import numpy as np
import pandas as pd
import json
import pyTigerGraph as tg
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

def evaluate_baseline(df):
    """
    Helper function split input dataframe into training/testing sets,
    fit SVM model, and evlauate performance on the testing set.
    """
    # splite data into train/test sets
    X_train, y_train = df[df['is_train'] == 1].drop(columns=['Unnamed: 0', 'vid', 'label', 'is_test', 'is_train']), df[df['is_train']==1]['label']
    X_test, y_test = df[df['is_test'] == 1].drop(columns=['Unnamed: 0', 'vid', 'label', 'is_train', 'is_test']), df[df['is_test']==1]['label']
    # fit SVM model to training data
    model = SVC(gamma='auto').fit(X_train, y_train)
    # get predictions and evaluate performance on test set
    preds = model.predict(X_test)
    test_acc = np.mean(y_test.to_numpy() == preds)
    return test_acc