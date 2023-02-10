# baseline Support Vector Machine classifier
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

def evaluate_baseline(data):
    """
    Helper function split input DataFrame into training/testing sets,
    fit SVM model, and evlauate performance on the testing set.
    """
    # splite data into train/test sets
    df = pd.DataFrame(data)
    X_train, y_train = df[df['is_train'] == '1'].drop(columns=['vid', 'label', 'is_test', 'is_train']), df[df['is_train']=='1']['label']
    X_test, y_test = df[df['is_test'] == '1'].drop(columns=['vid', 'label', 'is_train', 'is_test']), df[df['is_test']=='1']['label']
    # fit SVM model to training data
    np.random.seed(42)
    model = SVC(gamma='auto').fit(X_train, y_train)
    # get predictions and evaluate performance on test set
    preds = model.predict(X_test)
    test_acc = np.mean(y_test.to_numpy() == preds)
    return test_acc