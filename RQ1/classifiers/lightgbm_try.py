import json
import os

import sklearn
from numpy import *
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import ast
import re
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import time

if __name__ == '__main__':

    start_time = time.time()

    data = pd.read_csv('data_with_BERT_Embeddings.csv', encoding='utf-8')
    vectors_str = data['vector'].tolist()
    vectors = []
    for s in vectors_str:
        s_new = s.replace('  ', ',')
        l = eval(s_new)
        l_new = l[:300]

        vectors.append(l_new)

    labels = data['label'].tolist()

    # create 10-folds cross validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    # initialize the scores
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    params_dict = {
        'learning_rate':[0.1,0.09,0.08],
        'max_iter':[100,200,300],
        'max_depth':[4,5,6],
        'min_samples_leaf':[15,20]
    }

    # create a HistGradientBoostingClassifier
    lgbm = HistGradientBoostingClassifier(learning_rate=0.08, max_depth=4, max_iter=200, min_samples_leaf=20)

    # do 10-folds cross validation
    for train_index, test_index in kf.split(vectors):
        X_train, X_test = [vectors[i] for i in train_index], [vectors[i] for i in test_index]
        y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]

        lgbm.fit(X_train, y_train)

        # predict the test set
        predict = lgbm.predict_proba(X_test)[:, 1]
        threshold = 0.50
        y_pred = [1 if i > threshold else 0 for i in predict]

        # calculate the scores
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    # calculate the average scores
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    # output the scores
    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1:", avg_f1)
    print(f1_scores)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Time cost:{total_time} s')



