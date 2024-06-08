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
import time


if __name__ == '__main__':

    start_time = time.time()

    data = pd.read_csv('data_with_BERT_Embeddings.csv',encoding='utf-8')
    vectors_str = data['vector'].tolist()
    vectors = []
    for s in vectors_str:
        s_new = s.replace('  ',',')
        l = eval(s_new)
        l_new = l[:300]

        vectors.append(l_new)

    labels = data['label'].tolist()

    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    rgs = RandomForestClassifier(random_state=0, criterion='gini', max_features="sqrt", min_samples_leaf=2,
                                 min_samples_split=5, n_estimators=300)

    for train_index, test_index in kf.split(vectors):

        X_train, X_test = [vectors[i] for i in train_index], [vectors[i] for i in test_index]
        y_train, y_test = [labels[i] for i in train_index], [labels[i] for i in test_index]

        rgs.fit(X_train, y_train)

        predict = rgs.predict_proba(X_test)[:, 1]
        threshold = 0.50
        y_pred = [1 if i > threshold else 0 for i in predict]

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred))
        recall_scores.append(recall_score(y_test, y_pred))
        f1_scores.append(f1_score(y_test, y_pred))

    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    avg_precision = sum(precision_scores) / len(precision_scores)
    avg_recall = sum(recall_scores) / len(recall_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print("Average Accuracy:", avg_accuracy)
    print("Average Precision:", avg_precision)
    print("Average Recall:", avg_recall)
    print("Average F1:", avg_f1)
    print(f1_scores)

    end_time = time.time()
    total_time = end_time - start_time
    print(f'Time cost:{total_time} s')
