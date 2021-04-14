from sklearn.linear_model import LogisticRegression
import pandas as pd

import os
import pickle
import csv

MM_DATA = '/mm/project/data_in/'
MM_MODEL = '/mm/project/model/'

if not os.path.exists(MM_DATA):
    os.mkdir(MM_DATA)

if not os.path.exists(MM_MODEL):
    os.mkdir(MM_MODEL)

X_train = pd.read_csv(MM_DATA + 'X_train.csv')
y_train = pd.read_csv(MM_DATA + 'Y_train.csv')

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

with open(MM_MODEL + 'logistic_model.pkl', 'wb') as f:
    pickle.dump(logistic, f)

with open('/mm/step/train.csv', 'w') as csvfile:
    fieldnames = ['target', 'path', 'args']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'target': 'iris_classifier', 'path': MM_MODEL + 'logistic_model.pkl', 'args': 'LR'})