from sklearn.linear_model import LogisticRegression
import pandas as pd

import os
import pickle
import csv

home = os.environ['project_home']
workflow_home = os.environ['workflow_path']
step = "models"
target_path = os.environ['target_path']
seq = os.environ.get('seq', '0')

MM_DATA  = os.path.join(home, "data_in")
MM_MODEL = os.path.join(home, 'model')

if not os.path.exists(MM_DATA):
    os.mkdir(MM_DATA)

if not os.path.exists(MM_MODEL):
    os.mkdir(MM_MODEL)

X_train = pd.read_csv(os.path.join(MM_DATA, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(MM_DATA, 'Y_train.csv'))

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

with open(os.path.join(MM_MODEL, 'logistic_model.pkl'), 'wb') as f:
    pickle.dump(logistic, f)


# train_file = '/mm/step/train.csv'
train_file = os.path.join(home, workflow_home, step, target_path, seq, 'train.csv')

with open(train_file, 'w') as csvfile:
    fieldnames = ['target', 'path', 'args']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'target': 'iris_classifier', 'path': os.path.join(MM_MODEL, 'logistic_model.pkl'), 'args': 'LR'})