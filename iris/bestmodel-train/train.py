from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import os, sys

MM_MODEL = '/mm/project/model/'
MM_DATA='/mm/project/data_in/'
MM_INFO = '/mm/step/train.csv'

target = os.environ['target']
args = sys.argv
algo = args[1]

def create_df (col_list = []) : 
    df = pd.DataFrame(columns = col_list)
    return df

if not os.path.exists(MM_MODEL):
    os.mkdir(MM_MODEL)

X_train = pd.read_csv(MM_DATA + 'X_train.csv')
X_test = pd.read_csv(MM_DATA + 'X_test.csv')
y_train = pd.read_csv(MM_DATA + 'Y_train.csv')
y_test = pd.read_csv(MM_DATA + 'Y_test.csv')

train_col = ['target', 'path', 'score', 'args']
train_df = create_df(train_col)

import pickle

MODEL_NAME = 'iris_'  +algo+ '.pkl'
model= eval(algo + '()')
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score
y_predict = model.predict(X_test)

acc_score = accuracy_score(y_test, y_predict)

print("accuracy score :%f" % acc_score)
with open(os.path.join(MM_MODEL, MODEL_NAME), 'wb') as f:
    pickle.dump(model, f)

print("model saved, path :%s" % os.path.join(MM_MODEL + MODEL_NAME))

a = pd.DataFrame(data=[[ target, MM_MODEL+MODEL_NAME, acc_score, 1]],
             columns=train_col)
print(a)

train_df = train_df.append(a).reset_index(drop=True)
    
print(train_df)
train_df.to_csv(MM_INFO, index = False)