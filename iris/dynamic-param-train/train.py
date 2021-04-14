from sklearn.linear_model import LogisticRegression, ARDRegression
import pandas as pd

import os, sys

# cur_path  = os.getcwd()
MM_MODEL = '/mm/project/model/'
# print('current path : %s' % cur_path)
MM_DATA = '/mm/project/data_in/'
target = 'LOGISTIC'
#######################

if not os.path.exists(MM_MODEL):
    os.mkdir(MM_MODEL)
# Our model - a multiclass regression
# FIRST we initialize it with default params or specify some
X_train = pd.read_csv(MM_DATA + 'X_train.csv')
X_test = pd.read_csv(MM_DATA + 'X_test.csv')
y_train = pd.read_csv(MM_DATA + 'Y_train.csv')
y_test = pd.read_csv(MM_DATA + 'Y_test.csv')

logistic = LogisticRegression()
import sklearn.linear_model as lm

# dynamic param
if not os.environ['target'] == 'default':
    target = os.environ['target']

if not len(sys.argv) is 1:
    intercept_param = int(sys.argv[1])
else:
    intercept_param = 1

print('--------log--------')
print(target)

if target == 'LOGISTIC':
    model = LogisticRegression(intercept_scaling=intercept_param)

elif target == 'ARD':
    model = ARDRegression()
# Train on iris training set
# SECOND we give the model some training data
model.fit(X_train, y_train)

# THIRD we give our model some test data and predict something
from sklearn.metrics import accuracy_score

y_predict = model.predict(X_test)

acc_score = accuracy_score(y_test, y_predict)

print("accuracy score :%f" % acc_score)
import pickle

# save model
with open(MM_MODEL + 'iris_logistic.pkl', 'wb') as f:
    pickle.dump(model, f)

print("model saved, path :%s" % MM_MODEL)