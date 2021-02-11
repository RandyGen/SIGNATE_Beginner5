import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train.replace({'Individual': 0, 'Joint App': 1}, inplace=True)
train.replace({'FullyPaid': 0, 'ChargedOff': 1}, inplace=True) # 目的変数
train = pd.get_dummies(train, prefix='', prefix_sep='', columns=['term', 'grade', 'purpose'])

test.replace({'Individual': 0, 'Joint App': 1}, inplace=True)
test = pd.get_dummies(test, prefix='', prefix_sep='', columns=['term', 'grade', 'purpose'])

train['moving'] = 0
test['F3'] = 0
test['F5'] = 0

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, train['loan_status'], test_size=0.3, random_state=0, stratify=train['loan_status'])

X_train.drop(columns=['loan_amnt', 'employment_length', 'credit_score', 'loan_status'], inplace=True)
X_valid.drop(columns=['loan_amnt', 'employment_length', 'credit_score', 'loan_status'], inplace=True)
y_train.drop(columns=['loan_amnt', 'employment_length', 'credit_score'], inplace=True)
y_valid.drop(columns=['loan_amnt', 'employment_length', 'credit_score'], inplace=True)
categorical_feature = ['interest_rate', 'application_type', '3 years', '5 years', 'A1', 'A2', 'A3', 'A4', 'A5', 'B1',
                       'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2',
                       'E3', 'E4', 'E5', 'F3', 'F5', 'car', 'credit_card', 'debt_consolidation', 'home_improvement',
                       'house', 'major_purchase', 'medical', 'other', 'small_business', 'moving']

test.drop(columns=['loan_amnt', 'employment_length', 'credit_score'], inplace=True)

import lightgbm as lgb

lgb_train = lgb.Dataset(X_train, y_train, categorical_feature=categorical_feature)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train, categorical_feature=categorical_feature)

param = {
    'objective': 'binary',
        'metric': 'binary_error'
        }

model = lgb.train(param, lgb_train, valid_sets=[lgb_train, lgb_eval], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=100)

y_pred = model.predict(test, num_iteration=model.best_iteration)

X_pred = model.predict(X_valid, num_iteration=model.best_iteration)
X_pred = (X_pred > 0.18325).astype(int)
f1_score(y_valid, X_pred)

# train_pred = model.predict(X_train, num_iteration=model.best_iteration)
# train_pred = (train_pred > 0.5).astype(int)
# f1_score(y_train, train_pred)

y_pred = (y_pred > 0.18325).astype(int)
Id = test.id.astype(int)
my_solution = pd.DataFrame(y_pred, Id, columns=['loan_status'])
my_solution.to_csv("my_prediction_data.csv", header=False)

# ax = train.plot(kind='scatter', x='interest_rate', y='loan_amnt', color='DarkBlue', label='loan_status')
# train.plot(kind='scatter', x='interest_rate', y='loan_amnt', color='DarkGreen', label='loan_status', ax=ax)
