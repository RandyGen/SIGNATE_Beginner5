import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.simplefilter('ignore')
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# drop Outliers
train['Overall Qual'].where(train['SalePrice']<350000, inplace=True)
train['Year Built'].where(train['SalePrice']<350000, inplace=True)
train['Year Built'].where((train['SalePrice']<250000) | (train['Year Built']!=1920), inplace=True)
train['Year Remod/Add'].where(train['SalePrice']<300000, inplace=True)
train['Total Bsmt SF'].where(train['SalePrice']<300000, inplace=True)
train['1st Flr SF'].where(train['SalePrice']<350000, inplace=True)
train['Gr Liv Area'].where(train['SalePrice']<300000)
train['Bsmt Full Bath'].where(train['SalePrice']<350000, inplace=True)
train['Full Bath'].where(train['SalePrice']<350000, inplace=True)
train['Half Bath'].where(train['SalePrice']<350000, inplace=True)
train['Fireplaces'].where(train['SalePrice']<350000, inplace=True)
train['Garage Cars'].where(train['SalePrice']<350000, inplace=True)
train['Garage Cars'].where((train['SalePrice']<200000) | (train['Garage Cars']!=1), inplace=True)

categorical_feature=['Overall Qual','Year Built','Year Remod/Add',
                     'Total Bsmt SF','1st Flr SF','Gr Liv Area',
                     'Bsmt Full Bath','Full Bath','Half Bath',
                     'Fireplaces','Garage Cars','Garage Area'
#                      ,'Paved Drive','Electrical'
                    ]
Id = test['index'].astype(int)
train = train[['Overall Qual','Year Built','Year Remod/Add',
               'Total Bsmt SF','1st Flr SF','Gr Liv Area',
               'Bsmt Full Bath','Full Bath','Half Bath',
               'Fireplaces','Garage Cars','Garage Area'
#                ,'Paved Drive','Electrical'
               ,'SalePrice']]
test = test[categorical_feature]

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(train, train['SalePrice'],
                                                      test_size=0.3, random_state=0)
# , stratify=train['SalePrice'])
# stratify=train['SalePrice'] <- 連続値には不可

X_train.drop(columns='SalePrice', inplace=True)
X_valid.drop(columns='SalePrice', inplace=True)

from sklearn.model_selection import StratifiedKFold

# 5-fold CVモデルの学習
# 5つのモデルを保存するリストの初期化
models = []
pred_ave = []
first_judge = True
num_fold = 6

# 学習データの数だけの数列（0行から最終行まで連番）
row_no_list = list(range(len(y_train)))

# KFoldクラスをインスタンス化（これを使って5分割する）
K_fold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=42)

# KFoldクラスで分割した回数だけ実行（ここでは5回）
for train_cv_no, eval_cv_no in K_fold.split(row_no_list, y_train):
    # ilocで取り出す行を指定
    X_train_cv = X_train.iloc[train_cv_no, :]
    y_train_cv = pd.Series(y_train).iloc[train_cv_no]
    X_eval_cv = X_train.iloc[eval_cv_no, :]
    y_eval_cv = pd.Series(y_train).iloc[eval_cv_no]

    # 学習用
    lgb_train = lgb.Dataset(X_train_cv, y_train_cv,
                            categorical_feature=categorical_feature)
    # 検証用
    lgb_eval = lgb.Dataset(X_eval_cv, y_eval_cv, reference=lgb_train,
                           categorical_feature=categorical_feature)

    # パラメータを設定
    params = {'objective': 'regression',
              'metric': 'mse',
              #               'learning_rate':0.1,
              #               'num_iterations':100,
              #               'num_leaves':31,
              #               'max_depth':-1
              #               'weight_columns':[0.07495,0.110919,0.065571,0.067531,0.05158,
              #                                0.10085,0.053923,0.147468,0.065578,0.060578,
              #                                0.10546,0.095594]
              }

    # 学習
    evaluation_results = {}  # 学習の経過を保存する箱
    model = lgb.train(params,  # 上記で設定したパラメータ
                      lgb_train,  # 使用するデータセット
                      num_boost_round=1000,  # 学習の回数
                      valid_sets=[lgb_train, lgb_eval],  # モデル検証のデータセット
                      categorical_feature=categorical_feature,  # カテゴリー変数を設定
                      early_stopping_rounds=100,  # アーリーストッピング# 学習
                      verbose_eval=10)  # 学習の経過の非表示

    # テストデータで予測する
    y_pred = model.predict(test, num_iteration=model.best_iteration)

    if first_judge:
        pred_ave = y_pred
        first_judge = False
    else:
        pred_ave = pred_ave + y_pred

    # 学習が終わったモデルをリストに入れておく
    models.append(model)

pred_ave = pred_ave / num_fold

first_judge = True

for model in models:
    x_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    if first_judge:
        x_pred_ave = x_pred
        first_judge = False
    else:
        x_pred_ave = x_pred_ave + x_pred

x_pred_ave = x_pred_ave / num_fold
np.sqrt(mean_squared_error(y_valid, x_pred_ave))

# my_solution = pd.DataFrame(pred_ave, Id, columns=['SalePrice'])
# my_solution.to_csv("my_prediction_data.csv", header=False)
