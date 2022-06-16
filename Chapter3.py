#!/usr/bin/env python
# coding: utf-8

# ![表紙](https://www.oreilly.co.jp/books/images/picture978-4-87311-907-6.gif)
# 
# このノートブックはオライリー・ジャパンより発行の書籍[『セキュリティエンジニアのための機械学習』](https://www.oreilly.co.jp/books/9784873119076/)のサンプルコードです。コードの解説等は書籍をご参照ください。なお、このコードを動作させた結果について、著者およびオライリー・ジャパンは一切の責任を負いません。

# ## ファイルのメタデータを特徴量にしたマルウェア検出
# 
# 

# ### 最初にpandas-profilingを更新して、ランタイムを再起動しておく

# In[ ]:


#get_ipython().system('pip install -U pandas-profiling')


# In[ ]:


#get_ipython().system('wget https://github.com/oreilly-japan/ml-security-jp/raw/master/ch03/MalwareData.csv.gz')


# In[ ]:


#get_ipython().system('gzip -d MalwareData.csv.gz')


# In[ ]:


#get_ipython().system('pip install optuna')


# In[ ]:


import pandas as pd
MalwareDataset = pd.read_csv('MalwareData.csv', sep='|')


# In[ ]:


MalwareDataset.info()


# In[ ]:


import pandas_profiling

pandas_profiling.ProfileReport(MalwareDataset, minimal=True)


# In[1]:


import matplotlib.pyplot as plt

plt.hist(
    MalwareDataset.loc[MalwareDataset['legitimate'] == 1,\
                       'VersionInformationSize'],
    range=(0,26), alpha=0.5, label='1'
    )
plt.hist(
    MalwareDataset.loc[MalwareDataset['legitimate'] == 0,\
                       'VersionInformationSize'],
    range=(0,26), alpha=0.5, label='0'
    )
plt.legend(title='legitimate')
plt.xlim(0,26)


# In[ ]:


import matplotlib.pyplot as plt

plt.hist(
    MalwareDataset.loc[MalwareDataset['legitimate'] == 1,\
                            'MajorSubsystemVersion'],
         range=(0,10), alpha=0.5, label='1'
         )
plt.hist(
    MalwareDataset.loc[MalwareDataset['legitimate'] == 0,\
                            'MajorSubsystemVersion'],
         range=(0,10), alpha=0.5, label='0'
         )

plt.legend(title='legitimate')
plt.xlim(2,11)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(MalwareDataset.loc[MalwareDataset['legitimate'] == 1, 'SizeOfUninitializedData'], range=(0,10000), alpha=0.5, label='1')
plt.hist(MalwareDataset.loc[MalwareDataset['legitimate'] == 0, 'SizeOfUninitializedData'], range=(0,10000), alpha=0.5, label='0')
plt.legend(title='legitimate')
plt.xlim(0,10000)


# In[ ]:


print('The Number of features are  {} \n'.format(MalwareDataset.shape[1]))


# In[ ]:


MalwareDataset.columns


# In[ ]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection

# データセットから名前、md5ハッシュ値、ラベルといった列を除外してXに代入
X = MalwareDataset.drop(['Name', 'md5', 'legitimate'],axis='columns')
# データセットのラベル列のみを抽出してyに代入
y = MalwareDataset['legitimate']
# ExtraTreesClassifierを使用
FeatSelect=ExtraTreesClassifier().fit(X, y)
# SelectFromModelを使用して、
# ExtraTreesClassifierによる分類結果に寄与した重要度の大きい特徴量のみを抽出
Model = SelectFromModel(FeatSelect, prefit=True)
# 重要度の大きい特徴量のカラム名を取得
feature_idx = Model.get_support()
feature_name = X.columns[feature_idx]
# Xに選択した特徴量のみを代入しなおす
X = Model.transform(X)
# 重要度の大きい特徴量のカラム名を設定
X = pd.DataFrame(X)
X.columns = feature_name


# In[ ]:


Features = X.shape[1]
# 重要度をリストで抽出
FI = ExtraTreesClassifier().fit(X,y).feature_importances_
# 重要度を高い順にソート
Index = np.argsort(FI)[::-1][:Features]
# 重要度の高い順に、特徴量の名前と重要度を出力
for feat  in range(Features):
    print(
        "Feature: {}Importance: {:.5f}"\
          .format(MalwareDataset.columns[2+Index[feat]].ljust(30),
                  FI[Index[feat]])
          )


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import cross_validate

# データセットを訓練用とテスト用に分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=101
    )

# RandomForestClassifierのハイパーパラメータ探索用のクラスを設定
class Objective_RF:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        # 探索対象のパラメータの設定
        criterion = trial.suggest_categorical("criterion",
                                              ["gini", "entropy"])
        bootstrap = trial.suggest_categorical('bootstrap',
                                              ['True','False'])
        max_features = trial.suggest_categorical('max_features',
                                            ['auto', 'sqrt','log2'])
        min_samples_split = trial.suggest_int('min_samples_split',
                                              2, 5)
        min_samples_leaf = trial.suggest_int('min_samples_leaf',
                                             1,10)

        model = RandomForestClassifier(
            criterion = criterion,
            bootstrap = bootstrap,
            max_features = max_features,
            min_samples_split = min_samples_split,
            min_samples_leaf = min_samples_leaf
        )

        # 交差検証しながらベストのパラメータ探索を行う
        scores = cross_validate(model,
                                X=self.X,
                                y=self.y,
                                cv=5,
                                n_jobs=-1)
        
        # 5分割で交差検証した正解率の平均値を返す
        return scores['test_score'].mean()

# 探索の対象クラスを設定
objective = Objective_RF(X_train, y_train)
study = optuna.create_study()
# 最大で3分間探索を実行
study.optimize(objective, timeout=180)
# ベストのパラメータの出力
print('params:', study.best_params)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# optunaの探索結果として得られたベストのパラメータを設定
model = RandomForestClassifier(
    criterion = study.best_params['criterion'],
    bootstrap = study.best_params['bootstrap'],
    max_features = study.best_params['max_features'],
    min_samples_split = study.best_params['min_samples_split'],
    min_samples_leaf = study.best_params['min_samples_leaf']
)

# モデルの訓練
model.fit(X_train, y_train)

# テスト用のデータを使用して予測
pred = model.predict(X_test)

# 予測結果とテスト用のデータを使って正解率と、混同行列を出力
print("Accuracy: {:.5f} %".format(100 * accuracy_score(y_test, pred)))
print(confusion_matrix(y_test, pred))


# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')

feat_importances = pd.Series(
    model.feature_importances_,
    index=X.columns).sort_values(ascending=True)

feat_importances.plot(kind='barh')


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier

class Objective_GBC:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        # 探索対象のパラメーターの指定
        max_depth=int(
            trial.suggest_loguniform("max_depth", 3, 10))
        max_features = trial.suggest_categorical(
            "max_features", ["log2", "sqrt"])
        learning_rate = float(trial.suggest_loguniform(
            "learning_rate", 1e-2, 1e-0))
        criterion =  trial.suggest_categorical(
            "criterion", ["friedman_mse", "mse", "mae"])

        # モデルの初期化
        model = GradientBoostingClassifier(
            max_depth = max_depth,
            max_features = max_features,
            learning_rate = learning_rate,
            criterion=criterion
            )
        
        scores = cross_validate(model,
                                X=self.X, y=self.y,
                                cv=5,
                                n_jobs=-1)
        return scores['test_score'].mean()

# 探索の対象クラスを設定
objective = Objective_GBC(X_test, y_test)
study = optuna.create_study()

# 1回のみ探索
study.optimize(objective, n_trials=1)

# ベストのパラメーターの出力
print('params:', study.best_params)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 探索結果として得られたベストのパラメーターを設定
model = GradientBoostingClassifier(
    criterion = study.best_params['criterion'],
    learning_rate = study.best_params['learning_rate'],
    max_depth = study.best_params['max_depth'],
    max_features = study.best_params['max_features']
)

# モデルの訓練
model.fit(X_train, y_train)

# テスト用のデータを使用して予測
pred = model.predict(X_test)

# 予測結果とテスト用のデータを使って正解率と、混同行列を出力
print("Accuracy: {:.5f} %".format(100 * accuracy_score(y_test, pred)))
print(confusion_matrix(y_test, pred))


# In[ ]:


feat_importances = pd.Series(
    model.feature_importances_,
    index=X.columns).sort_values(ascending=True)
feat_importances.plot(kind='barh')


# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

class Objective_ABC:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        # 探索対象のパラメーターの指定
        algorithm =  trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"])
        learning_rate = float(trial.suggest_loguniform("learning_rate", 1e-2, 1e-0))

        # モデルの初期化
        model = AdaBoostClassifier(
            algorithm = algorithm,
            learning_rate = learning_rate
            )
        
        scores = cross_validate(model,
                                X=self.X, y=self.y,
                                cv=5,
                                n_jobs=-1)
        return scores['test_score'].mean()

# 探索の対象クラスを設定
objective = Objective_ABC(X_train, y_train)
study = optuna.create_study()

# 最大で1分間探索を実行
study.optimize(objective, timeout=60)

# ベストのパラメーターの出力
print('params:', study.best_params)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 探索結果として得られたベストのパラメーターを設定
model = AdaBoostClassifier(
    algorithm = study.best_params['algorithm'],
    learning_rate = study.best_params['learning_rate']
)
# モデルの訓練
model.fit(X_train, y_train)

# テスト用のデータを使用して予測
pred = model.predict(X_test)

# 予測結果とテスト用のデータを使って正解率と、混同行列を出力
print("Accuracy: {:.5f} %".format(100 * accuracy_score(y_test, pred)))
print(confusion_matrix(y_test, pred))


# In[ ]:


feat_importances = pd.Series(
    model.feature_importances_,
    index=X.columns).sort_values(ascending=True)

feat_importances.plot(kind='barh')


# ## Androidマルウェアのデータセットを使用した検出デモ
# 
# This dataset created & provided by following paper:
# Urcuqui, C., & Navarro, A. (2016, April). Machine learning classifiers for android malware analysis. In Communications and Computing (COLCOM), 2016 IEEE Colombian Conference on (pp. 1-6). IEEE.

# In[ ]:


#get_ipython().system('wget https://github.com/oreilly-japan/ml-security-jp/raw/master/ch03/archive.zip')


# In[ ]:


#get_ipython().system('unzip -q archive.zip')


# In[ ]:


import pandas as pd

# データセットのロード
AndroidDataset = pd.read_csv('train.csv', sep=';')


# In[ ]:


AndroidDataset.type.value_counts()


# In[ ]:


print(AndroidDataset.columns)


# In[ ]:


pd.Series.sort_values(
    AndroidDataset[AndroidDataset.type==1].sum(axis=0),
    ascending=False
    )[1:11]


# In[ ]:


top10 = ['android.permission.INTERNET',
         'android.permission.READ_PHONE_STATE',
         'android.permission.ACCESS_NETWORK_STATE',
         'android.permission.WRITE_EXTERNAL_STORAGE',
         'android.permission.ACCESS_WIFI_STATE',
         'android.permission.READ_SMS',
         'android.permission.WRITE_SMS',
         'android.permission.RECEIVE_BOOT_COMPLETED',
         'android.permission.ACCESS_COARSE_LOCATION',
         'android.permission.CHANGE_WIFI_STATE']


# In[ ]:


AndroidDataset.loc[AndroidDataset.type==1, top10].sum()


# In[ ]:


AndroidDataset.loc[AndroidDataset.type==0, top10].sum()


# In[ ]:


import matplotlib.pyplot as plt
fig, axs =  plt.subplots(nrows=2, sharex=True)

AndroidDataset.loc[AndroidDataset.type==0, top10].sum().plot.bar(ax=axs[0])
AndroidDataset.loc[AndroidDataset.type==1, top10].sum().plot.bar(ax=axs[1], color="red")


# In[ ]:


from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn import svm
import csv
import random


# In[ ]:


# X,yに特徴量とラベルをそれぞれ代入
X = AndroidDataset.iloc[:,:-1]
y = AndroidDataset.iloc[:, -1]


# In[ ]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn.model_selection import train_test_split
import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold, cross_validate

# データセットをテスト用のデータに20%を割り当てて分割
X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.2, shuffle=True, random_state=101)

# SVMのハイパーパラメータチューニング用のクラスを設定
class Objective_SVM:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        # チューニング対象のパラメーターを指定
        params = {
            'kernel': trial.suggest_categorical(
                'kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'C': trial.suggest_loguniform('C', 1e-5, 1e2),
            'gamma': trial.suggest_categorical(
                'gamma', ['scale','auto']),
        }
        # モデルの初期化
        model = SVC(**params)

        scores = cross_validate(model,
                                X=self.X, y=self.y,
                                n_jobs=-1)
        return scores['test_score'].mean()

# チューニングの対象クラスを設定
objective = Objective_SVM(X_train, y_train)
study = optuna.create_study(direction='maximize')
# 最大で1分間チューニングを実行
study.optimize(objective, timeout=60)
# ベストのパラメーターの出力
print('params:', study.best_params)


# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# 探索結果として得られたベストのパラメーターを設定

model = SVC(
    kernel = study.best_params['kernel'],
    C = study.best_params['C'],
    gamma = study.best_params['gamma']
)
# モデルの訓練
model.fit(X_train, y_train)
# テスト用のデータを使用して予測
pred = model.predict(X_test)
# 予測結果とテスト用のデータを使って正解率と、混同行列を出力
print("Accuracy: {:.5f} %".format(100 * accuracy_score(y_test, pred)))
print(confusion_matrix(y_test, pred))

