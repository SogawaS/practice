#!/usr/bin/env python
# coding: utf-8

# ![表紙](https://www.oreilly.co.jp/books/images/picture978-4-87311-907-6.gif)
# 
# このノートブックはオライリー・ジャパンより発行の書籍[『セキュリティエンジニアのための機械学習』](https://www.oreilly.co.jp/books/9784873119076/)のサンプルコードです。コードの解説等は書籍をご参照ください。なお、このコードを動作させた結果について、著者およびオライリー・ジャパンは一切の責任を負いません。

# In[7]:


#get_ipython().system('wget https://github.com/oreilly-japan/ml-security-jp/raw/master/ch03/dataset.csv')


# In[8]:


import pandas as pd

df = pd.read_csv("./dataset.csv", sep=';')


# In[ ]:


df.head()


# In[ ]:


features= df.iloc[:,14:167]
label = df.iloc[:,-1]
label = label.replace({"benignware":0,"malware":1})


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

classifiers = [RandomForestClassifier(n_estimators=100),
               DecisionTreeClassifier(),
               AdaBoostClassifier()]

for clf in classifiers:
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print("Score of {} is {}".format(clf.__class__.__name__, score*100))

