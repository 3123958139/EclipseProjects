# -*- coding:utf-8 -*-
import shelve

from sklearn.linear_model import SGDClassifier

import numpy as np


dbase = shelve.open('dbase')
mnist = dbase['mnist']
dbase.close()
# X数据集，y目标集
X, y = mnist["data"], mnist["target"]
# 分割成训练集和测试集两部分
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# 训练集打乱顺序
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
# 二元分类问题
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# 一个简单的分类三步曲
sgd_clf = SGDClassifier(random_state=42)  # 1. 选择分类器
sgd_clf.fit(X_train, y_train_5)  # 2. 训练分类器
some_digit = X[36000]
pred = sgd_clf.predict([some_digit])  # 3. 分类器预测
print(pred)
