# -*- coding:utf-8 -*-
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC

import numpy as np
# 数据准备datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
# 模型选择svm
clf = svm.SVC(gamma=0.001, C=100.)
# 训练模型fit
clf.fit(digits.data[:-1], digits.target[:-1])
# 模型预测predict
clf.predict(digits.data[-1:])
# 模型优化set_params
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)
clf = SVC()
clf.set_params(kernel='linear').fit(X, y)  # 线性核
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.predict(X_test)
clf.set_params(kernel='rbf').fit(X, y)  # rbf核
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
clf.predict(X_test)
