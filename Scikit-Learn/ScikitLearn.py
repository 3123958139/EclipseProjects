# -*- coding:utf-8 -*-
from sklearn.svm import SVC
import numpy as np
# 数据准备
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)
# 选择SVM模型
clf = SVC()
# 训练带参数优化的模型
clf.set_params(kernel='linear').fit(X, y)  # 线性核
# 拿训练好的模型进行预测
clf.predict(X_test)
# 训练带参数优化的模型
clf.set_params(kernel='rbf').fit(X, y)  # rbf核
# 拿训练好的模型进行预测
clf.predict(X_test)
