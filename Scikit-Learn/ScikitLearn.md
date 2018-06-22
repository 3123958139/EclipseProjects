# Scikit-Learn学习笔记

[TOC]

------

官方帮助文档可参考：**2017-11-09 ApacheCN 开源组织**提供**[scikit-learn 0.18 中文文档](http://cwiki.apachecn.org/pages/viewpage.action?pageId=10030181)** 

------

## 机器学习类型

#### 1. 降维

### A. 有监督学习 - 有监督神经网络

#### 2. 回归：预测数值

#### 3. 分类：预测类别

### B. 无监督学习 - 无监督神经网络

#### 4. 聚类

------

## 机器学习过程

### 1. 准备数据

### 2. 选择模型

### 3. 训练模型

### 4. 模型预测

### 5. 优化模型

------

更详尽的流程可参考（./pics/）：[Machine Learning Project Checklist](/pics/Machine Learning Project Checklist.tif)

------

## 入门实例

~~~python
# -*- coding:utf-8 -*-
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC

import numpy as np
# 1. 数据准备datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
# 2. 模型选择svm
clf = svm.SVC(gamma=0.001, C=100.)
# 3. 训练模型fit
clf.fit(digits.data[:-1], digits.target[:-1])
# 4. 模型预测predict
clf.predict(digits.data[-1:])
# 5. 模型优化set_params
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

~~~





