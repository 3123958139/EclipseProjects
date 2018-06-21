# -*- coding:utf-8 -*-
import shelve

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict

import matplotlib.pyplot as plt
import numpy as np


dbase = shelve.open('dbase')
mnist = dbase['mnist']
dbase.close()
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
#############################################################################
# 画ROC图
sgd_clf = SGDClassifier(random_state=42)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
roc_area = roc_auc_score(y_train_5, y_scores)  # ROC曲线的面积，越大趋近1越好


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('the ROC area is ' + str(roc_area))


plot_roc_curve(fpr, tpr)
plt.show()
