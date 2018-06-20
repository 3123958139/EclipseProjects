# -*- coding:utf-8 -*-
import shelve

import matplotlib
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
import numpy as np


path = 'D:\\Program Files\\eclipse-cpp-oxygen-3a-win32-x86_64\\tmp\\EclipseProjects\\Python\\pics\\'
dbase = shelve.open('shelve_dbase')
mnist = dbase['mnist']
dbase.close()

x, y = mnist['data'], mnist['target']
some_digit = x[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,
           interpolation='nearest')
plt.title('x[36000],y[36000]=' + str(y[36000]))
plt.axis('off')
plt.savefig(path + '3-1.png', dpi=75)
plt.close()

# 分割数据集
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
# 训练集打乱顺序
shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]
# 简化问题，只考虑5和非5两类
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# 进行小数据分类尝试
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, y_train_5)
res = sgd_clf.predict([some_digit])
print(res)
