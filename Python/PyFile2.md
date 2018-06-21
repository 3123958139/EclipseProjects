# 基于Scikit-Learn和TensorFlow库的机器学习

[TOC]

## 准备

*基础*

- Python参考手册

*补充*

- Numpy
- Pandas
- Matplotlib

*拓展*

- scikit-learn
- tensorflow

## 开始

reference book：《hands-on machine learning with scikit-learn and tensorflow》

### 机器学习基础

#### **机器学习的不同分类**

- 训练时是否有监督
  - 有监督：分类（regression——predicting classes）和回归（classification——predicting values）
    1. k近邻
    2. 线性回归
    3. 逻辑回归
    4. 支持向量机
    5. 决策树和随机森林
    6. 神经网络
  - 无监督：聚类和降维
    1. k均值
    2. 层次聚类（HCA）
    3. 期望最大化
    4. 主成份分析（PCA）
    5. Kernel PCA
    6. Locally-Linear Embedding（LLE）
    7. t-distributed Stochastic Neighbor EMbedding（t-SNE）
    8. Apriori
    9. Edat
    10. 异常点监测
    11. 关联规则
  - 半监督
    1. restricted Boltzmann machines（RBMs）
  - 强化学习
    1. 深度学习
- 是否是线上增量学习
  - 线下批量学习
  - 线上增量学习
- 是否使用模型拟合数据
  - 基于实例
  - 基于模型

#### **机器学习的标准步骤**

1. 观察数据，将数据拆为训练集、验证集和测试集
2. 用**验证集选择模型**
3. 用**训练集拟合模型**，找出最小损失的最优参数，用**测试集测试模型**
4. 使用训练好的模型进行泛化推断

------

the main steps you will go through

**Working with Real Data**

最好拿真实的数据做机器学习，不要用人工制造的数据集

1. **Look at the big picture.**
宏观一览

- Frame the Problem

  不要为了建模而建模，最好一开始就问下你老板，“预期从模型中得到什么收益？“因为这将决定后面建模的一系列步骤

- Select a Performance Measure

  选择评价指标

- Check the Assumptions

  列出和验证到目前为止你们所提出的所有假设前提，不当的假设很可能在早期就引起严重的问题

2. **Get the data.**

获得数据

- Create the Workspace

  配置好你的工作环境

- Download the Data

  掌握些数据源和下数据的技巧

- Take a Quick Look at the Data Structure

  一览数据的结构，如*df.head()*——多少列多少行？、*df.info()*——每列的数据类型是什么？、*df.value_counts()*——分类数据的概况？、*df.describe()*——数值数据的概况？、*df.boxplot()、df.hist()*——箱线图、直方图

- Create a Test Set

  添加ID列*df.set_index()*，随机挑选20%的数据作为测试集，即80%的数据作为训练集

3. **Discover and visualize the data to gain insights.**

利用图表对数据做进一步的探索，为避免损害原始数据集，在备份上操作*df.copy()*

- Visualizing Geographical Data

  画散点图*df.plot(king="scatter", alpha=0.1)*

- Looking for Correlations

  画相关图*df.corr()*或相关图矩阵*pd.tools.plotting.scatter_matrix*

- Experimenting with Attribute Combinations

  根据相关性做下整合

4. **Prepare the data for Machine Learning algorithms.**
准备数据

- Data Cleaning

  缺失值、异常值要么去掉*df.dropna()*要么替换*df.replace()*

- Handling Text and Categorical Attributes

  分类变量重编码

- Custom Transformers

  自定义变换

- Feature Scaling

  各种标准化*from sklearn.preprocessing import StandardScaler*

- Transformation Pipelines

  流水线
5. **Select a model and train it.**
选择模型与模型训练

- Training and Evaluating on the Training Set

  基于前面几步的准备工作，这一步很简单了，但需注意模型的过拟合和欠拟合问题

- Better Evaluation Using Cross-Validation

  使用交叉验证法进行训练和模型选择更好

6. **Fine-tune your model.**
优化模型

- Grid Search

  网格搜索

- Randomized Search

  随机搜索，蒙特卡洛？

- Ensemble Methods

  多管齐下法

- Analyze the Best Models and Their Errors

  误差分析

- Evaluate Your System on the Test Set

  拿之前准备的测试集试试你的模型

7. **Present your solution.**

基于上面的结果应该可以提出一个解决方案了

8. **Launch, monitor, and maintain your system.**

现在你的机器学习系统可以上线了但是要做好监控

**Try It Out!**

干吧！

------



#### **机器学习的主要难点**

- 训练数据不足
  - 增加数据
- 数据不具代表性
  - 采用更具有代表性的训练数据集
- 数据质量差
  - 异常值：丢弃或调整
  - 缺失值：丢弃或填充
- 特征无关
  - 从已有特征中挑选最有用的
  - 用已有特征合成新的特征
  - 从新数据中提取新的特征

- 过度拟合
  - 选择更简单或参数更少的模型
  - 搜集更多的数据
  - 训练数据进行去噪处理
- 欠拟合
  - 选择更有力或参数更多的模型
  - 使用特征工程提取更有效的特征
  - 降低模型约束

#### **Python数据的存储和持久化操作**

Python的数据持久化操作主要是6类：普通文件、DBM文件、Pickled对象存储、Shelve对象存储、对象数据库存储、关系数据库存储，具体可参考https://www.cnblogs.com/huajiezh/p/5470770.html

1. *普通文件*
2. *DBM文件*
3. *Pickled对象存储*

~~~python
# coding=gbk
import pickle
# 序列化
table = {'a': [1, 2, 3],
         'b': ['spam', 'eggs'],
         'c': {'name': 'bob'}}
print('序列化\t', table)
mydb = open('dbase', 'wb+')  # 主要这里要求wb+以byte格式写入
pickle.dump(table, mydb)
# 反序列化
mydb = open('dbase', 'rb+')  # 注意这里要rb+以byte格式读取
table = pickle.load(mydb)
print('反序列化\t', table)

~~~

> 序列化	 {'a': [1, 2, 3], 'b': ['spam', 'eggs'], 'c': {'name': 'bob'}}
> 反序列化	 {'a': [1, 2, 3], 'b': ['spam', 'eggs'], 'c': {'name': 'bob'}}

4. *Shelve对象存储*

~~~python
# -*- coding:utf-8 -*-
import shelve
# 存数据
dbase = shelve.open("mydbase")
object1 = ['The', 'bright', ('side', 'of'), ['life']]
object2 = {'name': 'Brian', 'age': 33, 'motto': object1}
dbase['brian'] = object2
dbase['knight'] = {'name': 'Knight', 'motto': 'Ni!'}
dbase.close()
# 取数据
dbase = shelve.open("mydbase")
print(dbase.keys())
print(dbase['knight'])
dbase.close()

~~~

> KeysView(<shelve.DbfilenameShelf object at 0x0000000002924A58>)
> {'name': 'Knight', 'motto': 'Ni!'}

5. *对象数据库存储*
6. *关系数据库存储*

#### **分类一**：二元分类的标准流程

*how to train binary classifiers？*

如何进行二元分类？

1. choose the appropriate metric for your task

为你的目标选择合适的指标

- 下载mnist数据集并持久化为shelve格式

~~~python
# -*- coding:utf-8 -*-
import shelve

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')
'''
mnist数据集的描述：
70000个手写体、打了标签、数字的图片，每个图片（instance）放在一个28*28=784个特征（feature）的格子了，每个格子取色从0到255
'''
dbase = shelve.open('dbase')
dbase['mnist'] = mnist
dbase.close()

~~~

**小评：对于这种结构化的数据不便于使用传统的文本文件或数据库表进行存储的可以利用shelve进行持久化处理**

- 读取shelve数据，数据集的分割

~~~python
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

~~~

**小评：目标集相当于给原数据打上了标签，同时需要对数据集进行分割成训练集和目标集两部分，最终应该是有2*2=4个部分的数据集合，无序的训练集需要shuffle打乱处理避免数据的打结，有序的训练集则要慎重不能随便打乱比如时间序列数据**

- 二分类问题

~~~python
# 多分类问题变二分类问题
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
# 一个简单的分类三步曲
sgd_clf = SGDClassifier(random_state=42)  # 1. 选择分类器
sgd_clf.fit(X_train, y_train_5)  # 2. 训练分类器
some_digit = X[36000] # 测试用的一个实例
pred = sgd_clf.predict([some_digit])  # 3. 分类器预测
print(pred)
~~~

> [ True]

**小结：多分类转二分类的处理方法值得注意，在数据处理好后，一个分类文件最简单的就可以分成三步进行：选择、训练、预测**

2. evaluate your classifiers using cross-validation

使用交叉验证法评估你的分类器

~~~python
# -*- coding:utf-8 -*-
import shelve

from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

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
#############################################################################
# 交叉验证
sgd_clf = SGDClassifier(random_state=42)  # sgd分类器，random_state是随机种子数
skfolds = StratifiedKFold(n_splits=3, random_state=42)
'''
将训练/测试数据集划分n_splits个互斥子集，每次用其中一个子集当作验证集，剩下的n_splits-1个作为训练集，进行n_splits次训练和测试，得到n_splits个结果
n_splits：表示划分几等份
shuffle：在每次划分时，是否进行洗牌
①若为Falses时，其效果等同于random_state等于整数，每次划分的结果相同
②若为True时，每次划分的结果都不一样，表示经过洗牌，随机取样的
random_state：随机种子数
'''
for train_index, test_index in skfolds.split(X_train, y_train_5):
    # 分类器
    clone_clf = clone(sgd_clf)
    # 训练集
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    # 验证集
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    # 训练分类器
    clone_clf.fit(X_train_folds, y_train_folds)
    # 分类器预测
    y_pred = clone_clf.predict(X_test_fold)
    # 准确率
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

~~~

> 0.9141
> 0.9669
> 0.96015

~~~python
# 另一种进行交叉验证的方法，结果与上述一致
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
~~~

> [ 0.9141   0.9669   0.96015]

**小结：交叉验证有两种方法进行，明显第二种比较方便，但是需注意，这里的0.9141只是从精度上说明分类器的效果比较好，但是精度通常不是首选的性能度量，特别是在数据分布有偏的情况下，具体的说明见下文**

3. select the precision/recall tradeoff that fits your needs

根据你的需要选择合适的精度

- 分类器效果的分解

```python
# -*- coding:utf-8 -*-
import shelve
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict
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
# 混合矩阵
sgd_clf = SGDClassifier(random_state=42)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
conf_matrix = confusion_matrix(y_train_5, y_train_pred)
print(conf_matrix)

```

> [[53986   593]
>  [ 1606  3815]]

![](/Python/pics/confusion_matrix.png)

| 符号      | 说明                                 |
| --------- | ------------------------------------ |
| TN        | True Negative——非5并判断正确为非5    |
| FN        | False Negative——非5但判断错误为5     |
| TP        | True Positive——5并判断正确为5        |
| FP        | False Negative——5但判断为非5         |
| Precision | 预测精度$Precision=\frac{TP}{TP+FP}$ |
| Recall    | 真阳性比率$Recall=\frac{TP}{TP+FN}$  |

**小结：预测精度（Precision ）与真阳性比率（Recall）从各自维度评价了分类器的效果，注意二者是反向相关关系**

- 分类器性能的综合性评价指标

$$
F_1=\frac{2}{\frac{1}{Precision}+\frac{1}{Recall}}
$$

~~~python
# 评价指标
precision = conf_matrix[1][1] / (conf_matrix[0][1] + conf_matrix[1][1])
recall = conf_matrix[1][1] / (conf_matrix[1][0] + conf_matrix[1][1])
f1_score = 2 / ((1 / precision) + (1 / recall))
print('precision=', precision, '\nrecall=', recall, '\nf1_score=', f1_score)
~~~

> [[53978   601]
>  [ 2238  3183]]
> precision= 0.841173361522 
> recall= 0.587161040398 
> f1_score= 0.691580662683

~~~python
# -*- coding:utf-8 -*-
import shelve

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
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
# 使用decision_function决定均衡点并画图
sgd_clf = SGDClassifier(random_state=42)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

p = precisions[precisions == recalls]  # 均衡点


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
    plt.text(0, p, 'cross point is (0, ' + str(p[0]) + ')')


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

~~~

![](/Python/pics/3_3.png)

**小结：可以通过F1得分和Threhold值来选择Precision和Recall的均衡点**

4. compare various models using ROC curves and ROC AUC scores

使用ROC和ROC AUC来选择模型

~~~python
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
roc_area = roc_auc_score(y_train_5, y_scores) # ROC曲线的面积，越接近1越好


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('the ROC area is ' + str(roc_area))


plot_roc_curve(fpr, tpr)
plt.show()

~~~

![](/Python/pics/3_4.png)

**小结：ROC曲线用于模型的选择，分类器的ROC面积越大则越好，如下图则说明随机森林分类器要比SGD分类器更好**

![](/Python/pics/3_5.png)

#### **分类**二：多元分类问题

#### **训练模型**

#### **SVM支持向量机**

- 适用场合

  - 适用**线性、非线性分类**，**线性、非线性回归**，**离群点监测**
  - 特别适合**中小规模数据量**的、**复杂**的**分类**问题

- 对SVM的一些感性认识

  - **支持向量**：帮助确定边界的点或数据

  - SVM对**特征尺度敏感**，即非标准化与标准化处理的结果有差别
  - SVM对**离群点敏感**，有些离群点会造成**硬边距分类**（不同类别的点可以严格分开）不可能，只能采用**软边距分类**（不同类别的点在边缘处混杂不可分开）进行折中，这时可以考虑**C超参数**进行控制，硬边距问题和软边距问题都是具有线性约束的凸二次优化问题
  - **C超参数**同时也是**调整SVM过拟合和欠拟合**的手段
  - SVM有多种核函数可以使用，比如**线性核**、**高斯RBF核**、**多项式核**等等，一般建议先尝试线性核和高斯RBF核，再尝试别的
  - SVM可以使用**判别函数（decision function）**进行分类预测
  - SVM可以通过**核化（kernelized）**降低求解复杂度
  - 可以构造**在线（online）SVM**
  - 对于**大规模的非线性问题**，需要考虑使用**神经网络**代替

- 一个实例

~~~python
# 导入svm和数据集
from sklearn import svm, datasets
# 调用SVC()
clf = svm.SVC()
# 载入鸢尾花数据集
iris = datasets.load_iris()
X = iris['data']
y = iris['target']
# fit()训练
clf.fit(X, y)
# predict()预测
pre_y = clf.predict(X[5:10])
print(pre_y)
print(y[5:10])
# 导入numpy
import numpy as np
test = np.array([[5.1, 2.9, 1.8, 3.6]])
# 对test进行预测
test_y = clf.predict(test)
print(test_y)

~~~

> [0 0 0 0 0]
> [0 0 0 0 0]
> [2]

#### **决策树**

- 适用场合
  - 与SVM一样，决策树是一种通用的机器学习算法，可以执行**分类**和**回归**任务，甚至**多输出任务**
- 特征描述
  - 非常强大的算法，能够拟合**复杂**的数据集
  - 决策树的许多特性之一是它们只需要**很少的数据准备**。特别是，它们根本**不需要特征缩放或中心化**
  - 决策树是随机森林的**组件**
  - 决策树属于**白盒算法**，你知道它是怎么得到结果的，但是随机森林和神经网络属于**黑盒算法**，它们做出了很好的预测，而且您可以很容易地检查它们为做出这些预测所做的计算，然而，通常很难简单地解释为什么
  - **类概率估计**：决策树还可以估计实例属于特定类别的概率
  - 决策树对训练数据做的假设很少(比如线性模型，后者显然假定数据是线性的)。如果不受约束，树结构将适应训练数据，非常接近，而且**很可能会过度拟合**。这样的模型通常被称为**非参数模型**，不是因为它没有任何参数(通常有很多)，而是由于参数个数在训练前不确定，因此模型结构可以自由地与数据紧密结合。相反，线性模型等参数模型具有预定的参数，因此**自由度有限**，降低了过拟合的风险(但**增加了欠拟合的风险**)
  - 为了避免训练数据的过度拟合，在训练过程中需要限制决策树的自由度。正如你现在所知道的，这被称为**正则化**
  - 决策树喜欢正交的决策边界(所有的分裂都垂直于一个轴)，这使得它们**对训练集旋转非常敏感**
  - 决策树可能是**不稳定**的，因为即使非常小的变异，可能会产生一颗完全不同的树
- 一个实例

~~~python
from sklearn import tree

X = [[0, 0], [1, 1]]
Y = [0, 1]
# 训练
clf = tree.DecisionTreeClassifier()
# 拟合
clf = clf.fit(X, Y)
# 预测
result = clf.predict([[2., 2.]])
print(result)

~~~

> [1]

#### **集成学习与随机森林**

- 适用场合
  - 

- 特点描述

  - 把一组预测者(例如分类器或回归者)的预测汇总在一起，你通常会得到比最好的个体预测器更好的预测。一组预测器被称为集合，因此这种技术被称为**集成学习**，集成学习算法被称为**集成方法**

  - **投票分类器**：有**软投票**和**硬投票**之分

    ![](/pics/7_1.png)

    ![](/pics/7_2.png)

  - 当预测器尽可能地**相互独立**时，集成方法效果最好，获得一组不同的分类器的一种方法是使用非常不同的训练算法，正如刚才讨论的那样。另一种方法是对每个预测器使用相同的训练算法，但在训练集的不同随机子集上对它们进行训练

  - 跟投票的思路不同，**Bagging and Pasting**的思路是同一算法训练多个模型，每个模型训练时只使用部分数据。如果抽样时有放回，称为Bagging，当抽样没有放回，称为Pasting。 预测时，每个模型分别给出自己的预测结果，再将这些结果**聚合**起来，预测器都可以通过不同的CPU核心甚至不同的服务器并行训练。类似地，预测可以并行进行

  ![](/pics/7_3.png)

  - 随机森林一般采用bagging算法训练模型 

- 一个实例

~~~python
# -*- conding:utf-8 -*-

from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# 产生moon数据并分开训练测试集
(X, y) = make_moons(1000, noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# 随机森林
rnd_clf = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
print(y_pred_rf)

~~~

> [1 1 0 1 0 0 0 0 0 1 1 0 0 1 1 0 1 1 0 0 0 0 1 1 1 0 0 0 1 1 0 1 0 1 0 1 0
>  1 0 0 0 1 0 0 0 0 1 1 1 0 1 1 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 1 1 0 1 0
>  0 0 1 1 1 0 0 0 1 0 1 1 1 0 0 0 0 0 1 0 1 1 0 0 0 1 1 0 0 1 0 0 1 0 0 0 1
>  1 0 0 0 1 0 0 0 1 0 1 0 0 1 0 0 0 0 1 1 1 0 1 1 0 0 0 0 0 1 0 0 1 1 1 0 1
>  1 0 1 0 1 1 1 1 1 1 1 0 1 0 0 1 1 1 1 0 0 1 0 1 0 0 1 0 0 1 0 1 1 0 1 0 0
>  1 1 1 1 1 1 1 1 1 0 1 0 1 0 0]

#### **维度归约**

- 一个实例

~~~python
# 生成Swiss roll数据
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

import matplotlib.pyplot as plt
data = make_swiss_roll(n_samples=1000, noise=0.0, random_state=None)
X = data[0]
y = data[1]
# 画3维图
ax = plt.subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
plt.show()
# LLe降维
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
# 画出降为图
X_reduced = lle.fit_transform(X)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y)

~~~

![](/pics/7_4.png)

### 神经网络与深度学习

------



## 突然发现“[ApacheCN 组织资源](http://www.apachecn.org/)”对书全套汉化了，666，致敬！

### 见https://github.com/apachecn/hands_on_Ml_with_Sklearn_and_TF

