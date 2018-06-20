# 基于Scikit-Learn和TensorFlow库的机器学习

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

- 机器学习的分类

  - 训练时是否有监督
    - 有监督：分类和回归
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

- 机器学习的标准步骤

  1. 观察数据，将数据拆为训练集、验证集和测试集
  2. 用**验证集选择模型**
  3. 用**训练集拟合模型**，找出最小损失的最优参数，用**测试集测试模型**
  4. 使用训练好的模型进行泛化推断

  ------

  the main steps you will go through

  **Working with Real Data**

  最好拿真实的数据做机器学习，不要用人工制造的数据集

  1. **Look at the big picture.**
  - Frame the Problem

    不要为了建模而建模，最好一开始就问下你老板，“预期从模型中得到什么收益？“因为这将决定后面建模的一系列步骤

  - Select a Performance Measure

    选择评价指标

  - Check the Assumptions

    列出和验证到目前为止你们所提出的所有假设前提，不当的假设很可能在早期就引起严重的问题

  2. **Get the data.**

  - Create the Workspace

    配置好你的工作环境

  - Download the Data

    掌握些下数据的技巧

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
  - Data Cleaning

    缺失值、异常值要么去掉*df.dropna()*要么替换*df.replace()*

  - Handling Text and Categorical Attributes

    分类变量重编码

  - Custom Transformers

    自定义变换

  - Feature Scaling

    各种标准化*from sklearn.preprocessing import StandardScaler*

  - Transformation Pipelines
  5. **Select a model and train it.**
  - Training and Evaluating on the Training Set

    基于前面几步的准备工作，这一步很简单了，但需注意模型的过拟合和欠拟合问题

  - Better Evaluation Using Cross-Validation

    使用交叉验证法进行训练和模型选择更好

  6. **Fine-tune your model.**
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

  

- 机器学习的主要难点

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

### 神经网络与深度学习



