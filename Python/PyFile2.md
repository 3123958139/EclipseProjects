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

  ~~~markdown
  the main steps you will go through
  1. Look at the big picture.
  - Frame the Problem
  - Select a Performance Measure
  - Check the Assumptions
  2. Get the data.
  - Create the Workspace
  - Download the Data
  - Take a Quick Look at the Data Structure
  - Create a Test Set
  3. Discover and visualize the data to gain insights.
  - Visualizing Geographical Data
  - Looking for Correlations
  - Experimenting with Attribute Combinations
  4. Prepare the data for Machine Learning algorithms.
  - Data Cleaning
  - Handling Text and Categorical Attributes
  - Custom Transformers
  - Feature Scaling
  - Transformation Pipelines
  5. Select a model and train it.
  - Training and Evaluating on the Training Set
  - Better Evaluation Using Cross-Validation
  6. Fine-tune your model.
  - Grid Search
  - Randomized Search
  - Ensemble Methods
  - Analyze the Best Models and Their Errors
  - Evaluate Your System on the Test Set
  7. Present your solution.
  8. Launch, monitor, and maintain your system.
  ~~~

  

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



