# file2

## TensorFlow套件

**C++底层**，**Python接口**

- TensorFlow
- TensorBoard
- TensorFlow Serving

## 安装TensorFlow

### 准备Python编程IDE

- **Anaconda**
- **Eclipse**

### 安装TensorFlow

- **Windows**下安装TensorFlow
- **Linux**下安装TensorFlow

也可以尝试下Windows和Linux下使用**Docker**安装TensorFlow

## TensorFlow入门基础

### TensorFlow的核心概念是数据流图

***关于图***

- 有向图：使用有向边连接节点构成有向图
- 节点：TensorFlow的一个节点就是一个操作（或计算）
- 边：TensorFlow的边一般有数据在里面流动，对没有数据在里面流动的边我们称其为依赖关系

**关于数据流**

- 数据流：TensorFlow里面的数据流我们叫做张量，标量、向量、矩阵都是张量的低阶形式，标量$a$是0阶1维张量、向量$(a_1,a_2,\cdots,a_n)$是1阶n维张量、矩阵$\begin{equation} \left( \begin{array}{ccc}  a_{11} & \cdots & a_{1m} \\ \cdots & \cdots & \cdots \\  a_{n1} & \cdots & a_{nm} \end{array} \right) \end{equation} $是2阶n$\times$m维张量

### TensorFlow的入门教程

***入门流图***

![](/pics/file2_1.jpg)

***入门代码***

~~~python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
#=========================================================================
# 1. 设计数据流图
#=========================================================================
#=========================================================================
# 2. 按图逐点定义
#=========================================================================
# 显式创建一个Graph对象
graph = tf.Graph()
# 设为默认Graph对象，在该Graph对象下按功能模块建立名称作用域name_scope和操作Operation
with graph.as_default():
    # 具有全局风格的Variable对象都放在variable这个域里面
    with tf.name_scope('variables'):
        # 追踪模型的运行次数
        global_step = tf.Variable(0,
                                  dtype=tf.int32,
                                  trainable=False,  # 训练时不可修改，只能手工修改
                                  name='global_step')
        # 追踪模型所以输出的累加和
        total_output = tf.Variable(0.0,
                                   dtype=tf.float32,
                                   trainable=False,
                                   name='total_output')
    # 模型的核心变换都放在transformation这个域里面
    with tf.name_scope('transformation'):
        # 输入层，输入的是张量，可以通过numpy将普通的数据转成张量输入
        with tf.name_scope('input'):
            a = tf.placeholder(tf.float32,  # 用placeholder占位符，运行时再接受数据
                               shape=[None],  # 输入不限长度的向量
                               name='input_placeholder_a')
        # 隐含层，两个操作节点
        with tf.name_scope('intermediate_layer'):
            b = tf.reduce_prod(a,
                               name='product_b')  # 张量乘法用reduce_prod
            c = tf.reduce_sum(a,
                              name='sum_c')  # 张量求和用reduce_sum
        # 输出层
        with tf.name_scope('output'):
            output = tf.add(b,
                            c,
                            name='output')
    # 对前面定义的两个Variable对象进行更新，放在update这个域里面
    with tf.name_scope('update'):
        # 更新输出
        update_total = total_output.assign_add(output)
        # 更新步数
        increment_step = global_step.assign_add(1)
        # 输出到tensorboard的数据放在这个域里面
        with tf.name_scope('summaries'):
            avg = tf.div(update_total,
                         tf.cast(increment_step,
                                 tf.float32),
                         name='average')
            tf.scalar_summary(b'output',
                              output,
                              name='output_summary')
            tf.scalar_summary(b'sum of outputs over time',
                              update_total,
                              name='total_summary')
            tf.scalar_summary(b'average of outputs over time',
                              avg,
                              name='average_summary')
        # 全局变量的初始化放到这个域里面
        with tf.name_scope('global_ops'):
            # 初始化全局变量
            init = tf.initialize_all_variables()
            # 将所有汇总数据合并到一个节点里面
            merged_summaries = tf.merge_all_summaries()

#=========================================================================
# 3. 运行数据流图
#=========================================================================
# 加载指定Graph并创建对话Session，所有与运行相关的命令都在这个Session里面
sess = tf.Session(graph=graph)
# 导出运行过程中的Graph数据
writer = tf.train.SummaryWriter('./improved_graph', graph)
# 运行前初始化
sess.run(init)

# 创建一个辅助函数run_graph，这样以后便无需反复输入相同的代码。我们希望将输入向量传给该函数，
# 而后者将运行数据流图，并将汇总数据保存下来


def run_graph(input_tensor):
    # placehoder必须这样接受数据
    feed_dict = {a: input_tensor}
    # 运行这些节点
    _, step, summary = sess.run([output, increment_step, merged_summaries],
                                feed_dict=feed_dict)
    # 写入汇总数据
    writer.add_summary(summary,
                       global_step=step)


# 输入数据并运行，可多次运行多个长度
run_graph([2, 8])
run_graph([2, 6, 8, 7, 8])
# 将汇总数据写入磁盘
writer.flush()
# 最后必须释放资源
writer.close()
sess.close()
#=========================================================================
# 4. 展示数据流图
#=========================================================================
# 运行tensorboard然后在浏览器打开http://localhost:6006/
os.system('tensorboard --logdir=improved_graph')

~~~

***补充说明***

实际上我们就是先**1. 设计数据流图**，然后**2. 按图逐点定义**，

- **张量Tensor对象**
- **数据流图Graph对象**
- **名称作用域Namescope对象**
- **操作节点Operation对象**
- **变量Variable对象**
- **占位符Placeholder对象**
- **对话Session对象**
- **执行Run对象**

最后**3. 执行数据流图**并导出事件数据给Tensorboard**4. 展示数据流图**

### TensorFlow的进阶教程

#### 机器学习

1. 降维

- 有监督学习

2. 回归
3. 分类

- 无监督学习

4. 聚类

#### 神经网络

*知识导图*

![神经元、神经网络](/pics/file2_2.png)

*功能模块*

- 自编码机
  - 输入层：数据标准化、加入高斯噪声
  - 隐含层：节点数小于输入层、输出层
  - 输出层：不是聚类而是提取高阶特征
- 单层感知器
- 多层感知器
  - 过拟合：Dropout，随机丢弃Dropout，正则化L1、L2
  - 参数调试：Adagrad，损失函数Loss，自适应优化器Adagrad
  - 梯度弥散：ReLU，隐含层激活函数ReLU，输出层激活函数Sigmod
- DNN深度神经网络

- CNN卷积神经网络
  - 密集数据
  - 抽取特征S-cells：卷积核滤波局部连接
  - 抗形变C-cells：激活函数，最大池化降采样
- RNN循环神经网络
  - 稀疏数据
  - LSTM神经网络

