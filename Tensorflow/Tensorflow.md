# TensorFlow学习笔记

[TOC]

------

参考书：《TensorFlow实战》（黄文坚）

------

## 准备

### 神经网络

- 神经元、输入层、隐含层、输出层、递归层、神经网络
- 深度神经网络（DNN）、卷积神经网络（CNN）、循环神经网络（RNN）
- 标量、向量、矩阵、张量

### TensorFlow

#### 入门概念

- TensorFlow是个由节点（Node）和边（Edge）构成的有向图（Directed Graph），节点代表一个运算或操作（Operation），可以具有自己的预先设置的属性（Attribution），在有数据的边中流动（Flow）的数据称为张量（Tensor），没有数据流动的边叫做依赖控制（Control Dependencies）
- 对话（Session）是用户使用TensorFlow的交互接口，用户可以通过Session的Extend方法添加新的节点和边来创建图，用户提供输入数据，进而通过Session的Run方法执行图
- TensorFlow有一个重要组件Client，它通过Session的接口与Master及多个Worker相连，其中每一个Worker可以与多个硬件设备Device相连，Client通过Session沟通Master指导Worker管理Device执行Graph，TensorFlow有两种实现：单机模式——Client、Master、Worker全部在同一台机器的同一个进程中，分布模式——Client、Master、Worker可以在不同机器的不同进程中
- TensorFlow提供三种不同的加速模式：数据并行、模型并行、流水并行

#### 入门实例

~~~python
# -*- coding:utf-8 -*-
import os
import tensorflow as tf
# 数据流图的定义，把图画出来
'''
数据流图的每一个节点都是一种计算或操作，
a=tf.constant指a这个节点是接受一个输入a经过无操作constant得到一个输出a，
c=tf.add指c这个节点接受a、b两个输入进行加法操作add得到一个输出c=a+b，
也可以在节点处使用tf.holder占个空位置，最后再代入数据，
'''
a = tf.constant(5, name='input_a')  # name是节点标签，相当于注释
b = tf.constant(3, name='input_b')
c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')
# 建立对话，数据流图建好后实际上还没有运行，需要Session对话再运行
sess = tf.Session()
# 使用.run执行节点的操作，得到节点的输出
op_b = sess.run(b)
print('b节点的输出：', op_b)
op_e = sess.run(e)
print('e节点的输出：', op_e)
# 可以把图的数据导出然后用tensorboard展示出来
writer = tf.summary.FileWriter('logdir_graph', sess.graph)
# 使用tensorboard展示，在浏览器打开地址即可
os.system('tensorboard --logdir=logdir_graph')
# 最后需关闭对话
writer.close()
sess.close()

~~~

![](/pics/tf_1.jpg)

#### 进阶实战

- 自编码器

  - 描述

    - 使用无监督学习逐层对特征进行降维提取，将网络的权重初始化到一个比较好的位置，辅助后面的监督学习
    - 输入与输出一致，使用稀疏的高阶特征重构自己，让中间隐含层节点的数量小于输入层、输出层节点的数量，相当于降维，给数据加入噪声进行随机遮挡，从而只能学习数据频繁出现的模式和结构

  - 实现

    ~~~python
    # -*- coding:utf-8 -*-
    from tensorflow.examples.tutorials.mnist import input_data
    
    import numpy as np
    import sklearn.preprocessing as prep
    import tensorflow as tf
    
    
    def xavier_init(fan_in, fan_out, constant=1):
        '''
                  参数初始化方法xavier initialization，
                  它可以根据某一层网络的输入、输出节点数量自动调整最合适的分布
        '''
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low,
                                 maxval=high,
                                 dtype=tf.float32)
    # 定义一个去噪自编码的class
    
    
    class AdditiveGaussianNoiseAutoencoder(object):
        def __init__(self,
                     n_input,  # 输入变量数
                     n_hidden,  # 隐含层节点数
                     transfer_function=tf.nn.softplus,  # 隐含层激活函数
                     optimizer=tf.train.AdamOptimizer(),  # 优化器
                     scale=0.1):  # 高斯噪声系数
            self.n_input = n_input
            self.n_hidden = n_hidden
            self.transfer = transfer_function
            self.scale = tf.placeholder(tf.float32)
            self.training_scale = scale
            self.weights = self._initialize_weights()
            # 定义网络结构
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            self.hidden = self.transfer(tf.add(
                tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                          self.weights['w1']),
                self.weights['b1']))
            self.reconstruction = tf.add(tf.matmul(self.hidden,
                                                   self.weights['w2']),
                                         self.weights['b2'])
            # 定义自编码器的损失函数
            self.cost = 0.5 * tf.reduce_sum(tf.pow(
                tf.subtract(self.reconstruction, self.x), 2.0))
            self.optimizer = optimizer.minimize(self.cost)
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
    
        def _initialize_weights(self):
            all_weights = dict()
            all_weights['w1'] = tf.Variable(
                xavier_init(self.n_input, self.n_hidden))
            all_weights['b1'] = tf.Variable(
                tf.zeros([self.n_hidden], dtype=tf.float32))
            all_weights['w2'] = tf.Variable(
                tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
            all_weights['b2'] = tf.Variable(
                tf.zeros([self.n_input], dtype=tf.float32))
            return all_weights
    
        def partial_fit(self, X):
            cost, opt = self.sess.run((self.cost, self.optimizer),
                                      feed_dict={self.x: X,
                                                 self.scale: self.training_scale})
            return cost
    
        def calc_total_cost(self, X):
            return self.sess.run(self.cost,
                                 feed_dict={self.x: X,
                                            self.scale: self.training_scale})
    
        def transform(self, X):
            return self.sess.run(self.hidden,
                                 feed_dict={self.x: X,
                                            self.scale: self.training_scale})
    
        def generate(self, hidden=None):
            if hidden is None:
                hidden = np.random.normal(size=self.weights['b1'])
            return self.sess.run(self.reconstruction,
                                 feed_dict={self.hidden: hidden})
    
        def reconstruct(self, X):
            return self.sess.run(self.reconstruction,
                                 feed_dict={self.x: X,
                                            self.scale: self.training_scale})
    
        def getweights(self):
            return self.sess.run(self.weights['w1'])
    
        def getBiases(self):
            return self.sess.run(self.weights['b1'])
    
    
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    
    
    def standard_scale(X_train, X_test):
        preprocessor = prep.StandardScaler().fit(X_train)
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)
        return X_train, X_test
    
    
    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]
    
    
    X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
    n_samples = int(mnist.train.num_examples)
    training_epochs = 20
    batch_size = 128
    display_step = 1
    autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                                   n_hidden=200,
                                                   transfer_function=tf.nn.softplus,
                                                   optimizer=tf.train.AdamOptimizer(
                                                       learning_rate=0.001),
                                                   scale=0.01)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(n_samples / batch_size)
        for i in range(total_batch):
            batch_xs = get_random_block_from_data(X_train, batch_size)
            cost = autoencoder.partial_fit(batch_xs)
            avg_cost += cost / n_samples * batch_size
    
        if epoch % display_step == 0:
            print('epoch:', '%04d' % (epoch + 1),
                  'cost=', '{:.9f}'.format(avg_cost))
    
    
    print('total cost:' + str(autoencoder.calc_total_cost(X_test)))
    
    ~~~

    



