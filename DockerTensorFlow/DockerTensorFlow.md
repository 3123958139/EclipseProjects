# TensorFlow笔记：搭建神经网络

## 入门概念

- 张量tensor、数据转换transforming data

- 常量constant、变量variable、占位符placeholder

- 前向传播：操作operation

- 反馈调节：损失函数Loss function、优化器Optimization

- 图graph、

- ~~~
  # coding:utf-8
  import os
  
  import numpy as np
  import tensorflow as tf
  
  
  input_x = np.random.RandomState(1234).rand(32, 2)
  label_y = [[int(i + j < 1)]for i, j in input_x]
  
  x = tf.placeholder(tf.float32,
                     shape=(None, 2),
                     name='x')
  y = tf.placeholder(tf.float32,
                     shape=(None, 1),
                     name='y')
  
  w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1),
                   name='w1')
  w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1),
                   name='w2')
  # 第一层前向传播
  op1 = tf.matmul(x, w1)
  # 第二层前向传播
  op2 = tf.matmul(op1, w2)
  y1 = op2
  loss = tf.reduce_mean(tf.square(y - y1))
  train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
  with tf.Session()as sess:
      init_op = tf.global_variables_initializer()
      sess.run(init_op)
      steps = 3000
      batch = 8
      for i in range(steps):
          start = (i * batch) % 32
          end = start + batch
          sess.run(y1, feed_dict={x: [[0.7, 0.5]]})
          if i % 500 == 0:
              total_loss = sess.run(loss, feed_dict={x: X, y: Y})
  
  ~~~

- 