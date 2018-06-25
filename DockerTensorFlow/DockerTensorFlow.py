import os

import numpy as np
import tensorflow as tf

graph = tf.Graph()
with graph.as_default():
    with tf.name_scope('rawData'):
        input_x = np.random.RandomState(1234).rand(32, 2)
        label_y = [[int(i + j < 1)]for i, j in input_x]
    with tf.name_scope('placeHolders'):
        x = tf.placeholder(tf.float32, shape=(None, 2), name='x')
        y = tf.placeholder(tf.float32, shape=(None, 1), name='y')
    with tf.name_scope('variables'):
        w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1), name='w1')
        w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1), name='w2')
    with tf.name_scope('operations'):
        op1 = tf.matmul(x, w1)
        op2 = tf.matmul(op1, w2)
    y1 = op2
    loss = tf.reduce_mean(tf.square(y - y1))
    opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session(graph=graph) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 3000
    batch = 8
    for i in range(steps):
        start = (i * batch) % 32
        end = start + batch
        sess.run(opt,
                 feed_dict={x: input_x[start:end], y: label_y[start:end]})
        if i % 50 == 0:
            total_loss = sess.run(loss, feed_dict={x: input_x, y: label_y})
            print(total_loss)
    with tf.summary.FileWriter('graph', sess.graph):
        os.system('tenserboard --logdir=graph')
