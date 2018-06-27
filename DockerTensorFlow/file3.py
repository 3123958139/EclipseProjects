import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#=========================================================================
#
#=========================================================================
Alpha = 1
X = np.random.rand(1000).astype(np.float32)
Beta = 2
Y = Alpha + X * Beta
#=========================================================================
#
#=========================================================================
with tf.Graph().as_default() as g:
    with tf.name_scope('foreward_training'):
        with tf.name_scope('placeholder'):
            x = tf.placeholder(shape=[None], dtype=tf.float32,  name='x')
        with tf.name_scope('constant'):
            alpha = tf.constant(1, dtype=tf.float32, name='alpha')
        with tf.name_scope('variable'):
            beta = tf.Variable(tf.random_normal(shape=[],
                                                dtype=tf.float32, name='beta'))
            error = tf.Variable(tf.random_normal(shape=[],
                                                 dtype=tf.float32, name='error'))
        with tf.name_scope('inference'):
            y = alpha + x * beta + error
    with tf.name_scope('backward_optimize'):
        with tf.name_scope('parameter'):
            training_rate = tf.constant(0.05, name='training_rate')
        with tf.name_scope('optimizer'):
            loss = tf.reduce_mean(tf.square(Y - y), name='loss')
            optimizer = tf.train.GradientDescentOptimizer(
                training_rate).minimize(loss)
#=========================================================================
#
#=========================================================================


def createTrainSaver():
    try:
        saver_path = os.path.dirname(__file__) + r'\file3_train_saver'
        os.system('md '.join(saver_path))
        saver_path = saver_path + '\\'
    except:
        pass
    latest_filename = 'saver.ckpt'
    train_saver = tf.train.Saver(name='train_saver')
    return saver_path, latest_filename, train_saver


with tf.Session(graph=g) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    with tf.summary.FileWriter('file3_graph', sess.graph):
        saver_path, latest_filename, train_saver = createTrainSaver()
        steps = 1000
        plt_data = []
        for step in range(steps):
            sess.run(optimizer, feed_dict={x: X})
            step_alpha = sess.run(alpha)
            step_beta = sess.run(beta)
            step_loss = sess.run(loss, feed_dict={x: X})
            plt_data.append(step_loss)
            if step % 10 == 0:
                print('\tstep =', step,
                      '\talpha=', step_alpha,
                      '\tbeta =', step_beta,
                      '\tloss =', step_loss)
            train_saver.save(sess, save_path=saver_path, global_step=step,
                             latest_filename=latest_filename)
        train_saver.save(sess, save_path=saver_path,
                         global_step=step, latest_filename=latest_filename)
        plt.plot(plt_data)
        plt.title('y=' + str(step_alpha) + '+' + str(step_beta) + 'x')
        plt.show()
#=========================================================================
#
#=========================================================================
os.system('explorer .')
os.system('start C:\\Users\\dengchaohai\\AppData\\Local\\Google\\Chrome' +
          '\\Application\\chrome.exe http://localhost:6006')
os.system('tensorboard --logdir=file3_graph')
