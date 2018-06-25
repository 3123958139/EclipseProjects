
import os
import tensorflow as tf
a = tf.constant(5, name='input_a')
b = tf.constant(3, name='input_b')
c = tf.multiply(a, b, name='mul_c')
d = tf.add(a, b, name='add_d')
e = tf.add(c, d, name='add_e')

sess = tf.Session()

op_b = sess.run(b)
print('b', op_b)
op_e = sess.run(e)
print('e', op_e)
writer = tf.summary.FileWriter('logdir_graph', sess.graph)
os.system('tensorboard --logdir=logdir_graph')
writer.close()
sess.close()
