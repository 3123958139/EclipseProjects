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
