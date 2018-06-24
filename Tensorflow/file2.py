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
