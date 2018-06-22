# TensorFlow学习笔记

[TOC]

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