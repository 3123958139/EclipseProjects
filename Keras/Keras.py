# -*- coding:utf-8 -*-
from keras.layers import Dense
from keras.models import Sequential

# 建立序贯模型
model = Sequential()
# 将一些网络层通过.add()堆叠起来构成一个模型
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
# 完成模型的搭建后使用.complie()编译模型，编译模型时必须指明损失函数和优化器
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# 完成编译后按batch对训练数据进行一定次数的迭代来训练网络
model.fit(x_train, y_train, epochs=5, batch_size=32)
# 随后可以使用.evaluate对模型进行评估
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
# 最后对新的数据进行预测
classes = model.predict(x_test, batch_size=128)
