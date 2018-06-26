# -*- coding:utf-8 -*-
import os

import numpy as np
import pandas as pd
import tensorflow as tf


#=========================================================================
# 随机生成一些数据来用
#=========================================================================
raw_x = np.random.RandomState(1234).rand(32, 2)
raw_y = [[int(i + j < 1)] for i, j in raw_x]
df = pd.DataFrame(data={'x1': raw_x[:, 0].tolist(),
                        'x2': raw_x[:, 1].tolist(),
                        'y': [i[0] for i in raw_y]},
                  index=[i for i in range(len(raw_x))])
df.to_csv('csv_file.csv', index=False)
#=========================================================================
# 将dataframe格式或array格式的数据转为tensor格式，其他的数据分割、塑形等操作也可放在这部分
#=========================================================================


def transformRawData(csv_file='csv_file.csv'):
    df = pd.read_csv(csv_file)

    input_x = np.array([df[['x1', 'x2']].iloc[i] for i in range(len(df))])

    label_y = [[df['y'].iloc[i]] for i in range(len(df))]

    return input_x, label_y


transformRawData()
