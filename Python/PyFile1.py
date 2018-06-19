# -*- coding:utf-8 -*-
import datetime
import matplotlib.pyplot as plt
import tushare as ts


path = 'D:\\Program Files\\eclipse-cpp-oxygen-3a-win32-x86_64\\tmp\\EclipseProjects\\Python'

df = ts.get_k_data('002237')  # tushare的数据接口get_k_data
df['date'] = df['date'].apply(
    lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))  # 日期由string转成datetime
df.to_csv("PyFile1.csv", index=False)  # 保存数据到本地csv
df.set_index('date', inplace=True)  # 将日期设为index便于画图
dfPlot = df.ix[:, :-2]  # 画图OHLC
if __name__ == '__main__':
    dfPlot.plot()
    plt.title('002237 plot')
    plt.savefig(path + '\\Pics\\fig1.png', dpi=75)  # 保存图片
    plt.close()  # 关闭
    dfPlot.hist()
    plt.title('002237 hist')
    plt.savefig(path + '\\Pics\\fig2.png', dpi=75)
    plt.close()
    dfPlot.boxplot()
    plt.title('002237 boxplot')
    plt.savefig(path + '\\Pics\\fig3.png', dpi=75)
    plt.close()
