"""
-------------------------------------------------
   File Name：  svm_compared
   Description : 单纯用来做测试的文件
   Author :  yun
   date：  2019/2/25
   Change Activity:  2019/2/25:
-------------------------------------------------
"""


import pandas as pd
import numpy as np
from keras.models import load_model
import os
import matplotlib.pyplot as plt
from get_data import AddAbnormal
from base import merge
from base import traversalDir_FirstDir
from pylab import *
from numpy import *

def z_norm(data):
    ''' 对数据做归一化处理
    Args:
        data(dataframe): 待处理的数据
    Returns:
        result: 归一化之后的数据
    '''
    standard = pd.read_csv("./data/Mean_Std.csv", index_col="Label")
    for i in range(data.shape[1]):
        tmp = data.iloc[:, i]
        if standard.loc[tmp.name, "Std"] != 0:
            data.iloc[:, i] = (tmp.values - standard.loc[tmp.name, "Mean"]) / standard.loc[tmp.name, "Std"]
        else:
            data.iloc[:, i] = 0

    return data

'''
对测试集进行格式转换
'''
def test_GetData(data, sequence_length=300):
    ''' 构建模型的输入数据
    Args:
        data(N array): 待处理的数据
    Returns:
        X, Y: 返回的转化后的模型输入数据
    '''
    result = []
    for i in range(len(data)-sequence_length):
        result.append(data[i: i+sequence_length])
    result = np.array(result) # shape (samples, sequence_length)

    X = result[:, 0:-1]
    Y = result[:, -1]

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y


if __name__ == "__main__":

    # 绘制正常数据曲线及其预测曲线
    read_path = r"./data/train"
    read_file_list = traversalDir_FirstDir(read_path)
    data = merge(read_file_list)
    data.set_index(["Time"], inplace=True)
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]
    data = data[diying_attribute]
    # data = data.iloc[0:9000, :]

    for i in range(data.shape[0]):
        data_copy = data.iloc[:, i:i+1].copy()
        data.iloc[:, i:i+1] = z_norm(data_copy)
        tmp = data.iloc[:, i].values
        plt.plot(tmp[5000:20000], c="b")
        # tmp_x, tmp_y = test_GetData(tmp)
        # new_model = load_model('./model/' + data.columns[i] + ".h5")
        # result = new_model.predict(tmp_x, batch_size=256)
        # gaps = np.abs(result[:, 0] - tmp_y)
        # plt.subplot(311)
        # plt.plot(tmp[299:-1], c="k")
        # plt.subplot(312)
        # plt.plot(result[:, 0], c="k")
        # plt.subplot(313)
        # plt.plot(gaps, c="k")
        plt.show()



    # # 绘制异常数据曲线及其预测曲线
    # read_path = r"./data/val_set"
    # read_file_list = traversalDir_FirstDir(read_path)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    #
    # for i in range(len(read_file_list)):
    #     data = pd.read_csv(read_file_list[i], engine="python", index_col="Time").iloc[0:9000, :]
    #     data_copy = data.iloc[:, i:i+1].copy()
    #     data.iloc[:, i:i+1] = z_norm(data_copy)
    #     tmp = data.iloc[:, i].values
    #     tmp_x, tmp_y = test_GetData(tmp)
    #     new_model = load_model('./model/' + data.columns[0] + ".h5")
    #     result = new_model.predict(tmp_x, batch_size=256)[:, 0]
    #     print(result)
    #     gaps = np.abs(result - tmp_y)
    #     tmp = tmp[299:-1]
    #     plt.subplot(211)
    #     plt.plot(tmp[3600:3900], c="b", label="真实值")
    #     plt.plot(result[3600:3900], c="y", label="预测值")
    #     # plt.plot(np.arange(3701, 3751), tmp[3701: 3751], c="r")
    #     # plt.plot(np.arange(3701, 3751), tmp[3701: 3751], c="r")
    #     # plt.plot(np.arange(5701, 5751), tmp[5701: 5751], c="r")
    #     # plt.plot(np.arange(7701, 7751), tmp[7701: 7751], c="r")
    #     plt.title("LSTM预测效果图")
    #     legend(loc=0,)
    #     # plt.subplot(312)
    #     # plt.plot(result, c="k")
    #     # plt.plot(np.arange(3701, 3751), result[3701: 3751], c="r")
    #     # plt.plot(np.arange(5701, 5751), result[5701: 5751], c="r")
    #     # plt.plot(np.arange(7701, 7751), result[7701: 7751], c="r")
    #     plt.subplot(212)
    #     plt.plot(gaps[3600:3900], c="k", label="正常值")
    #     plt.plot(np.arange(101, 153), gaps[3701: 3753], c="r", label="故障值")
    #     legend(loc=0, )
    #     # plt.plot(np.arange(3701, 3753), gaps[3701: 3753], c="r")
    #     # plt.plot(np.arange(5701, 5753), gaps[5701: 5753], c="r")
    #     # plt.plot(np.arange(7701, 7753), gaps[7701: 7753], c="r")
    #     plt.show()


    # # 计算共有多少条记录
    # N = 0
    # read_path = r"./data/train"
    # read_file_list = traversalDir_FirstDir(read_path)
    # data = merge(read_file_list)
    # N = N + data.shape[0]
    # read_path = r"./data/test"
    # read_file_list = traversalDir_FirstDir(read_path)
    # data = merge(read_file_list)
    # N = N + data.shape[0]
    # read_path = r"./data/val"
    # read_file_list = traversalDir_FirstDir(read_path)
    # data = merge(read_file_list)
    # N = N + data.shape[0]
    # num = 0
    # read_path = "./data/test_set"
    # read_file_list = traversalDir_FirstDir(read_path)
    # # 得到测试数据的真实标签
    # sources = np.zeros((142276, 24))
    # for i in range(len(read_file_list)):
    #     data = pd.read_csv(read_file_list[i], engine="python")["Class"]
    #     sources[:, i] = data.values
    # sources = np.sum(sources, axis=1)
    # source_label = [1 if source > 0 else 0 for source in sources]
    # num = num + np.sum(source_label)
    # read_path = "./data/val_set"
    # read_file_list = traversalDir_FirstDir(read_path)
    # # 得到测试数据的真实标签
    # sources = np.zeros((141017, 24))
    # for i in range(len(read_file_list)):
    #     data = pd.read_csv(read_file_list[i], engine="python")["Class"]
    #     sources[:, i] = data.values
    # sources = np.sum(sources, axis=1)
    # source_label = [1 if source > 0 else 0 for source in sources]
    # num = num + np.sum(source_label)
    # print(N)
    # print(num)



