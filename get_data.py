"""
-------------------------------------------------
   File Name：  get_data
   Description : 老版的构建异常数据
   Author :  yun
   date：  2019/2/18
   Change Activity:  2019/2/18:
-------------------------------------------------
"""
__author__ = 'yun'

import numpy as np
import random
from base import traversalDir_FirstDir
import pandas as pd
from keras.models import load_model
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_score as p
from sklearn.metrics import recall_score as r
import matplotlib.pyplot as plt
from base import filter
from sklearn.metrics import confusion_matrix

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

def z_norm(data):
    ''' 对数据做归一化处理
    Args:
        result(dataframe): 待处理的数据
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

def GetData(data, sequence_length=300):
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
    # 数据的顺序打乱
    np.random.shuffle(result)  # shuffles in-place

    X = result[:, 0:-1]
    Y = result[:, -1]

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y

def AddAbnormal(data):
    ''' 构建故障数据
    Args:
        data(N Series): 待处理的数据
    Returns:
        result: N-2 dataframe 处理后的带有故障的数据
    '''
    data = pd.DataFrame(data)
    data["Class"] = 0
    baseline = 2000
    length = len(data)
    abnormal_length = 50
    # 数据的倍数
    N = (np.max(data.iloc[:, 0].values) - np.min(data.iloc[:, 0].values))/2
    # print("N_all= ")
    # print(N_all)

    while baseline + abnormal_length < length:
        # num是发生故障的属性的个数， start是故障起始属性的下标
        tag = np.random.randint(low=0, high=2)
        # print("--------------------")
        # print("baseline == " + str(baseline))
        # print("num == " + str(num))
        # print("start == " + str(start))
        if tag == 1:
            data.iloc[baseline: baseline+50, 0] = create_abnormal(data.iloc[baseline: baseline+50, 0].values, N)
            data.iloc[baseline: baseline+50, 1] = 1
        else:
            pass
        baseline = baseline + 2000
    return data

def create_abnormal(data, N):
    ''' 对具体的故障数据部位做处理
    Args:
        data(N array): 待处理的数据
        N_all(N array): 倍数
    Returns:
        result: 处理后的带有故障的数据
    '''
    wave = np.random.rand(len(data)) * N + N / 2
    result = data + wave
    return result


if __name__ == "__main__":

    # 设置GPU的动态增长
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    # # 提取出均值方差进行存储
    # read_path = "./data/train"
    # read_file_list = traversalDir_FirstDir(read_path)
    # data = merge(read_file_list)
    # diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    # for i in range(len(diying_attribute)):
    #     diying_attribute[i] = "BX0101_" + diying_attribute[i]
    # data = data[diying_attribute]
    #
    # result = pd.DataFrame(columns=["Mean", "Std"], index=diying_attribute)
    #
    # for i in range(len(diying_attribute)):
    #     tmp = data[diying_attribute[i]].values
    #     result.loc[diying_attribute[i], "Mean"] = tmp.mean()
    #     result.loc[diying_attribute[i], "Std"] = tmp.std()
    # result.to_csv("./data/Mean_Std.csv", index=True, index_label="Label")
    # result = pd.read_csv("./data/Mean_Std.csv", index_col="Label")
    # result = result.round(3)
    # result.to_csv("./data/Mean_Std.csv", index=True, index_label="Label")



    # 提出预测的误差的均值方差进行存储
    # read_path = "./data/train"
    # read_file_list = traversalDir_FirstDir(read_path)
    # data = merge(read_file_list)
    # diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    # for i in range(len(diying_attribute)):
    #     diying_attribute[i] = "BX0101_" + diying_attribute[i]
    # data = data[diying_attribute]
    # data = z_norm(data)
    #
    # result_save = pd.DataFrame(columns=["Mean", "Std"], index=diying_attribute)
    # for i in range(len(diying_attribute)):
    #     print(i, " start！")
    #     new_model = load_model('./model/' + data.columns[i] + ".h5")
    #     tmp = data[diying_attribute[i]].values
    #     tmp_x, tmp_y = test_GetData(tmp)
    #     print(tmp.shape)
    #     print(tmp_x.shape)
    #     print(tmp_y.shape)
    #
    #     # 进行预测
    #     result = new_model.predict(tmp_x, batch_size=256)[:, 0]
    #     interval = np.abs(result - tmp_y)
    #     interval_mean = np.mean(interval)
    #     interval_std = np.std(interval)
    #     result_save.loc[diying_attribute[i], "Mean"] = interval_mean
    #     result_save.loc[diying_attribute[i], "Std"] = interval_std
    #
    # # 存储数据
    # result_save.to_csv("./data/Interval_Mean_Std.csv", index=True, index_label="Label")
    # result_save = pd.read_csv("./data/Interval_Mean_Std.csv", index_col="Label")
    # result_save = result_save.round(3)
    # result_save.to_csv("./data/Interval_Mean_Std.csv", index=True, index_label="Label")


    # 为每个属性选取合适的阈值存储
    # 读取数据并标准化
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]
    mean_and_std = pd.read_csv("./data/Interval_Mean_Std.csv", index_col="Label")
    read_path = "./data/val_set"
    read_file_list = traversalDir_FirstDir(read_path)

    # 选取合适的阈值
    threshold = np.arange(1, 10, 1)
    best_threshold = pd.DataFrame(index=diying_attribute, columns=["N"])
    for file in read_file_list:
        data = pd.read_csv(file, engine="python", index_col="Time")
        data_copy = data.iloc[:, 0:1].copy()
        data.iloc[:, 0:1] = z_norm(data_copy)
        tmp = data.iloc[:, 0].values
        tmp_x, tmp_y = test_GetData(tmp)
        new_model = load_model('./model/' + data.columns[0] + ".h5")
        result = new_model.predict(tmp_x, batch_size=256)
        gaps = np.abs(result[:, 0] - tmp_y)
        MEAN = mean_and_std.loc[data.columns[0], "Mean"]
        STD = mean_and_std.loc[data.columns[0], "Std"]
        label = data.iloc[299: -1, 1].values

        acc = []
        accuracies = []
        for j in threshold:
            y_pred = [1 if gap > MEAN + j*STD else 0 for gap in gaps]
            acc.append(matthews_corrcoef(label, y_pred))
        acc = np.array(acc)
        index = np.where(acc == acc.max())
        accuracies.append(acc.max())
        best_threshold.loc[data.columns[0], "N"] = threshold[index[0][0]]

        # 打印预测信息
        N = threshold[index[0][0]]
        y_pred = [1 if gap > MEAN + N * STD else 0 for gap in gaps]
        # y_pred = filter(y_pred)
        print("---------------------")
        print(data.columns[0], " 调整之后的预测精度展示：")
        print("N = ", N)
        print("mean = ",MEAN)
        print("std = ",STD)
        print("Test acc score: {:.6f}".format(ac(label, y_pred)))
        print("Test p score: {:.6f}".format(p(label, y_pred)))
        print("Test r score: {:.6f}".format(r(label, y_pred)))
        print("confusion matrix:")
        print(confusion_matrix(label, y_pred))

        # 绘图
        plt.switch_backend('agg')
        plt.subplot(2, 1, 1)
        plt.plot(tmp_y, c="b")
        plt.plot(result[:, 0], c="r")
        plt.title(data.columns[0])
        plt.subplot(2, 1, 2)
        plt.plot(gaps, c="b")
        plt.plot((MEAN + N*STD) * np.ones(len(gaps)))
        plt.savefig("./single_result_picture/" + data.columns[0] + ".png")

    # 存储
    best_threshold.to_csv("./data/attribute_N.csv", index=True, index_label="Label")







