"""
-------------------------------------------------
   File Name：  get_data
   Description : 得到预测结果标签
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
from sklearn.metrics import f1_score as f1
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

    # 为每个属性选取合适的阈值存储
    # 读取数据并标准化
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]
    mean_and_std = pd.read_csv("./data/Interval_Mean_Std.csv", index_col="Label")
    attribute_N = pd.read_csv("./data/attribute_N.csv", index_col="Label")
    read_path = "./data/test_set"
    read_file_list = traversalDir_FirstDir(read_path)

    # 得到测试数据的真实标签
    sources = np.zeros((142276, 24))
    for i in range(len(read_file_list)):
        data = pd.read_csv(read_file_list[i], engine="python")["Class"]
        sources[:, i] = data.values
    sources = np.sum(sources, axis=1)
    source_label = [1 if source > 0 else 0 for source in sources]

    # 对数据做预测并得到相应的预测标签
    aims = np.zeros((142276, 24))
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
        N = attribute_N.loc[data.columns[0], "N"]
        label = data.iloc[299: -1, 1].values

        y_pred = [1 if gap > MEAN + N*STD else 0 for gap in gaps]
        # 打印单维预测信息
        aims[299:-1, read_file_list.index(file)] = y_pred
        y_pred = filter(y_pred)
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

    # 计算并存储最后一个阈值
    source_label = source_label
    aims = np.sum(aims, axis=1)
    f = open('./data/end_N.txt')
    end_N = int(float(f.read()))
    f.close()
    # f1分数
    end_N = 5
    aim_label = [1 if aim > end_N else 0 for aim in aims]

    # 打印输出并存储阈值
    print("---------------------")
    aim_label = filter(aim_label)
    print("最终的结果为： ")
    print("非3倍方差")
    print("非固定阈值：")
    print("阈值为： ", end_N)
    print("Test acc score: {:.6f}".format(ac(source_label, aim_label)))
    print("Test p score: {:.6f}".format(p(source_label, aim_label)))
    print("Test r score: {:.6f}".format(r(source_label, aim_label)))
    print("confusion matrix:")
    print(confusion_matrix(source_label, aim_label))

    np.save("num_N.txt", np.array(aim_label))












