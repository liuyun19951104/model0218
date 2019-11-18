"""
-------------------------------------------------
   File Name：  base
   Description : 一些供调用的基础函数
   Author :  yun
   date：  2019/2/18
   Change Activity:  2019/2/18:
-------------------------------------------------
"""
__author__ = 'yun'

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_score as p
from sklearn.metrics import recall_score as r
from sklearn.preprocessing import OneHotEncoder

def traversalDir_FirstDir(path):
    ''' 遍历文件夹下的文件
    Args:
        path(str): 文件夹路径
    Returns:
        list: 文件名列表
    '''
    # 定义一个列表，用来存储结果
    list = []
    # 判断路径是否存在
    if (os.path.exists(path)):
        # 获取该目录下的所有文件或文件夹目录
        files = os.listdir(path)
        for file in files:
            list.append(path + "/" + file)
    list.sort()
    return list

def merge(path_list):
    '''
    纵向合并对应文件列表下的数据表
    :param path_list(list): 文件列表
    :return: result(dataframe): 合并之后的数据框dataframe
    '''
    data = []
    for file in path_list:
        tmp = pd.read_csv(file, engine="python")
        data.append(tmp)
    result = pd.concat(data, axis=0)
    result.sort_values('Time', inplace=True)

    return result

'''
对预测结果做一个过滤处理，根据最邻近的十个数判断是否为1
'''
def filter(data):
    '''
    :param data: N array 待处理数据
    :return: result: 处理之后的数据
    '''
    length = len(data)
    result = np.zeros(length)
    for i in range(length):
        if i < 5:
            if np.sum(data[0 : 10]) <= 4:
                result[i] = 0
            else:
                result[i] = 1
        elif i >= 5 and i < length-5:
            if np.sum(data[i-5: i+5]) <= 4:
                result[i] = 0
            else:
                result[i] = 1
        else:
            if np.sum(data[-10: ]) <= 4:
                result[i] = 0
            else:
                result[i] = 1
    return result

'''
计算F分数
'''
def f_score(y_true, y_pred, N):
    '''
    :param y_true: 真实值
    :param y_pred: 预测值
    :param N: F分数中的参数
    :return result: 最终计算得出的F分数
    '''
    num = (1 + N * N) * p(y_true, y_pred) * r(y_true, y_pred)
    deno = N * N * p(y_true, y_pred) + r(y_true, y_pred)
    result = num / deno
    return result

"""
将类标签转化成独热编码
"""
def One_hot(data):
    '''
    :param data: 待处理的标签
    :return result: 处理后的独热编码
    '''
    ohe = OneHotEncoder()
    ohe.fit([[0], [1]])
    result = ohe.transform(data.reshape(-1, 1)).toarray()

    return result


"""
将独热编码转化为类标签
"""
def Onehot_label(one_hots):
    '''
    :param one_hots: 待处理的独热编码
    :return label: 处理之后得到的标签
    '''
    label = [np.argmax(one_hot) for one_hot in one_hots]

    return label

