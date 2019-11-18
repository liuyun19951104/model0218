"""
-------------------------------------------------
   File Name：  CreateErrorData
   Description : 构建故障数据
   Author :  yun
   date：  2019/10/31
   Change Activity:  2019/10/31:
-------------------------------------------------
"""
__author__ = 'yun'

import pandas as pd
import numpy as np
from base import traversalDir_FirstDir, merge
import os
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # 绘图观察
    read_path = r"./data/train"
    read_file_list = traversalDir_FirstDir(read_path)
    data = merge(read_file_list)
    data.set_index(["Time"], inplace=True)
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]
    data = data[diying_attribute]

    print("数据形状为：")
    data = data
    print(data.shape)
    print("数据为：")
    print(data)

    # for i in range(len(diying_attribute)):
    #     tmp = data[diying_attribute[i]].values[0: 20000]
    #     print("---------------------------")
    #     plt.plot(tmp, c="b")
    #     plt.title(diying_attribute[i])
    #     plt.show()
