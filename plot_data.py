"""
-------------------------------------------------
   File Name：  plot_data
   Description : 绘图观察异常数据
   Author :  yun
   date：  2019/2/18
   Change Activity:  2019/2/18:
-------------------------------------------------
"""
__author__ = 'yun'

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from get_data import AddAbnormal
from base import merge
from base import traversalDir_FirstDir

if __name__ == "__main__":
    # 构造异常数据
    read_path = r"./data/test"
    save_path = r"./data/test_set"
    read_file_list = traversalDir_FirstDir(read_path)
    data = merge(read_file_list)
    data.set_index(["Time"], inplace=True)
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]
    data = data[diying_attribute]

    for i in range(len(diying_attribute)):
        tmp = data[diying_attribute[i]]
        tmp = AddAbnormal(tmp)
        tmp = tmp.round(3)
        tmp.to_csv(save_path + "/" + diying_attribute[i] + ".csv", index=True, index_label="Time")
        print("第" + str(i) + diying_attribute[i] + " finished!")




