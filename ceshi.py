"""
-------------------------------------------------
   File Name：  ceshi
   Description : 计算测试实验结果
   Author :  yun
   date：  2019/3/5
   Change Activity:  2019/3/5:
-------------------------------------------------
"""
__author__ = 'yun'

import pandas as pd
import numpy as np
from base import traversalDir_FirstDir
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_score as p
from sklearn.metrics import recall_score as r

read_path = "./data/test_set"
read_file_list = traversalDir_FirstDir(read_path)

# 得到测试数据的真实标签
sources = np.zeros((142276, 24))
for i in range(len(read_file_list)):
    data = pd.read_csv(read_file_list[i], engine="python")["Class"]
    sources[:, i] = data.values
sources = np.sum(sources, axis=1)
source_label = [1 if source > 0 else 0 for source in sources]


num_3 = np.load("num_3.txt.npy")
num_N = np.load("num_N.txt.npy")
one_3 = np.load("one_3.txt.npy")
one_N = np.load("one_N.txt.npy")

print("--------------------------------------------")
print("num_3")
print("Test acc score: {:.6f}".format(ac(source_label, num_3)))
print("Test p score: {:.6f}".format(p(source_label, num_3)))
print("Test r score: {:.6f}".format(r(source_label, num_3)))
data = pd.DataFrame()
data["y_true"] = source_label
data["y_pred"] = num_3
print("TP",data[(data["y_pred"]==1) & (data["y_true"]==1)].shape[0])
print("FP",data[(data["y_pred"]==1) & (data["y_true"]==0)].shape[0])


print("--------------------------------------------")
print("num_N")
print("Test acc score: {:.6f}".format(ac(source_label, num_N)))
print("Test p score: {:.6f}".format(p(source_label, num_N)))
print("Test r score: {:.6f}".format(r(source_label, num_N)))
data = pd.DataFrame()
data["y_true"] = source_label
data["y_pred"] = num_N
print("TP",data[(data["y_pred"]==1) & (data["y_true"]==1)].shape[0])
print("FP",data[(data["y_pred"]==1) & (data["y_true"]==0)].shape[0])

print("--------------------------------------------")
print("one_3")
print("Test acc score: {:.6f}".format(ac(source_label, one_3)))
print("Test p score: {:.6f}".format(p(source_label, one_3)))
print("Test r score: {:.6f}".format(r(source_label, one_3)))
data = pd.DataFrame()
data["y_true"] = source_label
data["y_pred"] = one_3
print("TP",data[(data["y_pred"]==1) & (data["y_true"]==1)].shape[0])
print("FP",data[(data["y_pred"]==1) & (data["y_true"]==0)].shape[0])

print("--------------------------------------------")
print("one_N")
print("Test acc score: {:.6f}".format(ac(source_label, one_N)))
print("Test p score: {:.6f}".format(p(source_label, one_N)))
print("Test r score: {:.6f}".format(r(source_label, one_N)))
data = pd.DataFrame()
data["y_true"] = source_label
data["y_pred"] = one_N
print("TP",data[(data["y_pred"]==1) & (data["y_true"]==1)].shape[0])
print("FP",data[(data["y_pred"]==1) & (data["y_true"]==0)].shape[0])
