"""
-------------------------------------------------
   File Name：  run
   Description : 在正常数据上训练LSTM模型
   Author :  yun
   date：  2019/2/18
   Change Activity:  2019/2/18:
-------------------------------------------------
"""
__author__ = 'yun'

import pandas as pd
import keras
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from base import traversalDir_FirstDir, merge
from model import build_model
from get_data import GetData, z_norm
import matplotlib.pyplot as plt
import numpy as np


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

if __name__ == "__main__":
    # 设置超参数
    batch_size = 128
    epochs = 5
    sequence_length = 300

    # 设置GPU的动态增长
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    # 训练模型
    read_path = "./data/train"
    read_file_list = traversalDir_FirstDir(read_path)
    data = merge(read_file_list)
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]
    data = data[diying_attribute]
    # 数据标准化
    data = z_norm(data)

    for i in range(0, 6):
        # 创建模型
        model = build_model(sequence_length)

        # 得到训练数据
        tmp = data.iloc[:, i]
        train_x, train_y = GetData(tmp.values, sequence_length)
        print(str(i)+ data.columns[i] + " start!" + "Training...")
        print(train_x.shape)
        print(train_y.shape)
        history = LossHistory()
        model.fit(
                train_x, train_y,
                batch_size=batch_size, nb_epoch=epochs, validation_split=0.05, callbacks=[history])
        model.save("./model/" + str(tmp.name) + ".h5")

        # 绘制损失函数曲线
        plt.switch_backend('agg')
        plt.scatter(np.array(range(len(history.losses))), np.array(history.losses), c='r', marker="*")
        plt.plot(history.losses, color='blue')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig("./picture/" + tmp.name + "_loss.png")
        print("training finished!")
