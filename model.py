"""
-------------------------------------------------
   File Name：  model
   Description : LSTM模型的网络结构
   Author :  yun
   date：  2019/2/18
   Change Activity:  2019/2/18:
-------------------------------------------------
"""
__author__ = 'yun'

from keras.models import Sequential, load_model
from keras.callbacks import History, EarlyStopping, Callback
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
import time
import numpy as np
import os

def build_model(sequence_length=300):
    ''' 构建LSTM预测模型
    Args:
        sequence_length(int): 模型输入序列长度
    Returns:
        model: 构建好的模型
    '''
    model = Sequential()
    layers = {'input': 1, 'hidden1': 64, 'hidden2': 256, 'hidden3': 100, 'output': 1}

    model.add(LSTM(
            input_length=sequence_length - 1,
            input_dim=layers['input'],
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden2'],
            return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
            layers['hidden3'],
            return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
            output_dim=layers['output']))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model