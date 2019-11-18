"""
-------------------------------------------------
   File Name：  svm_compared
   Description : 与BP神经网络模型对比效果
   Author :  yun
   date：  2019/2/25
   Change Activity:  2019/2/25:
-------------------------------------------------
"""
__author__ = 'yun'
import pandas as pd
import numpy as np
from keras.models import *
from keras.layers import *
from sklearn.metrics import accuracy_score as ac
from sklearn.metrics import precision_score as p
from sklearn.metrics import recall_score as r
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn import svm
from base import traversalDir_FirstDir, merge, One_hot
from get_data import z_norm

"""
配置网络结构
"""
def build_model():
    # 配置网络结构
    model = Sequential()

    # 第一隐藏层的配置：输入109，输出100
    model.add(Dense(100, input_dim=24))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # 第二隐藏层的配置：输入100，输出100
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    # 输出层的配置：输入100，输出3，用了softmax的输出层结构
    model.add(Dense(2))
    model.add(Activation("softmax"))

    # 编译模型，指明代价函数和更新方法
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

    return model


if __name__ == "__main__":

    # 读取地影属性
    diying_attribute = list(pd.read_csv("./parameter/diying.csv", header=None)[0])
    for i in range(len(diying_attribute)):
        diying_attribute[i] = "BX0101_" + diying_attribute[i]

    # 训练集数据读取
    read_path_train = "./data/train"
    file_path_list = traversalDir_FirstDir(read_path_train)
    train_data = merge(file_path_list)[diying_attribute]
    train_data["Class"] = 0

    # 验证集数据读取
    read_path_val = "./data/val"
    file_path_list = traversalDir_FirstDir(read_path_val)
    val_data = merge(file_path_list)[diying_attribute]
    read_path_val = "./data/val_set"
    file_path_list = traversalDir_FirstDir(read_path_val)
    sources = np.zeros((141017, 24))
    for i in range(len(file_path_list)):
        data = pd.read_csv(file_path_list[i], engine="python")["Class"]
        sources[:, i] = data.values
    sources = np.sum(sources, axis=1)
    source_label = [1 if source > 0 else 0 for source in sources]
    val_data["Class"] = source_label
    # 给异常样本增加十倍
    add_data = val_data[val_data["Class"]==1].copy()
    end_data = pd.DataFrame()
    for j in range(10):
        end_data = end_data.append(add_data, ignore_index=True)
    val_data = pd.concat([val_data, end_data], axis=0)

    # 测试集数据读取
    read_path_val = "./data/test"
    file_path_list = traversalDir_FirstDir(read_path_val)
    test_data = merge(file_path_list)[diying_attribute]
    read_path_val = "./data/test_set"
    file_path_list = traversalDir_FirstDir(read_path_val)
    sources = np.zeros((142276, 24))
    for i in range(len(file_path_list)):
        data = pd.read_csv(file_path_list[i], engine="python")["Class"]
        sources[:, i] = data.values
    sources = np.sum(sources, axis=1)
    source_label = [1 if source > 0 else 0 for source in sources]
    test_data["Class"] = source_label

    # 构建训练集和测试集
    train = pd.concat([train_data, val_data], axis=0)
    train.iloc[:, 0:-1] = z_norm(train.iloc[:, 0:-1].copy())
    train = train.values
    np.random.shuffle(train)
    train_x = train[:, 0:-1]
    train_y = One_hot(train[:, -1])
    test = test_data
    test.iloc[:, 0:-1] = z_norm(test.iloc[:, 0:-1].copy())
    test = test.values
    np.random.shuffle(test)
    test_x = test[:, 0:-1]
    test_y = test[:, -1]

    # 打印输出训练集和测试集的信息
    print("--------------------")
    print("训练集样本大小为：",train_x.shape[0])
    print("训练集正常样本大小为：", train_x.shape[0] - np.sum(train_y[:, 1]))
    print("训练集异常样本大小为：", np.sum(train_y[:, 1]))
    print("测试集样本大小为：",test_x.shape[0])
    print("测试集正常样本大小为：", test_x.shape[0] - np.sum(test_y))
    print("测试集异常样本大小为：", np.sum(test_y))

    # 训练并保存模型
    model = build_model()
    model.fit(train_x, train_y, epochs=10, batch_size=256)
    save_model(model, 'BP_model.h5')

    # 预测
    y_pred = model.predict(test_x)
    y_pred = np.argmax(y_pred, axis=1)
    print("--------------------")
    print("预测结果为：")
    print("Test acc score: {:.6f}".format(ac(test_y, y_pred)))
    print("Test p score: {:.6f}".format(p(test_y, y_pred)))
    print("Test r score: {:.6f}".format(r(test_y, y_pred)))
    print("confusion matrix:")
    print(confusion_matrix(test_y, y_pred))

    '''
    预测结果为：
    Test acc score: 0.975048
    Test p score: 0.000000
    Test r score: 0.000000
    confusion matrix:
    [[138726      0]
    [  3550      0]]
    '''




