# coding:utf-8
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np


def readCSV(s, path):
    '''
    :param s:
    :param path:
    :return:  训练集 特征 标签  都是二维数组
    '''
    df = pd.read_csv(path)  # dataFrame 类型
    class_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
    train_x = df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Fare']]  # dataFrame 类型
    continuous_columns = ['Fare']  # 连续特征不做one hot 编码
    '''
     df.drop(0,axis=0) 沿着每一列 把第一行去掉
     df.drop(0,axis=1) 沿着每一行 把第一列去掉 
    '''
    train_x = train_x.fillna("-1")

    if s == 'train':
        train_y = df['Survived'].values  # 返回 ndarray  一个维度的ndarray
        train_y = train_y.reshape(-1, 1)  # 变成两个维度的ndarray

    le = LabelEncoder()
    oht = OneHotEncoder()
    

    for index, column in enumerate(class_columns):
        # 把 category 离散数值化
        try:
            train_x[column] = le.fit_transform(train_x[column])
        except:
            pass
        # one hot 编码  返回稠密矩阵 二维数组   oht.fit_transform 会构建一个csr稀疏矩阵
        ont_x = oht.fit_transform(train_x[column].values.reshape(-1, 1)).toarray()
        if index == 0:
            x_t = ont_x
        else:
            # 横向堆叠 one-hot编码后的离散特征稠密矩阵
            x_t = np.hstack((x_t, ont_x))

    x_t = np.hstack((x_t, train_x[continuous_columns]))

    if s == 'train':
        return x_t, train_y

    return x_t
