"""
knn分类方法
"""
import numpy as np
from math import sqrt
from collections import Counter

def KNN_classify(k, x_train, y_train, x):
    """
    :param k:相识点的数量
    :param x_train: 训练数据样本
    :param y_train: 训练数据标签
    :param x: 新的数据点
    :return: 新的数据点的标签
    """
    
    #使用断言判断参数
    assert  1 <= k <= x_train.shape[0],"k must be valid"
    assert x_train.shape[0] == y_train.shape[0],"the size of x_train must equal to the size of y_train"
    assert  x_train.shape[1] == x.shape[0],"the feature num of x must be equal to x_train"
    
    
    
    #1、计算点的距离
    distinces = [sqrt(np.sum(x_train_item - x) ** 2) for x_train_item in x_train]
    #2、排序
    nearest = np.argsort(distinces)
    #3、获取前k个数据，找出标签最多的个数
    topK_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(topK_y)
    predict = votes.most_common(1)[0][0]
    return predict
