import numpy as np
from math import sqrt

"""
计算准确率
"""
def accuracy_score(y_true,y_predict):
    assert y_true.shape[0] == y_predict.shape[0],"the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)

"""
mse
"""
def mean_squared_error(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0],"the size of y_true must be equal to the size of y_predict"
    return np.sum((y_true - y_predict) ** 2) / len(y_true)

"""
rmse
"""
def root_mean_squared_error(y_true, y_predict):
    return sqrt(mean_squared_error(y_true, y_predict))

"""
mae
"""
def mean_absoulte_error(y_true, y_predict):
    assert y_true.shape[0] == y_predict.shape[0],"the size of y_true must be equal to the size of y_predict"
    return np.sum(np.absoulte(y_true, y_predict)) / len(y_true)

def r2_score(y_true,y_predict):
    assert y_true.shape[0] == y_predict.shape[0],"the size of y_true must be equal to the size of y_predict"
    return  1 - mean_squared_error(y_true,y_predict) / np.var(y_true)