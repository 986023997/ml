import numpy as np

"""
计算准确率
"""
def accuracy_score(y_true,y_predict):
    assert y_true.shape[0] == y_predict.shape[0],"the size of y_true must be equal to the size of y_predict"
    return sum(y_true == y_predict) / len(y_true)