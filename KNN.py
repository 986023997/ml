import numpy as np
from math import sqrt
from collections import Counter
from metrics import accuracy_score

"""
knn分类器
"""
class KNNClassifier:
    
    def __init__(self,k):
        assert 1 <= k,"k must be valid"
        self.k = k
        self.x_train = None
        self.y_train = None
    
    """
    根据训练数据，训练knn分类器
    """
    def fit(self, x_train, y_train):
        assert self.k <= x_train.shape[0],"the size of x_train must be at least k"
        assert x_train.shape[0] == y_train.shape[0],"the size of x_train must equal to the size of y_train"
        self.x_train = x_train
        self.y_train = y_train
        return self
    
    
    """
    预测分类
    """
    def predict(self, x_predict):
        assert self.x_train is not None and self.y_train is not None,"must fit befor predict"
        assert self.x_train.shape[1] == x_predict.shape[1],"the feature num of x must be equal to x_train"
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)
        

        
    """
    预测分类
    """
    def _predict(self, x):
        #使用断言判断参数
        assert  self.x_train.shape[1] == x.shape[0],"the feature num of x must be equal to x_train"
        #1、计算点的距离
        distinces = [sqrt(np.sum((x_train_item - x) ** 2)) for x_train_item in self.x_train]
        #2、排序,获取下标
        nearest = np.argsort(distinces)
        #3、获取前k个数据，找出标签最多的个数
        topK_y = [self.y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        predict = votes.most_common(1)[0][0]
        return predict

    
    def score(self, x_test,y_test):
        y_predict = self.predict(x_test)
        print(y_predict.shape)
        print(sum(y_test == y_predict))
        print(len(y_test))
        return accuracy_score(y_test,y_predict)