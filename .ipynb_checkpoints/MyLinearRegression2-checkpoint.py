import numpy as np

class SimpleLinearRegression2:
    def __init__(self):
        self.a_ = None
        self.b_ = None
        
        
    def fit(self,x_train,y_train):
        assert x_train.ndim == 1,"invalid x_train"
        assert len(x_train) == len(y_train),"the size of x_train must be equal to the size of y_train"
        
        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)
        
        #分子部分
        num = (x_train - x_mean).dot(y_train - y_mean)
        #分母部分
        d = (x_train - x_mean).dot(x_train - x_mean)
        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self
        
    def predict(self,x_predict):
        assert x_predict.ndim == 1,"invalid x_predict"
        assert self.a_ is not None and self.b_ is not None, "must fit before predict"
        
        return np.array([self._predict(x) for x in x_predict])
    
    def _predict(self,x):
        return self.a_ * x + self.b_
        
        
        
        