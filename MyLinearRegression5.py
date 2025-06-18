import numpy as np
from metrics import r2_score

class SimpleLinearRegression5:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        
    
    def dJ_sgd(self, theta, x_b_i, y_i):
        return x_b_i.T.dot(x_b_i.dot(theta) - y_i) * 2.
    
    def learning_rate(self,t,t0,t1):
        
        
        return t0/(t + t1)
    
    def sgd(self, x_b, y, initial_theta, n_iters=5,t0 = 5,t1 = 50):
        theta = initial_theta
        iter_count = 0
        m = len(x_b)
        for iter_ in range(n_iters):
            indexs = np.random.permutation(m)
            x_b_new = x_b[indexs]
            y_new = y[indexs]
            for i in range(m):
                gradient = self.dJ_sgd(theta, x_b_new[i], y_new[i]) 
                last_theta = theta  # 移除多余的分号
                theta = theta - self.learning_rate(iter_ * m + i,t0,t1) * gradient
        return theta
    
    
    def fit_sgd(self, x_train, y_train, n_iters=5,t0 = 5,t1 = 50):
        assert x_train.shape[0] == y_train.shape[0], "the size of x_train must be equal to the size of y_train"

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = self.sgd(x_b, y_train, initial_theta, n_iters,t0,t1)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, x_predict):
        assert self.coef_ is not None and self.interception_ is not None, "must fit before predict"
        
        # 确保输入是二维数组
        if x_predict.ndim == 1:
            x_predict = x_predict.reshape(-1, 1)
            
        assert x_predict.shape[1] == len(self.coef_), "the feature num of x_predict must be equal to the size of x_train"
        x_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        
        return x_b.dot(self._theta)
  
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)
    
    
    
    def __repr__(self):
        return "SimpleLinearRegression4()"