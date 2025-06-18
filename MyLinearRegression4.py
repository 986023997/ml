import numpy as np
from metrics import r2_score

class SimpleLinearRegression4:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        
    def J(self, theta, x_b, y):
        try:
            return np.sum((y - x_b.dot(theta)) ** 2) / len(y)
        except:
            return float("inf")  
        
    def dJ(self, theta, x_b, y):
        """
        for循环的方式
        res = np.empty(len(theta))
        res[0] = np.sum(x_b.dot(theta) - y)
        for i in range(1, len(theta)):
            res[i] = (x_b.dot(theta) - y).dot(x_b[:,i])
        return res * 2 / len(x_b)
        """
        #向量化的方式
        return x_b.T.dot(x_b.dot(theta) - y) * 2. / len(y)
    
    def dJ_sgd(self, theta, x_b_i, y_i):
        return x_b_i.T.dot(x_b_i.dot(theta) - y_i) * 2.
    
    def learning_rate(self,t):
        t0 = 5
        t1 = 50
        return t0/(t + t1)
    
    def sgd(self, x_b, y, initial_theta, n_iters=1e4):
        theta = initial_theta
        iter_count = 0  
        for iter_ in range(n_iters):
            rand_i = np.random.randint(len(x_b))
            gradient = self.dJ_sgd(theta, x_b[rand_i], y[rand_i]) 
            last_theta = theta  # 移除多余的分号
            theta = theta - self.learning_rate(iter_) * gradient
        return theta
    
    
    
    def gradient_descent(self, x_b, y, initial_theta, eta, n_iters=1e4, epsilon=1e-8):
        theta = initial_theta
        iter_count = 0  # 修正变量名
        while iter_count < n_iters:
            gradient = self.dJ(theta, x_b, y)  # 修正变量名
            last_theta = theta  # 移除多余的分号
            theta = theta - eta * gradient
            if abs(self.J(theta, x_b, y) - self.J(last_theta, x_b, y)) < epsilon:
                break
            iter_count += 1
          
        return theta
        
    def fit_gd(self, x_train, y_train, eta=0.01, n_iters=1e4):
        assert x_train.shape[0] == y_train.shape[0], "the size of x_train must be equal to the size of y_train"
        
        # 确保输入是二维数组
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
            
        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = self.gradient_descent(x_b, y_train, initial_theta, eta, n_iters)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self
    
    def fit_sgd(self, x_train, y_train, n_iters=1e4):
        assert x_train.shape[0] == y_train.shape[0], "the size of x_train must be equal to the size of y_train"

        # 确保输入是二维数组
        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)

        x_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        initial_theta = np.zeros(x_b.shape[1])
        self._theta = self.sgd(x_b, y_train, initial_theta, n_iters)
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