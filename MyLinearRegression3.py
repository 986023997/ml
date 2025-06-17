import numpy as np
from metrics import r2_score

class SimpleLinearRegression3:
    def __init__(self):
        self.coef_ = None
        self.interception_ = None
        self._theta = None
        
        
    def fit_normal(self,x_train,y_train):
        assert x_train.shape[0] == y_train.shape[0],"the size of x_train must be equal to the size of y_train"
        x_b = np.hstack([np.ones((len(x_train),1)),x_train])
        self._theta = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)
        self.interception_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self
        
    def predict(self,x_predict):
        assert self.coef_ is not None and self.interception_ is not None , "must fit before predict"
        assert x_predict.shape[1] == len(self.coef_),"the feature num of x_predict must be equal to the size of x_train"
        x_b = np.hstack([np.ones((len(x_predict),1)),x_predict])
        
        return x_b.dot(self._theta)
  
    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test,y_predict)
        
        
        
        