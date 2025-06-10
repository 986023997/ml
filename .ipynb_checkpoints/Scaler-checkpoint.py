import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        
    def fit(self,x):
        assert x.ndim == 2, "the dimension of x must be 2"
        self.mean_ = np.array([np.mean(x[:,i]) for i in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[:,i]) for i in range(x.shape[1])])

    def transform(self,x):
        assert x.ndim == 2, "the dimension of x must be 2"
        assert self.mean_ is not None and self.scale_ is not None,"must fit before transform"
        assert x.shape[1] == len(self.mean_),"the feature num of x must be equal to mean_ and scale"
        res = np.empty(shape = x.shape,dtype = float)
        for i in range(x.shape[1]):
            res[:,i] = (x[:,i] - self.mean_[i]) / self.scale_[i]
        return res
        