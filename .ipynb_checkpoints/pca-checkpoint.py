import numpy as np


class PCA:
    def __init__(self, n_components):
        assert n_components >= 1, "n_components must be valid"
        self.n_components = n_components
        self.components_ = None
        
    def demean(self, x):
        return x - np.mean(x,axis = 0)

    def f(self, w, x):
        return np.sum((x.dot(w) ** 2)) / len(x)

    
    def df(self, w, x):
        return x.T.dot(x.dot(w)) * 2 / len(x)


    
    def direction(self,w):
        return w / np.linalg.norm(w)

    
    def first_component(self, x, initial_w, eta, n_iters = 1e4, epsilon = 1e-8):
        w = self.direction(initial_w)
        cur_iter = 0

        while cur_iter < n_iters:
            gradient = self.df(w,x)
            last_w = w
            w = w + eta * gradient
            w = self.direction(w)
            if(abs(self.f(w,x) - self.f(last_w,x)) < epsilon):
                break
            cur_iter += 1
        return w
    
    
    def fit(self, x, eta = 0.01, n_iters = 1e4, epsilon = 1e-8):
        x_new = self.demean(x)
        self.components_ = np.empty(shape=(self.n_components,x.shape[1]))
        
        for i in range(self.n_components):
            initial_w = np.random.random(x_new.shape[1])
            w = self.first_component(x_new,initial_w,eta,n_iters,epsilon)
            self.components_[i,:] = w
            x_new = x_new - x_new.dot(w).reshape([-1,1]) * w
        return self
    
    
    def transform(self, x):
        assert x.shape[1] == self.components_.shape[1]
        return x.dot(self.components_.T)
    
        
    def inverse_transform(self, x):
        assert x.shape[1] == self.components_.shape[1]
        return x.dot(self.components_)