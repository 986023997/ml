import numpy as np

"""
将数据集划分为训练集和测试集
"""
def train_test_split(x,y,test_ratio = 0.2, seed = None):
    
    #校验
    assert x.shape[0] == y.shape[0],"the size of x must be equal to the  size of y"
    assert 0.0 <= test_ratio <= 1,"test_ratio must be in [0,1.0]"
    if seed:
        np.random.seed(seed)
    
    #生成随机索引
    shuffle_indexs = np.random.permutation(len(x))
    shuffle_indexs
    test_size = int(len(x) * test_ratio)
    
    # 划分
    train_indexs = shuffle_indexs[test_size:]
    test_indexs = shuffle_indexs[:test_size]
    x_train = x[train_indexs]
    y_train = y[train_indexs]
    x_test = x[test_indexs]
    y_test = y[test_indexs]
    return x_train,y_train,x_test,y_test

