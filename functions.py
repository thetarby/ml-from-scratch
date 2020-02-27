import numpy as np

def relu(X):
    return np.maximum(0,X)

def d_relu(X):
    def f(x):
        if x<=0: return 0
        else: return 1
    func = np.vectorize(f)
    return func(X)

def tanh(X):
    t=(np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    dt=1-t**2
    return t

def d_tanh(x):
    t=(np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    dt=1-t**2
    return dt

def mse(Pred,Act):
    return (Pred-Act)**2

def d_mse(Pred,Act):
    return Pred-Act

