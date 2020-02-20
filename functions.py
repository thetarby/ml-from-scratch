import numpy as np

def relu(X):
   return np.maximum(0,X)

def d_relu(X):
    def f(x):
        if x<=0: return 0
        else: return 1
    func = np.vectorize(f)
    return func(X)

def mse(Pred,Act):
    return (Pred-Act)**2

def d_mse(Pred,Act):
    return 2*Pred-2*Act