import numpy as np

class Function:

    def __init__(self):
        pass


    def call(self,X):
        pass


    def d(self,X):
        pass


class Relu(Function):

    def __init__(self):
        pass


    def call(self,X):
        return np.maximum(0,X)


    def d(self,X):
        def f(x):
            if x<=0: return 0
            else: return 1
        func = np.vectorize(f)
        return func(X)


class Mse(Function):

    def __init__(self):
        pass


    def call(self,Pred,Act):
        return 0.5*(Pred-Act)**2

        
    def d(self,Pred,Act):
        return Pred-Act

class LeakyRelu(Function):

    def __init__(self,alpha=0.01):
	    self.alpha=alpha


    def call(self,X):
        return np.where(X > 0, X, X * self.alpha) 

        
    def d(self,X):
        dx = np.ones_like(X)
        dx[X < 0] = self.alpha
        return dx

def tanh(X):
    t=(np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    dt=1-t**2
    return t

def d_tanh(X):
    t=(np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    dt=1-t**2
    return dt

def sigmoid(X):
    s=1/(1+np.exp(-X))  
    return s

def d_sigmoid(X):
    s=1/(1+np.exp(-X))
    ds=s*(1-s)  
    return ds


def mse(Pred,Act):
    return 0.5*(Pred-Act)**2

def d_mse(Pred,Act):
    return Pred-Act

