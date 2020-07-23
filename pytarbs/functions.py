import numpy as np

class Function:

    def __init__(self):
        pass


    def call(self,X):
        pass


    def d(self,X):
        pass


class Dummy(Function):

    def __init__(self):
        pass


    def call(self,X):
        return X


    def d(self,X):
        return np.ones_like(X)


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


class Sigmoid(Function):
    def __init__(self):
	    pass


    def call(self,X):
        s=1/(1+np.exp(-X))  
        return s

  
    def d(self,X):
        s=1/(1+np.exp(-X))
        ds=s*(1-s)  
        return ds


class SoftMaxLoss(Function):
  def __init__(self,epsilon=1e-12):
    self.epsilon=epsilon 
    self.predictions=None

  def call(self,x,targets):
    predictions=softmax(x)
    self.predictions=predictions
    
    c=cross_entropy(predictions, targets, self.epsilon)
    return c


  def d(self,x, targets):
    if self.predictions is None:
      self.predictions=softmax(x)
    c=cross_entropy_grad(self.predictions, targets)
    return softmax_grad(self.predictions)@c.T


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax_grad(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    x=-(targets*np.log(predictions)+(1-targets)*np.log(1-predictions))
    ce = np.sum(x)/N
    return ce

def cross_entropy_grad(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -(targets*(1/predictions)-(1-targets)*1/(1-predictions))

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

