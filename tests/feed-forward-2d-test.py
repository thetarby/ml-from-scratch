import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
import math 
import numpy as np
import pytarbs.functions as f
from pytarbs.feed_forward.model import *
from pytarbs.feed_forward.layer import *
#42 is the meaning of the life
np.random.seed(42)

model=FeedForward().add_layer(InputLayer(2)).add_layer(DenseLayer(2,16,f.LeakyRelu())).add_layer(DenseLayer(16,16,f.LeakyRelu())).add_layer(OutputLayer(16,1,f.Sigmoid()))
#function to be approximated. this will be the second graph 
#after training finishes. First graph will be prediction of the model
func= lambda x,y:((x>0 and y>0) or (y<0 and x<0))
def train():
    print("-----------------------------beginning------------------")

    for i in range(10000):
        #sample(1)
        x,y=np.random.rand(2)*2-1
        pred=model.forward([[x,y]])

        act= func(x,y)
        loss=f.d_mse(pred,act)
        print("-----------------------pass : {} ----------------".format((x,y)))
        print("prediction : "+str(pred), "act : "+str(act))
        model.backward(loss)

        model.update(lr=0.03)



def sample2d():
    #plot prediction
    f = plt.figure(1)
    plt.plot()
    for i in range(-25,25):
        for j in range(-25,25):
            x=1/50*i
            y=1/50*j
            if model.forward([[x,y]])>0.5:
                plt.scatter(x,y,color='r')
            else:plt.scatter(x,y,color='b')
    plt.show()
    
    #plot actual function
    plt.plot()
    for i in range(-25,25):
        for j in range(-25,25):
            x=1/50*i
            y=1/50*j
            if func(x,y) :
                plt.scatter(x,y,color='r')
            else:plt.scatter(x,y,color='b')
    plt.show()


train()
sample2d()
x=input()
