"""
these tests are for feed forward module.
feed forward module is pretty much the same as other models
but it moves gradient calculation from model class to layer class.

Since There is not any autograd package implemented in pytarbs separating gradient calculation
from model class makes more sense when there is a need for implementing new type of layers

feed-forward module is hence the same thing but a little bit differently organized. 
All the codes are the same but in different places

models in feed-forward module should be used with layers from feed-forward module
"""
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

model=FeedForward().add_layer(InputLayer(1)).add_layer(DenseLayer(1,8,f.LeakyRelu())).add_layer(DenseLayer(8,8,f.LeakyRelu())).add_layer(OutputLayer(8,1, None))

#function to be approximated. this will be plotted as blue in below plot
func= lambda x:np.sin(x*5)/2+1+x**2

def train():
    print("-----------------------------beginning------------------")


    for i in range(10000):
        #sample(1)
        x,y=np.random.rand(2)*2-1
        print(x,y)

        #predicted
        pred=model.forward([[x]])

        #actual value of the func
        act=func(x)
        print("-----------------------pass : {} ----------------".format((x,y)))
        print("prediction : "+str(pred), "act : "+str(act))
        model.backward(act)

        model.update(lr=0.01)
    """
        print("-----------------------------grads------------------")
        print(w_inp_hidden1_g)
        print(w_hidden1_out_g)

        print("-----------------------------updated------------------")
        print(w_inp_hidden1)
        print(w_hidden1_out)"""

#funtion to draw plot
def sample(x):
    x=[]
    y=[]
    xx = np.array(range(-50,50))/50
    yy = func(xx)
    for i in range(-50,50):
        n=1/50*i
        pred=model.forward([[n]])
        x.append(n)
        y.append(pred[0][0])
    plt.plot(xx,yy)
    plt.plot(np.array(x),np.array(y))
    plt.pause(0.001)
    plt.show()
    if(x): plt.clf()
    else: 
        input()


#sample2d()
#sample(0)

train()
sample(0)
#sample2d()
x=input()
