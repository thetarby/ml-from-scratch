import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
import math 
import numpy as np
from pytarbs import functions as f
from pytarbs.SeqModel import SeqModel
from pytarbs.layer import *
#42 is the meaning of the life
np.random.seed(42)

model=SeqModel().add_layer(InputLayer(2)).add_layer(DenseLayer(16,f.Relu())).add_layer(OutputLayer(1, f.Sigmoid()))

#function to be approximated. this will be the second graph 
#after training finishes. First graph will be prediction of the model
func= lambda x,y:((x>0 and y>0) or (y<0 and x<0))
def train():
    print("-----------------------------beginning weights------------------")


    for i in range(10000):
        #sample(1)
        x,y=np.random.rand(2)*2-1
        print(x,y)
        pred=model.forward([[x,y]])

        act= func(x,y)
        print("-----------------------pass : {} ----------------".format((x,y)))
        print("prediction : "+str(pred), "act : "+str(act))
        model.backward(act)

        model.update(lr=0.03)
    """
        print("-----------------------------grads------------------")
        print(w_inp_hidden1_g)
        print(w_hidden1_out_g)

        print("-----------------------------updated------------------")
        print(w_inp_hidden1)
        print(w_hidden1_out)"""




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


#sample2d()

train()
#sample(0)
sample2d()
x=input()
