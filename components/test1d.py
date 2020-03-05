import matplotlib.pyplot as plt
import math 
import numpy as np
import functions as f
from SeqModel import SeqModel

model=SeqModel(1,1).add_layer(8,f.Relu())
def train():
    print("-----------------------------beginning weights------------------")


    for i in range(2):
        #sample(1)
        x,y=np.random.rand(2)
        print(x,y)
        pred=model.forward([[x]])

        act=np.sin(x*15)/2+1+x**2 #2*n if n>0 else 0
        print("-----------------------pass : {} ----------------".format((x,y)))
        print("prediction : "+str(pred), "act : "+str(act))
        model.backward(act)
        model.update()
    """
        print("-----------------------------grads------------------")
        print(w_inp_hidden1_g)
        print(w_hidden1_out_g)

        print("-----------------------------updated------------------")
        print(w_inp_hidden1)
        print(w_hidden1_out)"""




def sample2d():
    f = plt.figure(1)
    plt.plot()
    for i in range(50):
        for j in range(50):
            x=1/50*i
            y=1/50*j
            if model.forward([[x,y]])>0.5:
                plt.scatter(x,y,color='r')
            else:plt.scatter(x,y,color='b')
    plt.show()

    plt.plot()
    for i in range(50):
        for j in range(50):
            x=1/50*i
            y=1/50*j
            if ((x>0.5 and y>0.5) or (y<0.5 and x<0.5)) :
                plt.scatter(x,y,color='r')
            else:plt.scatter(x,y,color='b')
    plt.show()
def sample(x):
    x=[]
    y=[]
    xx = np.array(range(0,100))/100
    yy = np.sin(xx*15)/2+1+xx**2 # 5*xx**2 - 15*(xx-0.3)**3 + xx/3  
    for i in range(100):
        n=1/100*i
        pred=model.forward([[n]])
        x.append(n)
        y.append(pred[0][0])
    plt.plot(xx,yy)
    plt.plot(np.array(x),np.array(y))
    plt.pause(0.001)
    plt.draw()
    if(x): plt.clf()
    else: 
        input()


#sample2d()
#sample(0)

train()
sample(0)
#sample2d()
x=input()