from model import Model
import matplotlib.pyplot as plt
import math 
import numpy as np
import functions as f

model=Model(input_layer=2,hidden_layer_neuron_number=2000,output_layer=1, hidden_activation=f.relu, output_activation=f.sigmoid)
def train():
    print("-----------------------------beginning weights------------------")


    for i in range(10000):
        #sample(1)
        x,y=np.random.rand(2)
        print(x,y)
        pred=model.forward([[x,y]])

        act=int( (x>0.5 and y>0.5) or (y<0.5 and x<0.5) ) #2*n if n>0 else 0
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


#sample2d()

train()
#sample(0)
sample2d()
x=input()



