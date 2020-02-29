import numpy as np
import functions as f
import matplotlib.pyplot as plt
import math 

layers=[]
"""
input 1 neuron
hidden_layer1 4 neuron
hidden_layer2 4 neuron
output layer 1 neuron
"""

#model:
model_input=3
w_inp_hidden1=np.random.rand(2000,2)*2-1
w_inp_hidden1_g=[]
b_hidden1=np.random.rand(1,2000)*2-1
b_hidden1_g=[]

hidden_layer1_in=[]
hidden_layer1=[]


w_hidden1_out=np.random.rand(1,2000)*2-1
w_hidden1_out_g=[]

output_in=[]



def forward_pass(inp):
    global model_input
    model_input=np.array(inp)
    res=np.matmul(model_input,np.transpose(w_inp_hidden1))
    global hidden_layer1_in
    hidden_layer1_in=res+b_hidden1
    res=f.relu(res+b_hidden1)
    
    global hidden_layer1
    hidden_layer1=res

    res=np.matmul(hidden_layer1,np.transpose(w_hidden1_out))
    
    global output_in
    output_in=res

    return f.sigmoid(res)

def back_pass(pred,act):
    e=f.mse(pred,act)
    print("error : {}".format(e))
    #derivative of out layer
    x=f.d_mse(pred,act)*f.d_sigmoid(output_in)
    y=f.d_mse(pred,act)*f.d_sigmoid(output_in)

    #derivative of hidden2-out weights
    x=np.matmul(x,hidden_layer1)
    global w_hidden1_out_g
    w_hidden1_out_g=x
    

    x=f.d_relu(hidden_layer1_in)*w_hidden1_out*y
    global b_hidden1_g
    b_hidden1_g=x

    res=[]
    for dim in model_input[0]:
        res.append((dim*x)[0])

    x=np.array(res)
    global w_inp_hidden1_g
    w_inp_hidden1_g=np.transpose(x)

def update():
    global w_inp_hidden1,w_hidden1_out,b_hidden1
    w_inp_hidden1=w_inp_hidden1-np.array(w_inp_hidden1_g)*0.01
    w_hidden1_out=w_hidden1_out-np.array(w_hidden1_out_g)*0.0005
    b_hidden1=b_hidden1-np.array(b_hidden1_g)*0.001


# Create the vectors X and Y
xx = np.array(range(0,100))/100
yy = np.sin(xx*15)/2+1+xx**2 # 5*xx**2 - 15*(xx-0.3)**3 + xx/3

# Create the plot
plt.plot(xx,yy)
# Show the plot
plt.show()

def train():
    print("-----------------------------beginning weights------------------")


    for i in range(1000):
        #sample(1)
        x,y=np.random.rand(2)
        print(x,y)
        pred=forward_pass([[x,y]])

        act=int( 0.15>((x-0.5)**2+(y-0.5)**2)) #2*n if n>0 else 0
        print("-----------------------pass : {} ----------------".format((x,y)))
        print("prediction : "+str(pred), "act : "+str(act))
        back_pass(pred,act)
        update()
    """
        print("-----------------------------grads------------------")
        print(w_inp_hidden1_g)
        print(w_hidden1_out_g)

        print("-----------------------------updated------------------")
        print(w_inp_hidden1)
        print(w_hidden1_out)"""

def sample(x):
    x=[]
    y=[]
    for i in range(100):
        n=1/100*i
        pred=forward_pass([[n]])
        x.append(n)
        y.append(pred[0][0])
    plt.plot(xx,yy)
    plt.plot(np.array(x),np.array(y))
    plt.pause(0.001)
    plt.draw()
    if(x): plt.clf()
    else: 
        input()


def sample2d():
    f = plt.figure(1)
    plt.plot()
    for i in range(50):
        for j in range(50):
            x=1/50*i
            y=1/50*j
            if forward_pass([[x,y]])>0.5:
                plt.scatter(x,y,color='r')
            else:plt.scatter(x,y,color='b')
    plt.show()

    plt.plot()
    for i in range(50):
        for j in range(50):
            x=1/50*i
            y=1/50*j
            if (0.15>((x-0.5)**2+(y-0.5)**2) ) :
                plt.scatter(x,y,color='r')
            else:plt.scatter(x,y,color='b')
    plt.show()

#sample2d()

train()
#sample(0)
sample2d()
x=input()
