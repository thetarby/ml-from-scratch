import numpy as np
import functions as f
import matplotlib.pyplot as plt

layers=[]
"""
input 1 neuron
hidden_layer1 4 neuron
hidden_layer2 4 neuron
output layer 1 neuron
"""

#model:
model_input=3
w_inp_hidden1=np.random.rand(8,1)
w_inp_hidden1_g=[]
b_hidden1=np.random.rand(1,8)-1
b_hidden1_g=[]

hidden_layer1_in=[]
hidden_layer1=[]


w_hidden1_out=np.random.rand(1,8)
w_hidden1_out_g=[]

output=[]



def forward_pass(inp):
    global model_input
    model_input=np.array([[inp]])
    res=np.matmul(np.array([[inp]]),np.transpose(w_inp_hidden1))
    global hidden_layer1_in
    hidden_layer1_in=res+b_hidden1
    res=f.relu(res+b_hidden1)
    
    global hidden_layer1
    hidden_layer1=res

    res=np.matmul(hidden_layer1,np.transpose(w_hidden1_out))
    return res

def back_pass(pred,act):
    e=f.mse(pred,act)
    print("error : {}".format(e))
    #derivative of out layer
    x=f.d_mse(pred,act)
    y=f.d_mse(pred,act)

    #derivative of hidden2-out weights
    x=np.matmul(x,hidden_layer1)
    global w_hidden1_out_g
    w_hidden1_out_g=x
    

    x=f.d_relu(hidden_layer1_in)*w_hidden1_out*y
    global b_hidden1_g
    b_hidden1_g=x

    x=np.matmul(model_input,x)
    global w_inp_hidden1_g
    w_inp_hidden1_g=np.transpose(x)

def update():
    global w_inp_hidden1,w_hidden1_out,b_hidden1
    w_inp_hidden1=w_inp_hidden1-np.array(w_inp_hidden1_g)*0.05
    w_hidden1_out=w_hidden1_out-w_hidden1_out_g*0.05
    b_hidden1=b_hidden1-b_hidden1_g*0.05


# Create the vectors X and Y
xx = np.array(range(0,100))/100
yy = xx ** 3

# Create the plot
plt.plot(xx,yy)
# Show the plot
plt.show()

def train():
    print("-----------------------------beginning weights------------------")


    for i in range(1000):
        sample()
        n=np.random.rand()
        pred=forward_pass(n)

        act=n**3 if n>0 else 0 #2*n if n>0 else 0
        print("-----------------------pass : {} ----------------".format(n))
        print("prediction : "+str(pred), "act : "+str(act))
        back_pass(pred,act)
        update()
        print(b_hidden1)
    """
        print("-----------------------------grads------------------")
        print(w_inp_hidden1_g)
        print(w_hidden1_out_g)

        print("-----------------------------updated------------------")
        print(w_inp_hidden1)
        print(w_hidden1_out)"""

def sample():
    x=[]
    y=[]
    for i in range(100):
        n=1/100*i
        pred=forward_pass(n)
        x.append(n)
        y.append(pred[0][0])
    plt.plot(xx,yy)
    plt.plot(np.array(x),np.array(y))
    plt.pause(0.001)
    plt.draw()
    plt.clf()
train()
