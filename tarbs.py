import numpy as np
import functions as f

layers=[]
"""
input 1 neuron
hidden_layer 4 neuron
output layer 1 neuron
"""

#model:
model_input=3
w_inp_hidden=np.random.rand(4,1)
w_inp_hidden_g=[]

hidden_layer=[]

w_hidden_out=np.random.rand(1,4)
w_hidden_out_g=[]

output=[]



def forward_pass(inp):
    global model_input
    model_input=np.array([[inp]])
    res=np.matmul(np.array([[inp]]),np.transpose(w_inp_hidden))
    res=f.relu(res)
    global hidden_layer
    hidden_layer=res
    res=np.matmul(res,np.transpose(w_hidden_out))
    return res

def back_pass(pred,act):
    e=f.mse(pred,act)
    print("error : {}".format(e))
    x=f.d_mse(pred,act)
    x=np.matmul(x,hidden_layer)
    global w_hidden_out_g
    w_hidden_out_g=x
    x=f.d_relu(x)
    x=np.matmul(model_input,x)
    global w_inp_hidden_g
    w_inp_hidden_g=np.transpose(x)

def update():
    global w_inp_hidden,w_hidden_out
    w_inp_hidden=w_inp_hidden-w_inp_hidden_g*0.01
    w_hidden_out=w_hidden_out-w_hidden_out_g*0.01

print("-----------------------------beginning weights------------------")
print(w_inp_hidden)
print(w_hidden_out)

for i in range(1000):
    n=np.random.rand()*10-5
    pred=forward_pass(n)

    print(hidden_layer)
    act=2*n if n>0 else 0
    print("-----------------------pass : {} ----------------".format(n))
    print("prediction : "+str(pred))
    back_pass(pred,act)
    update()
"""
    print("-----------------------------grads------------------")
    print(w_inp_hidden_g)
    print(w_hidden_out_g)

    print("-----------------------------updated------------------")
    print(w_inp_hidden)
    print(w_hidden_out)"""