import numpy as np
import pytarbs.functions as f

class FeedForward:
    def __init__(self):
        self.layers=[]
        self.weights=[]

    def add_layer(self,layer):
        self.layers.append(layer)
        return self 

    #method to forward pass an input
    def forward(self,inp):
        #make sure input is an numpy array 
        res=np.array(inp)

        #pass input to each layer
        for layer in self.layers:
            res=layer.forward(res)
        
        self.prediction=res

        return res


    #method to calculate gradients for each variable
    #act is the actual value
    def backward(self,loss):
        #calculate error with mse wrt to actual value
        #e=f.d_mse(self.prediction,actual)
        e=loss

        #TODO: make sure its shape is applicable
        chain=np.array(e)

        #if loss function's gradient values and output layer's size do not match give error 
        assert chain.size == self.layers[-1].out_features, 'Loss size do not match out layer size'

        #traverse layer list in reversed order
        for i,layer in reversed(list(enumerate(self.layers))):
            chain=layer.backward(chain)


    #method to update variables
    def update(self,lr):
        for layer in self.layers:
            layer.update(lr)