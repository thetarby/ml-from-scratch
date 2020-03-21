import numpy as np
from layer import *
import functions as f
class SeqModel:
    def __init__(self):
        self.layers=[]
        self.weights=[]

    def add_layer(self,layer):
        self.layers.append(layer)
        
        #if it is first layer do not add to weights
        if len(self.layers)==1: 
            return self
        #else add weights between layers
        prev_layer=self.layers[-2]
        self.weights.append(np.random.rand(layer.neuron_count,prev_layer.neuron_count)*2-1)
        return self


    #method to forward pass an input
    def forward(self,inp):
        #make sure input is an numpy array 
        res=np.array(inp)

        #pass input to input layer
        #it do not change anything only sets input and output for the input layer to avoid errors calculating grads
        res=self.layers[0].forward(res)
        """
            multiply with each weight matrix.
            Transpose is applied to make matrix multiplication valid. 
        """
        for i,layer in enumerate(self.layers):
            #if it is last layer break loop
            if(i==len(self.layers)-1): 
                break
            
            next_layer=self.layers[i+1]
            res=np.matmul(res,np.transpose(self.weights[i]))

            #forward them to layer which adds biases and applies activation function assigned to layer
            res=next_layer.forward(res)


        #save result to further use for calculating grads
        self.prediction=res

        return res


    #method to calculate gradients for each variable
    #act is the actual value
    def backward(self,actual):
        #remove gradients calculated from last backward
        self.weights_gradient=[]
        self.biases_gradient=[]

        #calculate error with mse wrt to actual value
        e=f.d_mse(self.prediction,actual)
        
        #TODO: make sure its shape is applicable
        chain=np.array(e)
        
        #derivative of output layer
        chain*=self.layers[-1].derivatives_wrt_input
        
        
        #traverse layer list in reversed order
        for i,layer in reversed(list(enumerate(self.layers))):
            
            if i==0: break
            
            prev_layer=self.layers[i-1]
            weights_between=self.weights[i-1]
            

            """
            every column in weights_between is a neuron. that vector is all the weights exiting this neuron.
            
            this column should have the same size as chain since chain is the gradients of inputs wrt to input of 
            layer(next layer after prev_layer)
            
            dot product this vector with chain and sum all. It is the gradient wrt input to prev_layer's first neuron 
            """
            #chain is a column vector now
            #every column in weights_between element-wise multiplied with chain vector
            temp=weights_between*np.transpose(chain)

            #gradient wrt to output of prev_layer
            # shape=(1,size of prev_layer) 
            #temp matrix is summed along its columns
            derivatives_wrt_output=np.sum(temp,0,keepdims=True)

            #derivative of output wrt to input for prev_layer
            #element wise multiplication
            #TODO: derivatives_wrt_input shape must be (1,x)
            new_chain=prev_layer.derivatives_wrt_input*derivatives_wrt_output

            #derivative of bias is same as new_chain since changing input is linearly dependent to changing bias
            prev_layer.biases_gradient=new_chain

            #this paremeter is for the gradients
            weights_between_g=[]
            for neuron in chain.flatten():
                weights_between_g.append(neuron*prev_layer.layer_out.flatten())
            
            #add to the gradient list
            self.weights_gradient.append(np.array(weights_between_g))
            
            #assign use new chain
            chain=new_chain
        
        #reverse it since it is traverse backward
        self.weights_gradient.reverse()
        #print(self.weights_gradient[0])
        self.biases_gradient.reverse()


    #method to update variables
    def update(self,lr):
        for i in range(len(self.weights)):
            self.weights[i]=self.weights[i]-self.weights_gradient[i]*lr
        for i in range(len(self.layers)):
            if self.layers[i].biases is not None:
                self.layers[i].biases=self.layers[i].biases-self.layers[i].biases_gradient*lr
    