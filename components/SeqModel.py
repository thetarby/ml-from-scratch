import numpy as np
from layer import Layer
import functions as f

class SeqModel:
    def __init__(self,input_dim, output_dim):
        self.input_dim=input_dim
        self.output_dim=output_dim

        self.layers=[Layer(input_dim, 'input'),Layer(output_dim,'output')]
        self.weights=[]

    
    def add_layer(self,neuron_count,activation_function):
        new_layer=Layer(neuron_count,'hidden', activation_function)
        self.layers.insert(-1,new_layer)

        #create weight matrix between each layer
        self.weights=[]
        for i,layer in enumerate(self.layers):
            #if it is last layer break loop
            if(i==len(self.layers)-1): 
                break

            next_layer=self.layers[i+1]
            self.weights.append(np.random.rand(next_layer.neuron_count,layer.neuron_count)*2-1)
        
        #return self to make chaining snytax available
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

        #calculate error with mse wrt to actual value
        e=f.d_mse(self.prediction,actual)
        chain=e

        #traverse layer list in reversed order
        for i,layer in reversed(list(enumerate(self.layers))):
            prev_layer=self.layers[i-1]
            weights_between=self.weights[i-1]
            chain=chain*layer.derivatives
            #this paremeter is for the gradients
            weights_between_g=[]
            for neuron in chain.flatten():
                weights_between_g.append(neuron*prev_layer.layer_out.flatten())
            
            #add to the gradient list
            self.weights_gradient.append(np.array(weights_between_g))
        
        #reverse it since it is traverse backward
        self.weights_gradient.reverse()
        print(self.weights_gradient)
    #method to update variables
    def update(self):
        for i in range(len(self.weights)):
            self.weights[i]-self.weights_gradient[i]*0.001
    