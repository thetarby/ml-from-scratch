import numpy as np
import functions as f

class Layer:
    def __init__(self):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
    
    def calculate_derivatives(self):
        pass


class DenseLayer(Layer):
    def __init__(self, neuron_count, layer_type,activation_function=None):
        self.neuron_count=neuron_count
        self.layer_type=layer_type
        # an array of shape (1, neuron_count)
        self.layer_in=None
        
        # an array of shape (1, neuron_count)
        self.layer_out=None
        
        #if it is a hidden layer add bias variables in the shape(1,x) which is basically one row matrix
        self.biases=np.zeros((1,neuron_count)) if layer_type=='hidden' else None

        self.activation_function=activation_function

    
    def forward(self,X):
        self.layer_in=X

        #if no activation function then inp=out
        self.layer_out=X
        
        #if there is bias apply it and then apply activation function
        if self.activation_function is not None:
            if self. biases is not None:
                self.layer_in=self.layer_in + self.biases
                self.layer_out=self.activation_function.call(self.layer_in)
            else:
                self.layer_out=self.activation_function.call(self.layer_in)
        
        self.calculate_derivatives()
        return self.layer_out


    def calculate_derivatives(self):
        self.derivatives_wrt_input=self.activation_function.d(self.layer_in) if (self.activation_function is not None) else np.ones((1,self.neuron_count))
