import numpy as np
import pytarbs.functions as f

class BaseLayer:
    def __init__(self,neuron_count):
        self.neuron_count=neuron_count
        self.biases=np.zeros((1,neuron_count))
        self.biases_gradient=np.zeros((1,neuron_count))

    def forward(self):
        pass

    def backward(self):
        pass

class InputLayer(BaseLayer):
    def __init__(self,neuron_count):
        super().__init__(neuron_count)

    def forward(self,X):
        self.layer_in=X
        self.layer_out=X
        return self.layer_out

    def backward(self,chain):
        return 1
    #method to update variables
    def update(self,lr):
        pass

class OutputLayer(BaseLayer):
    def __init__(self, in_features, out_features, activation_function=None):
        self.in_features=in_features
        self.out_features=out_features

        # an array of shape (1, neuron_count)
        self.layer_in=None
        self.act_func_in=None

        # an array of shape (1, neuron_count)
        self.layer_out=None     

        self.weights=np.random.rand(out_features, in_features)*2-1
        self.activation_function=activation_function if activation_function is not None else f.Dummy()

    
    def forward(self,X):
        self.layer_in=X

        #if no activation function then inp=out
        self.layer_out=X
        
        #forward input to weights
        self.act_func_in=np.matmul(X,np.transpose(self.weights))

        #if there is bias apply it and then apply activation function
        if self.activation_function is not None:
                self.layer_out=self.activation_function.call(self.act_func_in)
        
        return self.layer_out


    def backward(self,chain):
        new_chain=self.activation_function.d(self.act_func_in)*chain

        new_chain=self.weights*np.transpose(new_chain)

        #gradient wrt to output of prev_layer
        # shape=(1,size of prev_layer) 
        #temp matrix is summed along its columns
        new_chain=np.sum(new_chain,0,keepdims=True)

        #derivative of output wrt to input for prev_layer
        #element wise multiplication
        #TODO: derivatives_wrt_input shape must be (1,x)
        new_chain=new_chain

        #this paremeter is for the gradients
        self.weights_gradient=[]
        for neuron in (self.activation_function.d(self.act_func_in)*chain).flatten():
            self.weights_gradient.append(neuron*self.layer_in.flatten())
        self.weights_gradient=np.array(self.weights_gradient)
        return new_chain
    #method to update variables
    def update(self,lr):
        self.weights=self.weights-self.weights_gradient*lr

class DenseLayer(BaseLayer):
    def __init__(self, in_features, out_features, activation_function=None):
        self.in_features=in_features
        self.out_features=out_features

        # an array of shape (1, neuron_count)
        self.layer_in=None
        self.act_func_in=None

        # an array of shape (1, neuron_count)
        self.layer_out=None
        
        #if it is a hidden layer add bias variables in the shape(1,x) which is basically one row matrix
        self.biases=np.zeros((1, out_features))

        self.weights=np.random.rand(out_features, in_features)*2-1
        self.activation_function=activation_function if activation_function is not None else f.Dummy()

    
    def forward(self,X):
        self.layer_in=X

        #if no activation function then inp=out
        self.layer_out=X
        
        #forward input to weights
        self.act_func_in=np.matmul(X,np.transpose(self.weights))

        #if there is bias apply it and then apply activation function
        if self.activation_function is not None:
            if self.biases is not None:
                self.act_func_in= self.act_func_in + self.biases
                self.layer_out=self.activation_function.call(self.act_func_in)
            else:
                self.layer_out=self.activation_function.call(self.act_func_in)
        
        return self.layer_out


    def backward(self,chain):
        new_chain=self.activation_function.d(self.act_func_in)*chain
        
        #derivative of bias is same as new_chain since changing input is linearly dependent to changing bias
        self.biases_gradient=new_chain

        new_chain=self.weights*np.transpose(new_chain)

        #gradient wrt to output of prev_layer
        # shape=(1,size of prev_layer) 
        #temp matrix is summed along its columns
        new_chain=np.sum(new_chain,0,keepdims=True)

        #derivative of output wrt to input for prev_layer
        #element wise multiplication
        #TODO: derivatives_wrt_input shape must be (1,x)
        new_chain=new_chain

        #this paremeter is for the gradients
        self.weights_gradient=[]
        for neuron in (self.activation_function.d(self.act_func_in)*chain).flatten():
            self.weights_gradient.append(neuron*self.layer_in.flatten())
        self.weights_gradient=np.array(self.weights_gradient)
        return new_chain    
    #method to update variables
    def update(self,lr):
        self.weights=self.weights-self.weights_gradient*lr
        self.biases=self.biases-self.biases_gradient*lr
