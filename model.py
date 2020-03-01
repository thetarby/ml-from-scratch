import functions as f
import numpy as np


class Model:
    def __init__(self, input_layer=1 ,hidden_layer_neuron_number=2000, output_layer=1,hidden_activation=f.relu, output_activation=f.sigmoid):
        self.w_inp_hidden=np.random.rand(hidden_layer_neuron_number,input_layer)*2-1
        self.w_inp_hidden_g=[]
        self.b_hidden=np.random.rand(1,hidden_layer_neuron_number)*2-1
        self.b_hidden_g=[]

        self.hidden_layer_in=[]
        self.hidden_layer_out=[]

        self.w_hidden_out=np.random.rand(output_layer,hidden_layer_neuron_number)*2-1
        self.w_hidden_out_g=[]

        self.output_in=None


        self.hidden_activation=hidden_activation
        self.output_activation=output_activation

    def forward(self,input):
        #convert input to numpy array
        self.model_input=np.array(input)

        #pass input to first hidden layer
        result=np.matmul(self.model_input, np.transpose(self.w_inp_hidden))

        #save input values for each neuron in hidden layer to use later on calculating grads
        self.hidden_layer_in=result+self.b_hidden
        
        #add bias and apply activation function
        result=self.hidden_activation(self.hidden_layer_in)

        #save output values for each neuron in hidden layer to use later on calculating grads
        self.hidden_layer_out=result

        #pass result to output layer
        result=np.matmul(result,np.transpose(self.w_hidden_out))

        #save input values for output neuron to use later on calculating grads
        self.output_in=result

        #return result
        self.output=self.output_activation(result) if self.output_activation else result
        return self.output


    def backward(self,actual):
        #calculater error with mse wrt to actual value
        e=f.mse(self.output,actual)
        print("error : {}".format(e))

        #derivative of out layer
        d_out=f.d_mse(self.output,actual)*f.d_sigmoid(self.output_in)

        #derivative of hidden-out weights
        self.w_hidden_out_g=np.matmul(d_out,self.hidden_layer_out)

        #derivative of hidden layer biases
        self.b_hidden_g=f.d_relu(self.hidden_layer_in)*self.w_hidden_out*d_out
        
        #derivative of input-hidden weights
        res=[]
        for dim in self.model_input[0]:
            res.append((dim*self.b_hidden_g)[0])

        self.w_inp_hidden_g=np.transpose(np.array(res))


    def update(self):
        self.w_inp_hidden=self.w_inp_hidden-np.array(self.w_inp_hidden_g)*0.01
        self.w_hidden_out=self.w_hidden_out-np.array(self.w_hidden_out_g)*0.0005
        self.b_hidden=self.b_hidden-np.array(self.b_hidden_g)*0.001