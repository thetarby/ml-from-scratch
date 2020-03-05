import functions as f
from SeqModel import SeqModel

x=SeqModel(1,1).add_layer(8,f.Relu())
y=x.forward([[2]])
print(y)

x.backward(3)
x.update()
y=x.forward([[2]])
x.backward(4)
x.update()