import numpy as np
np.random.seed(0)

x = [[1,2,1.6,2.5],
     [2,5,-2.5,-1.5],
     [-1.9,1.7,-2.2,1.0]]

class dense_layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases



class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)




layer1 = dense_layer(4,5)
activation1 = Activation_Relu()

layer1.forward(x)

print(layer1.biases)
print("-----------------------------")
print(layer1.weights)
print("-----------------------------")
print(layer1.output)
print("-----------------------------")
activation1.forward(layer1.output)
#print(activation1.biases)
#print(activation1.weights)
print(activation1.output)
print("-----------------------------")