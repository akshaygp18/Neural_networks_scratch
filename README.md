# Neural_network_from_scratch
Neural_network_from_scratch

## Commands using -

### To create any empty file using git bash -
```
touch filename
```

### To create an environment
```
conda create -p ./env python=3.7 -y
```

### To activate the environment
```
conda activate ./env
```

### Importing the numpy package
```
Import numpy as np
```

### Initializing random numbers for the weights
```
np.random.seed(0)
```

### Initializing inputs 
```
x = [[1,2,1.6,2.5],
     [2,5,-2.5,-1.5],
     [-1.9,1.7,-2.2,1.0]]
```

### Created the dense_layer class
```
class dense_layer:
    def __init__(self,n_inputs,n_neurons):
        self.weights = np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases
```

### Created the activation_relu class
```
class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)
```

