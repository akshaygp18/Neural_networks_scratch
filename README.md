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
import numpy as np
```
### Importing the nnfs package and datasets
```
import nnfs
from nnfs.datasets import spiral_data
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

### Created the activation_softmax class
```
def forward(self,inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
```

