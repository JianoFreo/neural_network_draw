1. Installation
The most common way to install is via pip:

```bash
pip install neural-network-draw==0.1.0
```
---
2. Basic Usage
To use the library, they simply need to import the DrawNN class and provide a list representing the number of neurons in each layer.

```python
from neural_network_draw import DrawNN

# Example: 4 input neurons, two hidden layers (8 and 8), and 2 output neurons
my_network = [4, 8, 8, 2]

# Initialize the drawer and render the diagram
viewer = DrawNN(my_network)
viewer.draw()
```

3. More Examples

```python
from neural_network_draw import DrawNN

# Small network
DrawNN([2, 3, 1]).draw()

# Balanced network
DrawNN([4, 4, 4]).draw()

# Deeper network with multiple hidden layers
DrawNN([6, 8, 8, 4, 2]).draw()
```

```python
from neural_network_draw import NeuralNetwork

# Lower-level API if you want to build the layers manually
network = NeuralNetwork(8)
network.add_layer(3)
network.add_layer(8)
network.add_layer(2)
network.draw()
```