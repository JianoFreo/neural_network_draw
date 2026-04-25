# Install via pip

```bash
pip install neural-network-draw
```
---
Basic Usage
To use the library, they simply need to import the DrawNN class and provide a list representing the number of neurons in each layer.

```python
from neural_network_draw import DrawNN
```
### Example 1 : input neurons, three hidden layers (8, 10, 10), and 2 output neurons
```python

my_network = [4, 8, 10 ,10, 2]
# first index = input
# last index = output
# middles = hidden layers

# Initialize the drawer and render the diagram
viewer = DrawNN(my_network)
viewer.draw()
```
<img width="475" height="366" alt="image" src="https://github.com/user-attachments/assets/176da97c-db6a-4cfd-8ce9-9f0340c4989c" />

### Example 2 : Lower-level API if you want to build the layers manually
```python
network = NeuralNetwork(12)
network.add_layer(11)
network.add_layer(8)
network.add_layer(6)
network.add_layer(10)
network.draw()
```
<img width="571" height="382" alt="image" src="https://github.com/user-attachments/assets/500af726-8cab-4940-8727-6f1696238a7e" />

