1. Installation
The most common way to install is via pip:

```bash
pip install neural-network-draw
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