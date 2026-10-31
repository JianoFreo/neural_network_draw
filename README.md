# neural_network_draw

[![PyPI Version](https://img.shields.io/pypi/v/neural-network-draw)](https://pypi.org/project/neural-network-draw/)
[![Python](https://img.shields.io/pypi/pyversions/neural-network-draw)](https://pypi.org/project/neural-network-draw/)
[![Downloads](https://img.shields.io/pypi/dm/neural-network-draw)](https://pypi.org/project/neural-network-draw/)
[![License](https://img.shields.io/pypi/l/neural-network-draw)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/JianoFreo/neural_network_draw)](https://github.com/JianoFreo/neural_network_draw/issues)
[![Last Commit](https://img.shields.io/github/last-commit/JianoFreo/neural_network_draw)](https://github.com/JianoFreo/neural_network_draw/commits)

A Python library for **visualizing neural network architectures** using matplotlib.

This project focuses on **rendering neural networks as diagrams**, not training or running models.

It supports both:

*  Low-level step-by-step network building (`NeuralNetwork`)
*  High-level one-call architecture visualization (`DrawNN`)


# to install

```bash
pip install neural_network_draw
```
or ( would install the latest version if you have the v.0.1.0 installed )
```bash
pip install --upgrade neural-network-draw
```

#  What this library does

This library converts a neural network definition into a **visual graph representation**, including:

* Layers and neurons
* Connections between neurons
* Optional weights visualization
* Bias values
* Activation labels
* Input/output annotations
* Clean, publication-ready diagrams

---

#  Architecture Overview

The project has two main APIs:

## 1.  `NeuralNetwork` (Low-level API)

A step-by-step builder for full control over each layer.

### Example:

```python
network = NeuralNetwork(4)

network.add_layer(3, labels=["pixel_1", "pixel_2", "pixel_3"])

network.add_layer(
    4,
    weights=[
        [0.10, -0.20, 0.30, 0.40],
        [0.50, 0.60, -0.70, 0.80],
        [-0.10, 0.20, 0.30, -0.40],
    ],
    biases=[0.10, 0.20, 0.30, 0.40],
    activation="ReLU",
)

network.add_layer(
    2,
    weights=[
        [0.20, -0.30],
        [0.10, 0.40],
        [0.50, -0.60],
        [0.20, 0.10],
    ],
    biases=[0.05, -0.05],
    activation="Softmax",
    labels=["Cat", "Dog"],
)

network.draw(show_weights=True)
```
<img width="596" height="720" alt="image" src="https://github.com/user-attachments/assets/6949d89f-0eeb-459a-8b13-1229b35f3b4e" />

---

## 2.  `DrawNN` (High-level API)

<img width="676" height="692" alt="image-1 (1)" src="https://github.com/user-attachments/assets/88296c93-112b-49a8-b753-f2281012f009" />

## 3. Labeled structure

<img width="650" height="757" alt="image-2" src="https://github.com/user-attachments/assets/1d957bb4-6097-457a-a87a-2c5cbe5d50a4" />

A simplified interface for quickly defining architectures.

### Example:

```python
dnn = DrawNN(
    [3, 4, 2],
    weights=[...],
    biases=[None, [...], [...]],
    activations=[None, "ReLU", "Softmax"],
    input_labels=["x1", "x2", "x3"],
    output_labels=["Cat", "Dog"],
    title="High-Level API Example",
)

dnn.draw(show_weights=True)
```

---

#  Visualization Features

##  Network structure rendering

* Fully automatic layer spacing
* Centered neuron layout

##  Weights display

* Optional numeric display
* Color/thickness encodes magnitude & sign

##  Bias visualization

* Shown per neuron (optional)
* Supports partial or full specification

##  Activation labels

* ReLU
* Softmax
* Sigmoid
* Custom strings supported

##  Input / output labeling

* Human-readable feature names
* Class labels or regression outputs

##  Connection highlighting

* Highlight signal paths through specific neurons
* Visualize forward propagation routes
* Identify computational paths of interest
* Use `highlight_path` parameter to specify neurons to highlight per layer

### Example:

```python
# Highlight a specific path through the network
# This highlights neuron 0 in layer 0 → neurons 1,2 in layer 1 → neuron 0 in layer 2
network = NeuralNetwork(4)
network.add_layer(3, labels=["pixel_1", "pixel_2", "pixel_3"])
network.add_layer(4, activation="ReLU")
network.add_layer(2, activation="Softmax", labels=["Cat", "Dog"])

# Set the path to highlight
network.set_highlight_path([[0], [1, 2], [0]])
network.draw(show_weights=True)
```

# Or use DrawNN with highlight_path:

```python
dnn = DrawNN(
    [3, 4, 2],
    weights=[...],
    biases=[None, [...], [...]],
    activations=[None, "ReLU", "Softmax"],
    input_labels=["x1", "x2", "x3"],
    output_labels=["Cat", "Dog"],
    highlight_path=[[0], [1, 2], [0]],  # Highlight specific neuron path
)
dnn.draw(show_weights=True)
```

Highlighted connections are displayed in bright green with thicker lines to stand out from regular connections.

---
## Neuron Computation (Forward Propagation)

A neuron computes a weighted sum of its inputs, adds a bias, and applies an activation function.

### 1. Linear Transformation

$$
z_j^{(l)} = \sum_{i=1}^{n_{l-1}} w_{ij}^{(l)} a_i^{(l-1)} + b_j^{(l)}
$$

**Where:**

- $w_{ij}^{(l)}$ = Weight connecting neuron $i$ in the previous layer to neuron $j$ in the current layer.
- $a_i^{(l-1)}$ = Activation output from neuron $i$ in the previous layer.
- $b_j^{(l)}$ = Bias of neuron $j$.
- $z_j^{(l)}$ = Pre-activation value.
- $l$ = Current layer.
- $i$ = Neuron in the previous layer.
- $j$ = Neuron in the current layer.

### 2. Activation Functions

Activation functions determine how a neuron transforms its input.

**ReLU (Rectified Linear Unit)**

$$
f(z) = \max(0, z)
$$

Returns $z$ if it is positive; otherwise, returns $0$.

**Sigmoid**

$$
f(z) = \frac{1}{1 + e^{-z}}
$$

Maps a value to a range between $0$ and $1$.

**Tanh (Hyperbolic Tangent)**

$$
f(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

Maps a value to a range between $-1$ and $1$.

**Softmax**

For an output layer containing $n$ neurons:

$$
f(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{n} e^{z_k}}
$$

Converts output scores into probabilities that sum to $1$.

### 3. Layer-wise Computation

For an entire neural network layer, the computation uses a weight matrix, an activation vector, and a bias vector.

**Linear transformation:**

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

**Activation:**

$$
a^{(l)} = f(z^{(l)})
$$

Where:

- $W^{(l)}$ = Weight matrix of the current layer.
- $a^{(l-1)}$ = Activation vector from the previous layer.
- $b^{(l)}$ = Bias vector of the current layer.
- $z^{(l)}$ = Pre-activation vector.
- $f$ = Activation function applied to the layer.

---

## Dependencies

- Python 3.8+
- matplotlib
- NumPy (if used internally in the layout logic)


