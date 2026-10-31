# neural_network_draw

[![PyPI Version](https://img.shields.io/pypi/v/neural-network-draw)](https://pypi.org/project/neural-network-draw/)
[![Python](https://img.shields.io/pypi/pyversions/neural-network-draw)](https://pypi.org/project/neural-network-draw/)
[![Downloads](https://img.shields.io/pypi/dm/neural-network-draw)](https://pypi.org/project/neural-network-draw/)
[![License](https://img.shields.io/pypi/l/neural-network-draw)](LICENSE)

A Python library for **visualizing neural network architectures** using matplotlib.

This project focuses on **rendering neural networks as diagrams**, not training or running models.

It supports three ways of creating a diagram:

*  Low-level step-by-step network building (`NeuralNetwork`)
*  High-level one-call architecture visualization (`DrawNN`)
*  **New in 0.3.0:** drawing a PyTorch model directly (`DrawNN(model)`)


---

# To install

```bash
pip install neural-network-draw
```

PyTorch support is optional. To install it together with the library:

```bash
pip install "neural-network-draw[torch]"
```

To update an existing installation to the latest version:

```bash
pip install --upgrade neural-network-draw
```

---

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

The project offers three ways to create a diagram. They all share the same rendering engine:

| Way | Use it when | Input |
|---|---|---|
| `NeuralNetwork` (low-level) | you want full control over each layer | layers added one by one |
| `DrawNN` (high-level) | you want a quick diagram from a list of layer sizes | a list such as `[3, 4, 2]` plus options |
| `DrawNN(model)` (PyTorch) | you already have a PyTorch model | a `torch.nn.Module` |

## 1.  `NeuralNetwork` (Low-level API)

A step-by-step builder for full control over each layer.
Weights are given as `[previous_layer_size][this_layer_size]`.

### Example:

```python
from neural_network_draw import NeuralNetwork

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

### Labeled structure

<img width="650" height="757" alt="image-2" src="https://github.com/user-attachments/assets/1d957bb4-6097-457a-a87a-2c5cbe5d50a4" />

A simplified interface for quickly defining architectures.

### Example:

```python
from neural_network_draw import DrawNN

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

## 3.  PyTorch models (just pass the model)

No need to build lists of sizes, weights or biases by hand. Pass a `torch.nn.Module`
and everything is read from it automatically: layer sizes, **trained weights**, biases
and activation names.

```python
import torch.nn as nn
from neural_network_draw import DrawNN, draw_model

model = nn.Sequential(
    nn.Linear(3, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Softmax(dim=1),
)

DrawNN(model).draw(show_weights=True)   # or the one-liner:
draw_model(model, show_weights=True)
```

Custom `nn.Module` classes work too, and anything you pass explicitly is added on top
of (or overrides) what was read from the model:

```python
DrawNN(
    model,
    input_labels=["x1", "x2", "x3"],
    output_labels=["Cat", "Dog"],
    highlight_path=[[0], [1, 2], [0]],
    title="My model",
).draw(show_weights=True)
```

Also available: `DrawNN.from_pytorch(model)` and `NeuralNetwork.from_pytorch(model)`
(the latter returns a `NeuralNetwork` you can keep customizing).

### What is read from the model

* every `nn.Linear` becomes a layer (the first one also creates the input layer), with its real weights and biases
* activation modules (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `GELU`, ...) are shown under the `Linear` layer before them; several in a row are shown as `ReLU → Sigmoid`
* `Dropout*`, `Identity` and `Flatten` don't change the data at inference, so they are left out of the diagram
* nested `Sequential`s, `ModuleList`s and custom modules are fine as long as they are built from the above
* values are detached and copied to plain Python floats (CPU or CUDA, any dtype); the model's parameters, gradients and train/eval mode are never changed

### Unsupported layers

**Nothing is silently misdrawn.** Anything the diagram can't represent accurately (`Conv`, `BatchNorm`, `LayerNorm`, RNNs, `GLU`, custom operations, skip connections / branching, parameters owned directly by a container, ...) raises an `UnsupportedModelError` that names the offending modules. If you'd rather skip them, pass `strict=False` to get a warning and a diagram without those parts.

```python
DrawNN(model_with_batchnorm, strict=False).draw()   # warns, then omits the BatchNorm layers
```

### Layer order

By default layers are read in the order they are defined, which is the forward order for `nn.Sequential` and for most custom modules. If a custom module defines its layers in a different order than it uses them, pass `example_input` and the real execution order is traced with one eval-mode, no-gradient forward pass (hooks are removed and train/eval flags restored afterwards; layers that never run are left out):

```python
DrawNN(model, example_input=torch.zeros(1, 4)).draw()
```

Activations written as `F.relu(...)` inside `forward()` can't be detected; pass `activations=[None, "ReLU", ...]` for those.

### Tensors in the manual APIs

Tensors and NumPy arrays are also accepted anywhere plain lists were (`weights=`, `biases=`, `add_layer(...)`). Note the library's weight layout is `[previous_size][layer_size]`; PyTorch's `Linear.weight` is `[out][in]`, so use `weight.T` if you pass one by hand (`DrawNN(model)` does this for you).

PyTorch is optional: the original list-based API works without it installed.

### Limits

* only fully-connected networks (`nn.Linear`); no convolution, normalisation layers, RNNs or branching
* layers wider than 64 neurons give a warning, because every neuron and connection is drawn

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

### Or use DrawNN with highlight_path:

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

| # | Demo |
|---|------|
| 1-4 | Original examples: structure, labels, low-level and high-level full features |
| 5 | PyTorch `nn.Sequential` drawn directly |
| 6 | Custom `nn.Module` + `draw_model` one-liner |
| 7 | Trained weights: XOR network before vs after training |
| 8 | Labels and a highlighted path on a PyTorch model |
| 9 | Nested modules (blocks, `Sequential` in `Sequential`, `ModuleList`) |
| 10 | Layers defined out of order, fixed with `example_input` |
| 11 | Unsupported layers: `strict=True` error vs `strict=False` warning |
| 12 | Tensors (CPU/CUDA, `requires_grad`) in the manual APIs |
| 13 | Activation gallery (stacked activations, ignored Dropout/Identity) |
| 14 | A larger model |

The PyTorch examples are skipped automatically if torch isn't installed.

## Running the tests

```bash
pip install -e . pytest torch
pytest tests/
```

The PyTorch tests skip themselves if torch isn't installed, and the CUDA tests run only when a GPU is available.

---

## What's new in 0.3.0

* draw a `torch.nn.Module` directly: `DrawNN(model)`, `DrawNN.from_pytorch(model)`, `NeuralNetwork.from_pytorch(model)`, `draw_model(model)`
* `UnsupportedModelError` / `strict=False` for layers that can't be drawn accurately
* `example_input=` to trace the real layer execution order
* tensors and NumPy arrays accepted in `weights=`, `biases=` and `add_layer(...)`
* optional install: `pip install "neural-network-draw[torch]"`
* 10 new runnable demos

Existing code needs no changes.

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
- matplotlib (NumPy is installed with it)
- PyTorch (optional, only needed to draw `torch.nn.Module` models)