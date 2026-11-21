<div align="center">
  <img
    src="https://github.com/user-attachments/assets/201e73b0-f0df-4196-bb2d-2a87fb32ece6"
    width="300"
    alt="Neural Network Draw"
  />

  <br><br>

  <p>
    <a href="https://pypi.org/project/neural-network-draw/">
      <img src="https://img.shields.io/pypi/v/neural-network-draw" alt="PyPI Version">
    </a>
    <a href="https://pypi.org/project/neural-network-draw/">
      <img src="https://img.shields.io/pypi/pyversions/neural-network-draw" alt="Python">
    </a>
    <a href="https://pypi.org/project/neural-network-draw/">
      <img src="https://img.shields.io/pypi/dm/neural-network-draw" alt="Downloads">
    </a>
    <a href="LICENSE">
      <img src="https://img.shields.io/pypi/l/neural-network-draw" alt="License">
    </a>
  </p>
</div>

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

<h2 align="center">Stats</h2>

<div align="center">
  <img
    src="https://github.com/user-attachments/assets/ea3b59a8-537d-4d41-ba03-860442636b7e"
    width="90%"
    alt="Stats 1"
  />
  <img
    src="https://github.com/user-attachments/assets/192412b6-aeec-4e62-bb42-66190f677d51"
    width="90%"
    alt="Stats 2"
  />
  <img
    src="https://github.com/user-attachments/assets/feeac44e-7a52-42f3-a967-18d37f7fdc78"
    width="90%"
    alt="Stats 3"
  />
</div>

---

<p align="center">
  If you find <strong>Neural Network Draw</strong> useful, consider supporting its development. Every coffee helps keep the project growing!
</p>

<p align="center">
  <a href="https://buymeacoffee.com/jianofreo">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee" width="200" />
  </a>
</p>

<p align="center">
  Thank you for supporting open-source development! ❤️
</p>

