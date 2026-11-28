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


The project offers three ways to create a diagram. They all share the same rendering engine:

| Way | Use it when | Input |
|---|---|---|
| `NeuralNetwork` (low-level) | you want full control over each layer | layers added one by one |
| `DrawNN` (high-level) | you want a quick diagram from a list of layer sizes | a list such as `[3, 4, 2]` plus options |

### Examples

<table>
<tr>
<th width="50%">Code</th>
<th width="50%">Output</th>
</tr>

<tr>
<td colspan="2"><b>Example 1: Custom PyTorch model (structure, labels, weights, highlighted path)</b></td>
</tr>
<tr>
<td valign="top" width="50%">

```python
import torch.nn as nn
from neural_network_draw import DrawNN, draw_model


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(6, 6)
        self.act2 = nn.Tanh()
        self.out = nn.Linear(6, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.softmax(self.out(x))


def run():
    model = IrisNet()

    # Option 1: Structure only
    draw_model(model)

    # Option 2: With labels, weights, and a highlighted path
    DrawNN(
        model,
        input_labels=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        output_labels=["Setosa", "Versicolor", "Virginica"],
        highlight_path=[[0], [1, 2], [0], [1]],
    ).draw(show_weights=True)
```

</td>
<td valign="top" width="50%" align="center">
<img width="100%" alt="IrisNet with labels, weights and highlighted path" src="https://github.com/user-attachments/assets/2b52cc8f-ee90-405c-9354-d1345f4f3822" />
<br><sub><b>IrisNet: labels, weights and highlighted path</b></sub>
</td>
</tr>

<tr>
<td colspan="2"><b>Example 2: Compact model with a title (weights hidden)</b></td>
</tr>
<tr>
<td valign="top" width="50%">

```python
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.act1 = nn.Linear(4, 6), nn.ReLU()
        self.fc2, self.act2 = nn.Linear(6, 6), nn.Tanh()
        self.out, self.softmax = nn.Linear(6, 3), nn.Softmax(dim=1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.softmax(self.out(x))


def run():
    torch.manual_seed(0)
    DrawNN(
        IrisNet(),
        input_labels=["sepal_len", "sepal_wid", "petal_len", "petal_wid"],
        output_labels=["Setosa", "Versicolor", "Virginica"],
        highlight_path=[[0], [1, 2], [0], [1]],
        title="Iris classifier (from a PyTorch model)",
    ).draw(show_weights=False)
```

</td>
<td valign="top" width="50%" align="center">
<img width="100%" alt="Iris classifier drawn from a PyTorch model" src="https://github.com/user-attachments/assets/0de30ca6-4104-46f3-8693-3b73609e006e" />
<br><sub><b>Iris classifier with a title, weights hidden</b></sub>
</td>
</tr>

<tr>
<td colspan="2"><b>Example 3: Larger <code>nn.Sequential</code> model (weight colours and thickness)</b></td>
</tr>
<tr>
<td valign="top" width="50%">

```python
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


def run():
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(8, 10), nn.ReLU(),
        nn.Linear(10, 6), nn.ReLU(),
        nn.Linear(6, 3), nn.Softmax(dim=1),
    )
    DrawNN(model, title="8 -> 10 -> 6 -> 3").draw(show_weights=True)


if __name__ == "__main__":
    run()
```

</td>
<td valign="top" width="50%" align="center">
<img width="100%" alt="8-10-6-3 network with weight colours" src="https://github.com/user-attachments/assets/66bf218c-c134-42d2-8839-2e0259ece900" />
<br><sub><b>8 → 10 → 6 → 3: blue = positive, red = negative, thickness = size</b></sub>
</td>
</tr>
</table>

---
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

