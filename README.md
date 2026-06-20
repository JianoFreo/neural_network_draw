
# neural_network_draw

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


---
#  Dependencies

* Python 3.8+
* matplotlib
* numpy (if used internally in layout logic)

---

#  Key Design Philosophy

This library is designed to be:

>  A **visual compiler for neural network architectures**

It transforms structured model definitions into **clean educational diagrams**.

---

#  Use Cases

* Visualize forward and backward propagation
* Explaining architectures in blogs
* ML presentations
* Debugging model structure

---


* Interactive zoom/hover tooltips
* SVG / web export (React-friendly)
* Auto-generation from PyTorch models
* Animation of forward pass
* Graph-based auto-layout improvements

---


