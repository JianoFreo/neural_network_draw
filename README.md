# neural-network-draw

Draw basic feed-forward neural network architecture diagrams using matplotlib.

## Installation

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Quick Start

```python
from neural_network_draw import DrawNN

network = DrawNN([3, 5, 4, 2])
network.draw()
```

## CLI Usage

After installation, you can run:

```bash
python -m neural_network_draw 3 5 4 2
```

If no layer sizes are provided, a default network is drawn.
