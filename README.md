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

## Install on Another Device

### Option 1: Install from Git repository

If this project is hosted on GitHub (or another Git host), users can install it directly:

```bash
python -m pip install "git+https://github.com/JianoFreo/neural-network-draw.git"
```

### Option 2: Install from a downloaded folder

If someone has a ZIP copy of this project, they can extract it, open a terminal in the project folder, and run:

```bash
python -m pip install .
```

### Option 3: Install from PyPI

After publishing a release, anyone can install with:

```bash
python -m pip install neural-network-draw
```

To publish a new release

```bash
python -m pip install build twine
python -m build
twine upload dist/*
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
