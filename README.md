# neural-network-draw

Draw simple feed-forward neural network architecture diagrams with matplotlib.

## Features

- Draws neurons as circles and connections as lines
- Supports input, hidden, and output layers
- Minimal API: pass a list of layer sizes and call draw()

## Requirements

- Python 3.8+
- matplotlib 3.7+

## Installation

From this project folder:

```bash
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

## Install On Another Device

### Option 1: Install from Git repository

```bash
python -m pip install "git+https://github.com/YOUR_USERNAME/YOUR_REPO.git"
```

### Option 2: Install from a downloaded folder

Extract the project, open a terminal in the project folder, then run:

```bash
python -m pip install .
```

### Option 3: Install from PyPI (after publishing)

```bash
python -m pip install neural-network-draw
```

To publish:

```bash
python -m pip install build twine
python -m build
twine upload dist/*
```

## Quick Start

```python
from neural_netwrok_draw import DrawNN

network = DrawNN([3, 5, 4, 2])
network.draw()
```

## API

- DrawNN(neural_network)
	- neural_network: list of integers, for example [3, 5, 4, 2]
- DrawNN.draw()
	- Renders the network diagram using matplotlib

## Example

```python
from neural_netwrok_draw import DrawNN

# 4 layers: input(4), hidden(6, 6), output(1)
DrawNN([4, 6, 6, 1]).draw()
```

## Troubleshooting

- ImportError for matplotlib
	- Install dependency manually: python -m pip install matplotlib
- Package installs but import fails
	- Verify your import name is: neural_netwrok_draw
- No window appears on draw()
	- Ensure your environment supports GUI windows for matplotlib

## License

Add your license text or LICENSE file before publishing publicly.
