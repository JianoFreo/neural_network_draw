from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.exists() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from neural_network_draw import DrawNN, Layer, NeuralNetwork, Neuron

__all__ = ["Neuron", "Layer", "NeuralNetwork", "DrawNN"]


if __name__ == "__main__":
    DrawNN([3, 5, 4, 2]).draw()

