from .drawNN import DrawNN
from .layer import Layer
from .neuralNetwork import NeuralNetwork
from .neuron import Neuron


def draw_model(model, show_weights=False, figsize=None, **kwargs):
    """One-liner: draw a PyTorch model. Extra kwargs go to DrawNN (title, input_labels, ...)."""
    DrawNN(model, **kwargs).draw(show_weights=show_weights, figsize=figsize)


__all__ = ["Neuron", "Layer", "NeuralNetwork", "DrawNN", "draw_model"]
