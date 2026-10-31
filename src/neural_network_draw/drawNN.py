try:
    from .neuralNetwork import NeuralNetwork
except ImportError:
    from neuralNetwork import NeuralNetwork

try:
    from ._convert import is_torch_module, to_plain
    from .torch_support import extract_architecture
except ImportError:
    from _convert import is_torch_module, to_plain
    from torch_support import extract_architecture

DEFAULT_TITLE = "Neural Network Architecture"


class DrawNN:
    def __init__(self, neural_network, weights=None, biases=None, activations=None,
                 input_labels=None, output_labels=None, title=DEFAULT_TITLE,
                 highlight_path=None):
        """
        neural_network: EITHER a list of ints, e.g. [3, 4, 4, 2] (sizes of each layer),
                 OR a PyTorch model (torch.nn.Module). With a model, the layer sizes,
                 weights, biases and activations are read from it automatically
                 (see DrawNN.from_pytorch). Anything you pass explicitly
                 (weights, biases, activations, labels, ...) overrides what was read.
        weights: optional list of 2D lists, one per connection between consecutive layers,
                 e.g. weights[0] connects layer 0 -> layer 1, shaped [size0][size1]
        biases: optional list of lists, one list of biases per layer (skip/None for input layer)
        activations: optional list of strings, one activation name per layer
                     (skip/None for the input layer, which has no activation)
        input_labels: optional list of feature names for the input layer neurons
        output_labels: optional list of class names for the output layer neurons
        title: custom title shown above the diagram
        highlight_path: optional list of lists specifying which neurons to highlight per layer,
                       e.g. [[0], [1, 2], [0]] highlights specific neuron paths through the network
        """
        if is_torch_module(neural_network):
            arch = extract_architecture(neural_network)
            if title == DEFAULT_TITLE:
                title = f"{type(neural_network).__name__} Architecture"
            neural_network = arch["sizes"]
            weights = arch["weights"] if weights is None else weights
            biases = arch["biases"] if biases is None else biases
            activations = arch["activations"] if activations is None else activations

        # tensors / numpy arrays are accepted anywhere plain lists are
        self.neural_network = [int(n) for n in neural_network]
        self.weights = to_plain(weights)
        self.biases = to_plain(biases)
        self.activations = activations
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.title = title
        self.highlight_path = highlight_path

    @classmethod
    def from_pytorch(cls, model, **kwargs):
        """Build a DrawNN straight from a torch.nn.Module (same as DrawNN(model, ...))."""
        return cls(model, **kwargs)

    def draw(self, show_weights=False, figsize=None):
        widest_layer = max(self.neural_network)
        network = NeuralNetwork(widest_layer, title=self.title)

        n_layers = len(self.neural_network)
        for idx, layer_size in enumerate(self.neural_network):
            layer_weights = None
            if self.weights is not None and idx > 0 and idx - 1 < len(self.weights):
                layer_weights = self.weights[idx - 1]

            layer_biases = None
            if self.biases is not None and idx < len(self.biases):
                layer_biases = self.biases[idx]

            layer_activation = None
            if self.activations is not None and idx < len(self.activations):
                layer_activation = self.activations[idx]

            layer_labels = None
            if idx == 0 and self.input_labels is not None:
                layer_labels = self.input_labels
            elif idx == n_layers - 1 and self.output_labels is not None:
                layer_labels = self.output_labels

            network.add_layer(
                layer_size,
                weights=layer_weights,
                biases=layer_biases,
                activation=layer_activation,
                labels=layer_labels,
            )

        if self.highlight_path is not None:
            network.set_highlight_path(self.highlight_path)

        network.draw(show_weights=show_weights, figsize=figsize)
