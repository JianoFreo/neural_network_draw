try:
    from .neuralNetwork import NeuralNetwork
except ImportError:
    from neuralNetwork import NeuralNetwork


class DrawNN:
    def __init__(self, neural_network, weights=None, biases=None, activations=None,
                 input_labels=None, output_labels=None, title="Neural Network Architecture"):
        """
        neural_network: list of ints, e.g. [3, 4, 4, 2] (sizes of each layer)
        weights: optional list of 2D lists, one per connection between consecutive layers,
                 e.g. weights[0] connects layer 0 -> layer 1, shaped [size0][size1]
        biases: optional list of lists, one list of biases per layer (skip/None for input layer)
        activations: optional list of strings, one activation name per layer
                     (skip/None for the input layer, which has no activation)
        input_labels: optional list of feature names for the input layer neurons
        output_labels: optional list of class names for the output layer neurons
        title: custom title shown above the diagram
        """
        self.neural_network = neural_network
        self.weights = weights
        self.biases = biases
        self.activations = activations
        self.input_labels = input_labels
        self.output_labels = output_labels
        self.title = title

    def draw(self, show_weights=False, figsize=(10, 6)):
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

        network.draw(show_weights=show_weights, figsize=figsize)
