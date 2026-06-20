try:
    from ._deps import plt
except ImportError:
    from _deps import plt

try:
    from .layer import Layer
except ImportError:
    from layer import Layer


class NeuralNetwork:
    def __init__(self, number_of_neurons_in_widest_layer, title="Neural Network Architecture"):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        self.layer_weights = []  # weights[i] connects layer i-1 -> layer i (None for layer 0)
        self.show_weights = False
        self.title = title

    def add_layer(self, number_of_neurons, weights=None, biases=None, activation=None, labels=None):
        """
        number_of_neurons: int, size of this layer
        weights: optional 2D list [prev_layer_size][number_of_neurons] of weights connecting
                 the previous layer to this one (ignored for the first layer added)
        biases: optional list of length number_of_neurons with a bias value per neuron
        activation: optional string, e.g. "ReLU", "Sigmoid", shown under each neuron
        labels: optional list of length number_of_neurons with per-neuron labels
                (input feature names for the input layer, class names for the output layer)
        """
        layer = Layer(
            self, number_of_neurons, self.number_of_neurons_in_widest_layer,
            biases=biases, activation=activation, labels=labels,
        )
        self.layers.append(layer)
        self.layer_weights.append(weights)

    def draw(self, show_weights=False, figsize=(10, 6)):
        plt.figure(figsize=figsize)

        for i, layer in enumerate(self.layers):
            layer_type = -1 if i == len(self.layers) - 1 else i
            layer.draw(layer_type, weights=self.layer_weights[i], show_weights=show_weights)

        plt.axis('scaled')
        plt.axis('off')
        plt.title(self.title, fontsize=15)
        plt.show()
