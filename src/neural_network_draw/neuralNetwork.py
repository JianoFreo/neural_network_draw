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
        self.layer_specs = []  # (weights, biases, activation, labels) per layer
        self.title = title

    def add_layer(self, number_of_neurons, weights=None, biases=None, activation=None, labels=None):
        """
        number_of_neurons: int, size of this layer
        weights: optional 2D list [prev_layer_size][number_of_neurons] connecting the
                 previous layer to this one (ignored for the first layer added)
        biases: optional list of length number_of_neurons
        activation: optional string shown under each neuron, e.g. "ReLU"
        labels: optional list of length number_of_neurons (input feature names for the
                input layer, class names for the output layer)
        """
        self.layer_specs.append({
            "size": number_of_neurons,
            "weights": weights,
            "biases": biases,
            "activation": activation,
            "labels": labels,
        })

    def __build_layers(self, show_weights):
        # Spacing scales up automatically based on which features are active,
        # so a plain structure-only diagram stays compact while a diagram
        # showing weights/biases/activations/labels gets enough room for
        # everything to be readable without overlap.
        any_weights = any(spec["weights"] is not None for spec in self.layer_specs)
        any_activation = any(spec["activation"] is not None for spec in self.layer_specs)
        # Input-layer labels live in a dedicated left column (not above the
        # layer), so the figure needs extra left margin, not extra top margin.
        input_has_labels = bool(self.layer_specs) and self.layer_specs[0]["labels"] is not None

        horizontal_spacing = 2.0
        if any_weights and show_weights:
            horizontal_spacing = 2.8
        elif self.number_of_neurons_in_widest_layer >= 8:
            horizontal_spacing = 1.8  # keep very wide layers from becoming huge

        vertical_spacing = 6.0
        if any_weights and show_weights:
            vertical_spacing += 2.0   # room for weight-value text on connections
        if any_activation:
            vertical_spacing += 1.0   # room for activation labels under neurons

        self.layers = []
        for spec in self.layer_specs:
            layer = Layer(
                self, spec["size"], self.number_of_neurons_in_widest_layer,
                biases=spec["biases"], activation=spec["activation"], labels=spec["labels"],
                horizontal_spacing=horizontal_spacing, vertical_spacing=vertical_spacing,
            )
            self.layers.append(layer)

        return input_has_labels

    def draw(self, show_weights=False, figsize=None):
        input_has_labels = self.__build_layers(show_weights)

        if figsize is None:
            width = max(8, self.number_of_neurons_in_widest_layer * 1.6)
            if input_has_labels:
                width += 2.5  # room for the left-side input label column
            height = max(6, len(self.layers) * 2.6)
            figsize = (width, height)

        plt.figure(figsize=figsize)

        for i, layer in enumerate(self.layers):
            layer_type = -1 if i == len(self.layers) - 1 else i
            layer_weights = self.layer_specs[i]["weights"]
            layer.draw(layer_type, weights=layer_weights, show_weights=show_weights)

        plt.axis('scaled')
        plt.axis('off')
        plt.title(self.title, fontsize=15, pad=14)
        plt.tight_layout()
        plt.show()
