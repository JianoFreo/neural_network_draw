try:
    from ._deps import atan2, cos, plt, sin
except ImportError:
    from _deps import atan2, cos, plt, sin

try:
    from .neuron import Neuron
except ImportError:
    from neuron import Neuron


class Layer:
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer,
                 biases=None, activation=None, labels=None):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer

        # Optional per-neuron biases for this layer (list of numbers, len == number_of_neurons)
        self.biases = biases
        # Optional activation function name shown under each neuron in this layer (e.g. "ReLU")
        self.activation = activation
        # Optional per-neuron labels (input feature names or output class names)
        self.labels = labels

        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__initialize_neurons(number_of_neurons)

    def __initialize_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin(number_of_neurons)
        for _ in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (
            self.number_of_neurons_in_widest_layer - number_of_neurons
        ) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y - self.vertical_distance_between_layers
        return 0

    def __get_previous_layer(self, network):
        return network.layers[-1] if network.layers else None

    def __line_between_two_neurons(self, neuron1, neuron2, weight=None, show_weights=False):
        dx = neuron2.x - neuron1.x
        dy = neuron2.y - neuron1.y

        angle = atan2(dx, dy)  # safer than atan

        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        # Color/width connections by weight sign & magnitude when weights are supplied
        line_color = "black"
        line_width = 1.0
        if weight is not None:
            line_color = "tab:blue" if weight >= 0 else "tab:red"
            line_width = min(0.5 + abs(weight) * 1.5, 4.0)

        line = plt.Line2D(
            (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
            color=line_color,
            linewidth=line_width,
            zorder=1,
        )
        plt.gca().add_line(line)

        if show_weights and weight is not None:
            mid_x = (neuron1.x + neuron2.x) / 2
            mid_y = (neuron1.y + neuron2.y) / 2
            plt.text(
                mid_x, mid_y,
                f"{weight:.2f}",
                fontsize=6, color=line_color, ha="center", va="center",
                backgroundcolor="white", zorder=3,
            )

    def draw(self, layer_type=0, weights=None, show_weights=False):
        """
        weights: optional 2D list/array shaped [len(previous_layer.neurons)][len(self.neurons)]
                 giving the weight connecting each previous-layer neuron to each neuron
                 in this layer. Ignored if there is no previous layer.
        show_weights: if True, prints the weight value at the midpoint of each connection.
        """
        for j, neuron in enumerate(self.neurons):
            bias = self.biases[j] if self.biases is not None else None
            label = self.labels[j] if self.labels is not None else None
            neuron.draw(self.neuron_radius, bias=bias, activation=self.activation, label=label)

            if self.previous_layer:
                for i, prev_neuron in enumerate(self.previous_layer.neurons):
                    weight = None
                    if weights is not None:
                        try:
                            weight = weights[i][j]
                        except (IndexError, TypeError):
                            weight = None
                    self.__line_between_two_neurons(
                        neuron, prev_neuron, weight=weight, show_weights=show_weights
                    )

        # Layer-type label (Input / Hidden N / Output)
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons

        if layer_type == 0:
            plt.text(x_text, self.y, 'Input Layer', fontsize=12)
        elif layer_type == -1:
            plt.text(x_text, self.y, 'Output Layer', fontsize=12)
        else:
            plt.text(x_text, self.y, f'Hidden Layer {layer_type}', fontsize=12)
