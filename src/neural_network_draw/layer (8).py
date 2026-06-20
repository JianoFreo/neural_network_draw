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
                 biases=None, activation=None, labels=None,
                 horizontal_spacing=2.4, vertical_spacing=7.5, neuron_radius=0.5):
        self.vertical_distance_between_layers = vertical_spacing
        self.horizontal_distance_between_neurons = horizontal_spacing
        self.neuron_radius = neuron_radius
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer

        self.biases = biases
        self.activation = activation
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

    def leftmost_x(self):
        return min(n.x for n in self.neurons) if self.neurons else 0

    def rightmost_x(self):
        return max(n.x for n in self.neurons) if self.neurons else 0

    def __line_between_two_neurons(self, neuron1, neuron2, weight=None):
        dx = neuron2.x - neuron1.x
        dy = neuron2.y - neuron1.y
        angle = atan2(dx, dy)
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)

        line_color = "black"
        line_width = 1.0
        if weight is not None:
            line_color = "tab:blue" if weight >= 0 else "tab:red"
            line_width = min(0.5 + abs(weight) * 1.5, 4.0)

        line = plt.Line2D(
            (neuron1.x - x_adjustment, neuron2.x + x_adjustment),
            (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
            color=line_color, linewidth=line_width, alpha=0.85, zorder=1,
        )
        plt.gca().add_line(line)
        return line_color

    def __place_weight_labels(self, connections, min_gap):
        """
        connections: list of dicts with x1,y1,x2,y2,weight,color (one per edge
        that should show a weight value).
        Greedy collision avoidance: for each connection, try candidate t
        positions along the line (instead of always the midpoint) and pick
        the first one whose label center is at least `min_gap` away from
        every already-placed label. This is what actually stops the "-0.82 /
        0.78" style overlaps -- a formula-only spread isn't enough once a
        layer has many crossing connections converging in the same area.
        """
        placed = []
        candidate_ts = [0.5, 0.35, 0.65, 0.25, 0.75, 0.42, 0.58, 0.3, 0.7]

        for conn in connections:
            best_pt = None
            for t in candidate_ts:
                tx = conn["x1"] + (conn["x2"] - conn["x1"]) * t
                ty = conn["y1"] + (conn["y2"] - conn["y1"]) * t
                if all(((tx - px) ** 2 + (ty - py) ** 2) ** 0.5 >= min_gap for px, py in placed):
                    best_pt = (tx, ty)
                    break
            if best_pt is None:
                # Nothing was far enough; fall back to the candidate that
                # maximizes distance from the nearest existing label.
                best_pt, best_dist = None, -1
                for t in candidate_ts:
                    tx = conn["x1"] + (conn["x2"] - conn["x1"]) * t
                    ty = conn["y1"] + (conn["y2"] - conn["y1"]) * t
                    nearest = min(
                        (((tx - px) ** 2 + (ty - py) ** 2) ** 0.5 for px, py in placed),
                        default=min_gap,
                    )
                    if nearest > best_dist:
                        best_dist = nearest
                        best_pt = (tx, ty)
            placed.append(best_pt)
            plt.text(
                best_pt[0], best_pt[1],
                f"{conn['weight']:.2f}",
                fontsize=6.5, color=conn["color"], ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=conn["color"], linewidth=0.4, alpha=0.97),
                zorder=3,
            )

    def draw(self, layer_type=0, weights=None, show_weights=False):
        has_activation = self.activation is not None
        below_label_distance = 0.75 if (has_activation and layer_type == -1) else 0.35

        n = len(self.neurons)
        prev_n = len(self.previous_layer.neurons) if self.previous_layer else 0
        # Weight VALUES (the little numbers on each line) only get drawn up
        # to a connection-count threshold; past that even with collision
        # avoidance they'd need to be too small to read. Color and
        # line-thickness still encode every weight's sign/magnitude
        # regardless of layer size.
        draw_weight_text = show_weights and (prev_n * n) <= 24

        connections_to_label = []

        for j, neuron in enumerate(self.neurons):
            bias = self.biases[j] if self.biases is not None else None
            neuron.draw(self.neuron_radius, bias=bias, activation=self.activation)

            if self.previous_layer:
                for i, prev_neuron in enumerate(self.previous_layer.neurons):
                    weight = None
                    if weights is not None:
                        try:
                            weight = weights[i][j]
                        except (IndexError, TypeError):
                            weight = None
                    color = self.__line_between_two_neurons(neuron, prev_neuron, weight=weight)
                    if draw_weight_text and weight is not None:
                        connections_to_label.append({
                            "x1": prev_neuron.x, "y1": prev_neuron.y,
                            "x2": neuron.x, "y2": neuron.y,
                            "weight": weight, "color": color,
                        })

        if connections_to_label:
            self.__place_weight_labels(connections_to_label, min_gap=self.neuron_radius * 1.6)

        # Input-layer feature labels: dedicated column to the LEFT of the
        # whole network. Labels are stacked vertically (one row per neuron,
        # evenly spaced) so they never collide with each other no matter how
        # many input neurons there are, and a straight leader line connects
        # each label to its actual neuron. Sitting to the side (not above
        # the layer) also means they can never collide with the title.
        if layer_type == 0 and self.labels is not None:
            label_x = self.leftmost_x() - self.neuron_radius - 1.6
            n_labels = len(self.neurons)
            row_height = max(0.7, self.neuron_radius * 2.4)
            # Center the label column vertically around the layer's y.
            top_y = self.y + row_height * (n_labels - 1) / 2
            for j, neuron in enumerate(self.neurons):
                if j < len(self.labels) and self.labels[j] is not None:
                    label_y = top_y - row_height * j
                    neuron.draw_side_label(self.neuron_radius, self.labels[j], label_x, label_y)

        # Output-layer class labels stay rotated underneath (there's no risk
        # of colliding with a title down there).
        if layer_type == -1 and self.labels is not None:
            for j, neuron in enumerate(self.neurons):
                if j < len(self.labels) and self.labels[j] is not None:
                    neuron.draw_below_label(self.neuron_radius, self.labels[j], below_label_distance)

        # Layer-type label ("Input Layer" / "Hidden Layer N" / "Output Layer")
        x_text = self.rightmost_x() + self.neuron_radius + 1.2
        if layer_type == 0:
            plt.text(x_text, self.y, 'Input Layer', fontsize=12, va="center")
        elif layer_type == -1:
            plt.text(x_text, self.y, 'Output Layer', fontsize=12, va="center")
        else:
            plt.text(x_text, self.y, f'Hidden Layer {layer_type}', fontsize=12, va="center")
