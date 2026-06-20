try:
    from ._deps import plt
except ImportError:
    from _deps import plt


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, bias=None, activation=None,
              face_color="white", edge_color="black"):
        circle = plt.Circle(
            (self.x, self.y),
            radius=neuron_radius,
            fill=True,
            facecolor=face_color,
            edgecolor=edge_color,
            linewidth=1.3,
            zorder=4,
        )
        plt.gca().add_patch(circle)

        # Bias value shown inside the neuron
        if bias is not None:
            bias_text = f"{bias:.2f}" if isinstance(bias, (int, float)) else str(bias)
            plt.text(
                self.x, self.y,
                bias_text,
                fontsize=7.5, ha="center", va="center", zorder=6,
                fontweight="bold", color="black",
            )

        # Activation function name shown just under the neuron
        if activation is not None:
            plt.text(
                self.x, self.y - neuron_radius - 0.22,
                str(activation), fontsize=7.5, ha="center", va="top",
                style="italic", fontweight="bold", color="dimgray", zorder=5,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor="none", alpha=0.85),
            )

    def draw_side_label(self, neuron_radius, label, label_x, label_y, ha="right"):
        """
        Draw a feature label at a fixed point in a dedicated left-side
        column (label_x, label_y) -- NOT next to the neuron's own y -- with
        a straight leader line connecting the label to this neuron. Layer
        staggers label_y for each neuron in the layer so labels never
        collide with each other regardless of how many neurons there are,
        and the whole column sits to the left of the network, clear of the
        title above.
        """
        plt.plot(
            [label_x, self.x], [label_y, self.y],
            color="gray", linewidth=0.7, linestyle=(0, (3, 2)), zorder=2,
        )
        plt.text(
            label_x, label_y,
            str(label), fontsize=9, ha=ha, va="center", zorder=5,
            fontweight="bold",
        )

    def draw_below_label(self, neuron_radius, label, distance):
        """Output/class label rotated under the neuron (used for output layer)."""
        plt.text(
            self.x, self.y - neuron_radius - distance,
            str(label), fontsize=9, ha="center", va="top",
            rotation=-40, rotation_mode="anchor", zorder=5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )
