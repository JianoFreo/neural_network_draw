try:
    from ._deps import plt
except ImportError:
    from _deps import plt


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius, bias=None, activation=None, label=None,
              face_color="white", edge_color="black"):
        circle = plt.Circle(
            (self.x, self.y),
            radius=neuron_radius,
            fill=True,
            facecolor=face_color,
            edgecolor=edge_color,
            zorder=4,
        )
        plt.gca().add_patch(circle)

        # Feature / class label drawn to the left of the neuron
        if label is not None:
            plt.text(
                self.x - neuron_radius - 0.3, self.y,
                str(label), fontsize=9, ha="right", va="center", zorder=5,
            )

        # Activation function name shown just under the neuron
        if activation is not None:
            plt.text(
                self.x, self.y - neuron_radius - 0.35,
                str(activation), fontsize=7, ha="center", va="top",
                style="italic", color="dimgray", zorder=5,
            )

        # Bias value shown inside/near the neuron
        if bias is not None:
            plt.text(
                self.x, self.y,
                f"{bias:.2f}" if isinstance(bias, (int, float)) else str(bias),
                fontsize=7, ha="center", va="center", zorder=5,
            )