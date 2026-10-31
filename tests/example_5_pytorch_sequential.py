"""Draw a PyTorch nn.Sequential model directly -- no arrays to build by hand."""
import torch.nn as nn

from neural_network_draw import DrawNN


def run():
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
        nn.Softmax(dim=1),
    )

    # sizes, weights, biases and activations are all read from the model
    DrawNN(model).draw(show_weights=True)


if __name__ == "__main__":
    run()
