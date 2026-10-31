"""A larger model: weight numbers are hidden on big layers, but line colour
(blue = positive, red = negative) and thickness still show every weight."""
import torch
import torch.nn as nn

from neural_network_draw import DrawNN


def run():
    torch.manual_seed(7)
    model = nn.Sequential(
        nn.Linear(8, 10), nn.ReLU(),
        nn.Linear(10, 6), nn.ReLU(),
        nn.Linear(6, 3), nn.Softmax(dim=1),
    )
    DrawNN(model, title="8 -> 10 -> 6 -> 3").draw(show_weights=True)


if __name__ == "__main__":
    run()
