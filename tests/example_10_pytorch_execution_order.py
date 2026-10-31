"""Layers defined in a different order than they are used.

By default layers are read in definition order, which would be wrong here, so the
library refuses (instead of drawing a wrong diagram) and tells you to pass
example_input. With example_input one forward pass reveals the real order.
"""
import torch
import torch.nn as nn

from neural_network_draw import DrawNN, UnsupportedModelError


class OutOfOrder(nn.Module):
    def __init__(self):
        super().__init__()
        self.second = nn.Linear(5, 2)      # defined first ...
        self.act = nn.ReLU()
        self.first = nn.Linear(3, 5)       # ... but used first

    def forward(self, x):
        return self.second(self.act(self.first(x)))


def run():
    torch.manual_seed(3)
    model = OutOfOrder()

    try:
        DrawNN(model)
    except UnsupportedModelError as e:
        print("Without example_input ->", e, "\n")

    DrawNN(model, example_input=torch.zeros(1, 3),
           title="Real execution order (traced with example_input)").draw(show_weights=True)


if __name__ == "__main__":
    run()
